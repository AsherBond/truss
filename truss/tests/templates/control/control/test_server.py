import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from truss.truss_handle.patch.custom_types import PatchRequest

# Needed to simulate the set up on the model docker container
sys.path.append(
    str(
        Path(__file__).parent.parent.parent.parent.parent
        / "templates"
        / "control"
        / "control"
    )
)

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "templates"))
sys.path.append(
    str(Path(__file__).parent.parent.parent.parent.parent / "templates" / "shared")
)

from truss.templates.control.control.application import create_app  # noqa
from truss.templates.control.control.helpers.custom_types import (  # noqa
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
)


@pytest.fixture
def truss_original_hash():
    return "1234"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@pytest.fixture
def ports():
    return {
        "control_server_port": _get_free_port(),
        "inference_server_port": _get_free_port(),
    }


@pytest.fixture
def app(truss_container_fs, truss_original_hash, ports):
    with _env_var({"HASH_TRUSS": truss_original_hash}):
        inf_serv_home = truss_container_fs / "app"
        control_app = create_app(
            {
                "inference_server_home": inf_serv_home,
                "inference_server_process_args": ["python", "main.py"],
                "control_server_host": "*",
                "control_server_port": ports["control_server_port"],
                "inference_server_port": ports["inference_server_port"],
                "oversee_inference_server": False,
                "pip_path": "pip",
            }
        )
        inference_server_controller = control_app.state.inference_server_controller
        try:
            inference_server_controller.start()
            yield control_app
        finally:
            inference_server_controller.stop()


@pytest.fixture(
    params=[
        pytest.param(("asyncio", {"use_uvloop": True}), id="asyncio+uvloop"),
        pytest.param(("asyncio", {"use_uvloop": False}), id="asyncio"),
    ]
)
def anyio_backend(request):
    return request.param


@pytest.fixture()
async def client(app, ports):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"http://localhost:{ports['control_server_port']}"
    ) as async_client:
        yield async_client


@pytest.mark.anyio
async def test_restart_server(client):
    resp = await client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()

    # Try second restart
    resp = await client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()


@pytest.mark.anyio
async def test_patch_model_code_update_existing(app, client):
    mock_model_file_content = """
class Model:
    def predict(self, request):
        return {'prediction': [1]}
"""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="model.py", content=mock_model_file_content
        ),
    )
    await _verify_apply_patch_success(client, patch)
    with (app.state.inference_server_home / "model" / "model.py").open() as model_file:
        new_model_file_content = model_file.read()
    assert new_model_file_content == mock_model_file_content


@pytest.mark.anyio
async def test_patch_model_code_update_predict_on_long_load_time(app, client):
    mock_model_file_content = """
class Model:
    def load(self):
        import time
        time.sleep(3)

    def predict(self, request):
        return {'prediction': [1]}
"""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="model.py", content=mock_model_file_content
        ),
    )
    await _verify_apply_patch_success(client, patch)
    resp = await client.post("/v1/models/model:predict", json={})
    resp.status_code == 200
    assert resp.json() == {"prediction": [1]}


@pytest.mark.anyio
async def test_patch_model_code_create_new(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="touched", content=empty_content
        ),
    )
    await _verify_apply_patch_success(client, patch)
    assert (app.state.inference_server_home / "model" / "touched").exists()


@pytest.mark.anyio
async def test_patch_model_code_create_in_new_dir(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE, path="new_directory/touched", content=empty_content
        ),
    )
    await _verify_apply_patch_success(client, patch)
    assert (
        app.state.inference_server_home / "model" / "new_directory" / "touched"
    ).exists()


@pytest.mark.anyio
async def test_404(client):
    resp = await client.post("/control/nonexitant")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_invalid_patch(client):
    patch_request = PatchRequest(hash="dummy", prev_hash="invalid", patches=[])
    resp = await client.post("/control/patch", json=patch_request.to_dict())
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "inadmissible_patch"
    assert "msg" not in resp.json()


@pytest.mark.anyio
async def test_patch_failed_recoverable(client):
    will_fail_patch = Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.ADD, requirement="not_a_valid_python_requirement"
        ),
    )
    resp = await _apply_patches(client, [will_fail_patch])
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "patch_failed_recoverable"


@pytest.mark.anyio
async def test_patch_failed_unrecoverable(client):
    will_pass_patch = Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(action=Action.ADD, requirement="requests"),
    )
    will_fail_patch = Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.ADD, requirement="not_a_valid_python_requirement"
        ),
    )
    resp = await _apply_patches(client, [will_pass_patch, will_fail_patch])
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "patch_failed_unrecoverable"


@pytest.mark.anyio
async def test_health_check(client):
    resp = await client.get("/v1/models/model")
    assert resp.status_code == 200
    assert resp.json() == {}


@pytest.mark.anyio
async def test_health_check_retries(client, app):
    async def mock_send(*args, **kwargs):
        return httpx.Response(
            status_code=503, json={"error": "Model with name model is not ready."}
        )

    app.state.proxy_client.send = AsyncMock(side_effect=mock_send)

    await client.get("/v1/models/model")

    # Health check did not retry
    assert app.state.proxy_client.send.call_count == 1


@pytest.mark.anyio
async def test_metrics(client):
    resp = await client.get("/metrics")
    # Redirect to /metrics/
    assert resp.status_code == 307
    # Follow redirect
    resp = await client.get("/metrics/")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_retries(client, app):
    app.state.proxy_client.send = AsyncMock(
        side_effect=[
            httpx.ConnectTimeout("Connect timeout"),
            httpx.ReadTimeout("Read timeout"),
            httpx.ReadError("Read error"),
            httpx.ConnectError("Connect error"),
            httpx.RemoteProtocolError("Remote protocol error"),
        ]
    )

    with (
        patch("endpoints.INFERENCE_SERVER_START_WAIT_SECS", new=4),
        pytest.raises(httpx.RemoteProtocolError),
    ):
        await client.get("/v1/models/model")

    # We should have made 5 attempts
    assert app.state.proxy_client.send.call_count == 5


async def _verify_apply_patch_success(client, patch: Patch):
    resp = await client.get("/control/truss_hash")
    original_hash = resp.json()["result"]
    print(f"ORIGINAL HASH: {original_hash}")
    patch_request = PatchRequest(hash="dummy", prev_hash=original_hash, patches=[patch])
    resp = await client.post("/control/patch", json=patch_request.to_dict())
    resp = await _apply_patches(client, [patch])
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()


async def _apply_patches(client, patches: List[Patch]):
    resp = await client.get("/control/truss_hash")
    original_hash = resp.json()["result"]
    patch_request = PatchRequest(hash="dummy", prev_hash=original_hash, patches=patches)
    return await client.post("/control/patch", json=patch_request.to_dict())


@contextmanager
def _env_var(kvs: Dict[str, str]):
    orig_env = os.environ.copy()
    try:
        os.environ.update(kvs)
        yield
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
