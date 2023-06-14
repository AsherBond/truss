import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient
from truss_common.patch.types import (  # noqa
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
)
from truss_server.control.application import create_app  # noqa
from truss_server.shared.serialization import (  # noqa
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)


@pytest.fixture
def truss_original_hash():
    return "1234"


@pytest.fixture
def app(truss_container_fs, truss_original_hash):
    with _env_var({"HASH_TRUSS": truss_original_hash}):
        inf_serv_home = truss_container_fs / "app"
        control_app = create_app(
            {
                "inference_server_home": inf_serv_home,
                "inference_server_process_args": ["python", "inference_server.py"],
                "control_server_host": "*",
                "control_server_port": 8081,
                "inference_server_port": 8082,
                "oversee_inference_server": False,
                "retry_attempt_max_timeout": 5,
                "pip_path": "pip",
            }
        )
        inference_server_controller = control_app.state.inference_server_controller
        try:
            inference_server_controller.start()
            yield control_app
        finally:
            inference_server_controller.stop()


@pytest.fixture()
def client(app):
    return TestClient(app)


def test_restart_server(client):
    resp = client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()

    # Try second restart
    resp = client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()


def test_patch_model_code_update_existing(app, client):
    mock_model_file_content = """
class Model:
    def predict(self, request):
        return {'prediction': [1]}
"""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="model.py",
            content=mock_model_file_content,
        ),
    )
    _verify_apply_patch_success(client, patch)
    with (app.state.inference_server_home / "model" / "model.py").open() as model_file:
        new_model_file_content = model_file.read()
    assert new_model_file_content == mock_model_file_content


@pytest.mark.parametrize("use_binary", [True, False])
def test_patch_model_code_update_predict_on_long_load_time(use_binary, client):
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
            action=Action.UPDATE,
            path="model.py",
            content=mock_model_file_content,
        ),
    )
    _verify_apply_patch_success(client, patch)
    if use_binary:
        resp = client.post(
            "/v1/models/model:predict_binary",
            data=truss_msgpack_serialize({}),
            headers={"Content-type": "application/octet-stream"},
        )
    else:
        resp = client.post("/v1/models/model:predict", json={})
    resp.status_code == 200
    if use_binary:
        resp_body = truss_msgpack_deserialize(resp.content)
    else:
        resp_body = resp.json()

    assert resp_body == {"prediction": [1]}


def test_patch_model_code_create_new(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="touched",
            content=empty_content,
        ),
    )
    _verify_apply_patch_success(client, patch)
    assert (app.state.inference_server_home / "model" / "touched").exists()


def test_patch_model_code_create_in_new_dir(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="new_directory/touched",
            content=empty_content,
        ),
    )
    _verify_apply_patch_success(client, patch)
    assert (
        app.state.inference_server_home / "model" / "new_directory" / "touched"
    ).exists()


def test_404(client):
    resp = client.post("/control/nonexitant")
    assert resp.status_code == 404


def test_invalid_patch(client):
    patch_request = {
        "hash": "dummy",
        "prev_hash": "invalid",
        "patches": [],
    }
    resp = client.post("/control/patch", json=patch_request)
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "inadmissible_patch"
    assert "msg" not in resp.json()


def test_unsupported_patch(client):
    unsupported_patch = {
        "type": "unsupported",
        "body": {},
    }
    resp = _apply_patches(client, [unsupported_patch])
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "unsupported_patch"


def test_patch_failed_recoverable(client):
    will_fail_patch = Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=Action.ADD, requirement="not_a_valid_python_requirement"
        ),
    )
    resp = _apply_patches(client, [will_fail_patch.to_dict()])
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "patch_failed_recoverable"


def test_patch_failed_unrecoverable(client):
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
    resp = _apply_patches(
        client, [will_pass_patch.to_dict(), will_fail_patch.to_dict()]
    )
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["type"] == "patch_failed_unrecoverable"


def _verify_apply_patch_success(client, patch: Patch):
    original_hash = client.get("/control/truss_hash").json()["result"]
    patch_request = {
        "hash": "dummy",
        "prev_hash": original_hash,
        "patches": [patch.to_dict()],
    }
    resp = client.post("/control/patch", json=patch_request)
    resp = _apply_patches(client, [patch.to_dict()])
    assert resp.status_code == 200
    assert "error" not in resp.json()
    assert "msg" in resp.json()


def _apply_patches(client, patches: List[dict]):
    original_hash = client.get("/control/truss_hash").json()["result"]
    patch_request = {
        "hash": "dummy",
        "prev_hash": original_hash,
        "patches": patches,
    }
    return client.post("/control/patch", json=patch_request)


@contextmanager
def _env_var(kvs: Dict[str, str]):
    orig_env = os.environ.copy()
    try:
        os.environ.update(kvs)
        yield
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


def _patch_slow_model(client):
    mock_model_file_content = """
class Model:
    def load(self):
        import time
        time.sleep(9) # longer than INFERENCE_SERVER_START_WAIT_SECS
    def predict(self, request):
        return {'prediction': [1]}
"""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="model.py",
            content=mock_model_file_content,
        ),
    )
    _verify_apply_patch_success(client, patch)


def test_patch_model_code_update_predict_model_not_ready(client):
    _patch_slow_model(client)
    resp = client.post("/v1/models/model:predict", json={})
    assert resp.status_code == 200
    assert resp.json()["error"]["type"] == "model_not_ready"
    assert (
        "Model has started running, but not ready yet." in resp.json()["error"]["msg"]
    )


def test_patch_model_code_update_predict_binary_model_not_ready(client):
    _patch_slow_model(client)
    resp = client.post(
        "/v1/models/model:predict_binary",
        data=truss_msgpack_serialize({}),
        headers={"Content-type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    deserialized_resp = truss_msgpack_deserialize(resp.content)
    assert type(deserialized_resp) == dict
    assert deserialized_resp["error"]["type"] == "model_not_ready"
    assert (
        "Model has started running, but not ready yet."
        in deserialized_resp["error"]["msg"]
    )