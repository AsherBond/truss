import asyncio
import logging
import re
from pathlib import Path
from typing import Dict

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from truss.server.control.endpoints import control_app
from truss.server.control.helpers.errors import ModelLoadFailed, PatchApplicatonError
from truss.server.control.helpers.inference_server_controller import (
    InferenceServerController,
)
from truss.server.control.helpers.inference_server_process_controller import (
    InferenceServerProcessController,
)
from truss.server.control.helpers.inference_server_starter import (
    async_inference_server_startup_flow,
)
from truss.server.control.helpers.truss_patch.model_container_patch_applier import (
    ModelContainerPatchApplier,
)
from truss.server.control.patch_appliers.model_container_patch_applier import (
    ModelContainerPatchApplier,
)
from truss.server.shared.logging import setup_logging


async def handle_patch_error(_, exc):
    error_type = _camel_to_snake_case(type(exc).__name__)
    return JSONResponse(
        content={
            "error": {
                "type": error_type,
                "msg": str(exc),
            }
        }
    )


async def generic_error_handler(_, exc):
    return JSONResponse(
        content={
            "error": {
                "type": "unknown",
                "msg": f"{type(exc)}: {exc}",
            }
        }
    )


async def handle_model_load_failed(_, error):
    # Model load failures should result in 503 status
    return JSONResponse({"error": str(error)}, 503)


def create_app(base_config: Dict):
    app_state = State()

    setup_logging()

    app_logger = logging.getLogger(__name__)

    app_state.logger = app_logger

    for k, v in base_config.items():
        setattr(app_state, k, v)

    app_state.inference_server_process_controller = InferenceServerProcessController(
        app_state.inference_server_home,
        app_state.inference_server_process_args,
        app_state.inference_server_port,
        app_logger=app_logger,
    )

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=32)

    app_state.proxy_client = httpx.AsyncClient(
        base_url=f"http://localhost:{app_state.inference_server_port}", limits=limits
    )

    pip_path = getattr(app_state, "pip_path", None)

    patch_applier = ModelContainerPatchApplier(
        Path(app_state.inference_server_home),
        app_logger,
        pip_path,
    )

    oversee_inference_server = getattr(app_state, "oversee_inference_server", True)

    app_state.inference_server_controller = InferenceServerController(
        app_state.inference_server_process_controller,
        patch_applier,
        app_logger,
        oversee_inference_server,
    )

    async def start_background_inference_startup():
        asyncio.create_task(
            async_inference_server_startup_flow(
                app_state.inference_server_controller,
                app_logger,
            )
        )

    app = FastAPI(
        title="Truss Live Reload Server",
        on_startup=[start_background_inference_startup],
        exception_handlers={
            PatchApplicatonError: handle_patch_error,
            ModelLoadFailed: handle_model_load_failed,
            Exception: generic_error_handler,
        },
    )
    app.state = app_state

    app.include_router(control_app)
    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
