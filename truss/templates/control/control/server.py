import asyncio
import os
import pathlib

import uvicorn
import yaml
from application import create_app

CONTROL_SERVER_PORT = int(os.environ.get("CONTROL_SERVER_PORT", "8080"))
INFERENCE_SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "8090"))


def _identify_python_executable_path() -> str:
    if "PYTHON_EXECUTABLE" in os.environ:
        return os.environ["PYTHON_EXECUTABLE"]

    raise RuntimeError("Unable to find python, make sure it's installed.")


class ControlServer:
    def __init__(
        self,
        python_executable_path: str,
        inf_serv_home: str,
        control_server_port: int,
        inference_server_port: int,
    ):
        super().__init__()
        self._python_executable_path = python_executable_path
        self._inf_serv_home = inf_serv_home
        self._control_server_port = control_server_port
        self._inference_server_port = inference_server_port

        config_path = pathlib.Path(self._inf_serv_home) / "config.yaml"
        if config_path.exists():
            self._config = yaml.safe_load(config_path.read_text())
        else:
            self._config = {}

    def run(self):
        application = create_app(
            {
                "inference_server_home": self._inf_serv_home,
                "inference_server_process_args": [
                    self._python_executable_path,
                    f"{self._inf_serv_home}/main.py",
                ],
                "control_server_host": "0.0.0.0",
                "control_server_port": self._control_server_port,
                "inference_server_port": self._inference_server_port,
            }
        )

        application.state.logger.info(
            f"Starting live reload server on port {self._control_server_port}"
        )

        extra_kwargs = {}
        if self._config:
            transport = self._config.get("runtime", {}).get("transport", {})
            if transport and transport.get("kind") == "websocket":
                if ping_interval_seconds := transport.get("ping_interval_seconds"):
                    extra_kwargs["ws_ping_interval"] = ping_interval_seconds
                if ping_timeout_seconds := transport.get("ping_timeout_seconds"):
                    extra_kwargs["ws_ping_timeout"] = ping_timeout_seconds

        cfg = uvicorn.Config(
            application,
            host=application.state.control_server_host,
            port=application.state.control_server_port,
            # We hard-code the http parser as h11 (the default) in case the user has
            # httptools installed, which does not work with our requests & version
            # of uvicorn.
            http="h11",
            **extra_kwargs,
        )
        cfg.setup_event_loop()

        server = uvicorn.Server(cfg)
        asyncio.run(server.serve())


if __name__ == "__main__":
    control_server = ControlServer(
        python_executable_path=_identify_python_executable_path(),
        inf_serv_home=os.environ["APP_HOME"],
        control_server_port=CONTROL_SERVER_PORT,
        inference_server_port=INFERENCE_SERVER_PORT,
    )
    control_server.run()
