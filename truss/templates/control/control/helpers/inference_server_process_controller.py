import subprocess

from helpers.context_managers import current_directory


class InferenceServerProcessController:
    def __init__(self, inference_server_home, inference_server_process_args) -> None:
        self._inference_server_process = None
        self._inference_server_home = inference_server_home
        self._inference_server_process_args = inference_server_process_args

    def start(self):
        with current_directory(self._inference_server_home):
            self._inference_server_process = subprocess.Popen(
                self._inference_server_process_args
            )

    def stop(self):
        if self._inference_server_process is not None:
            # TODO(pankaj) send sigint wait and then kill
            poll = self._inference_server_process.poll()
            if poll is None:
                self._inference_server_process.kill()