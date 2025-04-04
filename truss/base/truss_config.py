import logging
import os
import sys
from dataclasses import _MISSING_TYPE, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import yaml

from truss.base.constants import (
    HTTP_PUBLIC_BLOB_BACKEND,
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
)
from truss.base.custom_types import ModelFrameworkType
from truss.base.errors import ValidationError
from truss.base.trt_llm_config import (
    ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION,
    TRTLLMConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMModel,
    TrussTRTLLMQuantizationType,
)
from truss.base.validation import (
    validate_cpu_spec,
    validate_memory_spec,
    validate_node_count,
    validate_python_executable_path,
    validate_secret_name,
    validate_secret_to_path_mapping,
)
from truss.util.requirements import parse_requirement_string

DEFAULT_MODEL_FRAMEWORK_TYPE = ModelFrameworkType.CUSTOM
DEFAULT_MODEL_TYPE = "Model"
DEFAULT_MODEL_MODULE_DIR = "model"
DEFAULT_BUNDLED_PACKAGES_DIR = "packages"
DEFAULT_MODEL_CLASS_FILENAME = "model.py"
DEFAULT_MODEL_CLASS_NAME = "Model"
DEFAULT_TRUSS_STRUCTURE_VERSION = "2.0"
DEFAULT_MODEL_INPUT_TYPE = "Any"
DEFAULT_PYTHON_VERSION = "py39"
DEFAULT_DATA_DIRECTORY = "data"
DEFAULT_EXAMPLES_FILENAME = "examples.yaml"
DEFAULT_SPEC_VERSION = "2.0"
DEFAULT_PREDICT_CONCURRENCY = 1
DEFAULT_STREAMING_RESPONSE_READ_TIMEOUT = 60
DEFAULT_ENABLE_TRACING_DATA = False  # This should be in sync with tracing.py.
DEFAULT_ENABLE_DEBUG_LOGS = False
DEFAULT_IS_WEBSOCKET_ENDPOINT = False
DEFAULT_CPU = "1"
DEFAULT_MEMORY = "2Gi"
DEFAULT_USE_GPU = False

DEFAULT_BLOB_BACKEND = HTTP_PUBLIC_BLOB_BACKEND

VALID_PYTHON_VERSIONS = ["py38", "py39", "py310", "py311"]

X = TypeVar("X")
Y = TypeVar("Y")


def transform_optional(x: Optional[X], fn: Callable[[X], Optional[Y]]) -> Optional[Y]:
    if x is None:
        return None

    return fn(x)


logger = logging.getLogger(__name__)


class Accelerator(Enum):
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    V100 = "V100"
    A100 = "A100"
    A100_40GB = "A100_40GB"
    H100 = "H100"
    H200 = "H200"
    H100_40GB = "H100_40GB"


@dataclass
class AcceleratorSpec:
    accelerator: Optional[Accelerator] = None
    count: int = 0

    def to_str(self) -> Optional[str]:
        if self.accelerator is None or self.count == 0:
            return None
        if self.count > 1:
            return f"{self.accelerator.value}:{self.count}"
        return self.accelerator.value

    @staticmethod
    def from_str(acc_spec: Optional[str]):
        if acc_spec is None:
            return AcceleratorSpec()
        parts = acc_spec.split(":")
        count = 1
        if len(parts) not in [1, 2]:
            raise ValidationError("`accelerator` does not match parsing requirements.")
        if len(parts) == 2:
            count = int(parts[1])
        try:
            acc = Accelerator[parts[0]]
        except KeyError as exc:
            raise ValidationError(f"Accelerator {acc_spec} not supported") from exc
        return AcceleratorSpec(accelerator=acc, count=count)


@dataclass
class ModelRepo:
    repo_id: str = ""
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None

    @staticmethod
    def from_dict(d):
        repo_id = d.get("repo_id")
        if repo_id is None or repo_id == "":
            raise ValueError("Repo ID for Hugging Face model cannot be empty.")
        revision = d.get("revision", None)

        allow_patterns = d.get("allow_patterns", None)
        ignore_pattenrs = d.get("ignore_patterns", None)

        return ModelRepo(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_pattenrs,
        )

    def to_dict(self, verbose=False):
        data = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "allow_patterns": self.allow_patterns,
            "ignore_patterns": self.ignore_patterns,
        }

        if not verbose:
            # only show changed values
            data = {k: v for k, v in data.items() if v is not None}

        return data


@dataclass
class ModelCache:
    models: List[ModelRepo] = field(default_factory=list)

    @staticmethod
    def from_list(items: List[Dict[str, str]]) -> "ModelCache":
        return ModelCache([ModelRepo.from_dict(item) for item in items])

    def to_list(self, verbose=False) -> List[Dict[str, str]]:
        return [model.to_dict(verbose=verbose) for model in self.models]


@dataclass
class CacheInternal:
    models: List[ModelRepo] = field(default_factory=list)

    @staticmethod
    def from_list(items: List[Dict[str, str]]) -> "CacheInternal":
        return CacheInternal([ModelRepo.from_dict(item) for item in items])

    def to_list(self) -> List[Dict[str, str]]:
        models = []
        for model in self.models:
            models.append({"repo_id": model.to_dict()["repo_id"]})
        return models


@dataclass
class HealthChecks:
    restart_check_delay_seconds: Optional[int] = None
    restart_threshold_seconds: Optional[int] = None
    stop_traffic_threshold_seconds: Optional[int] = None

    @staticmethod
    def from_dict(d):
        return HealthChecks(
            restart_check_delay_seconds=d.get("restart_check_delay_seconds"),
            restart_threshold_seconds=d.get("restart_threshold_seconds"),
            stop_traffic_threshold_seconds=d.get("stop_traffic_threshold_seconds"),
        )

    def to_dict(self):
        return {
            "restart_check_delay_seconds": self.restart_check_delay_seconds,
            "restart_threshold_seconds": self.restart_threshold_seconds,
            "stop_traffic_threshold_seconds": self.stop_traffic_threshold_seconds,
        }


@dataclass
class Runtime:
    predict_concurrency: int = DEFAULT_PREDICT_CONCURRENCY
    streaming_read_timeout: int = DEFAULT_STREAMING_RESPONSE_READ_TIMEOUT
    enable_tracing_data: bool = DEFAULT_ENABLE_TRACING_DATA
    enable_debug_logs: bool = DEFAULT_ENABLE_DEBUG_LOGS
    health_checks: HealthChecks = field(default_factory=HealthChecks)
    is_websocket_endpoint: bool = DEFAULT_IS_WEBSOCKET_ENDPOINT

    @staticmethod
    def from_dict(d):
        predict_concurrency = d.get("predict_concurrency", DEFAULT_PREDICT_CONCURRENCY)
        num_workers = d.get("num_workers", 1)
        if num_workers != 1:
            raise ValueError(
                "After truss 0.9.49 only 1 worker per server is allowed. "
                "For concurrency utilize asyncio, autoscaling replicas "
                "and as a last resort thread/process pools inside the "
                "truss model."
            )
        streaming_read_timeout = d.get(
            "streaming_read_timeout", DEFAULT_STREAMING_RESPONSE_READ_TIMEOUT
        )
        enable_tracing_data = d.get("enable_tracing_data", DEFAULT_ENABLE_TRACING_DATA)
        enable_debug_logs = d.get("enable_debug_logs", DEFAULT_ENABLE_DEBUG_LOGS)
        health_checks = HealthChecks.from_dict(d.get("health_checks", {}))
        is_websocket_endpoint = d.get(
            "is_websocket_endpoint", DEFAULT_IS_WEBSOCKET_ENDPOINT
        )

        return Runtime(
            predict_concurrency=predict_concurrency,
            streaming_read_timeout=streaming_read_timeout,
            enable_tracing_data=enable_tracing_data,
            enable_debug_logs=enable_debug_logs,
            health_checks=health_checks,
            is_websocket_endpoint=is_websocket_endpoint,
        )

    def to_dict(self):
        return {
            "predict_concurrency": self.predict_concurrency,
            "streaming_read_timeout": self.streaming_read_timeout,
            "enable_tracing_data": self.enable_tracing_data,
            "health_checks": self.health_checks.to_dict(),
        }


class ModelServer(Enum):
    """
    To determine the image builder path for trusses built from alternative server backends.
    This enum is also used to gate development deployments to BasetenRemote
    https://github.com/basetenlabs/truss/blob/7505c17a2ddd4a6fa626b9126772999dc8f3fa86/truss/remote/baseten/remote.py#L56-L57
    """

    TrussServer = "TrussServer"
    TRT_LLM = "TRT_LLM"


@dataclass
class Build:
    model_server: ModelServer = ModelServer.TrussServer
    arguments: Dict = field(default_factory=dict)
    secret_to_path_mapping: Dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        model_server = ModelServer[d.get("model_server", "TrussServer")]
        arguments = d.get("arguments", {})
        secret_to_path_mapping = d.get("secret_to_path_mapping", {})
        validate_secret_to_path_mapping(secret_to_path_mapping)
        if not isinstance(secret_to_path_mapping, dict):
            raise ValueError(
                "Please pass a valid mapping for `secret_to_path_mapping`."
            )

        return Build(
            model_server=model_server,
            arguments=arguments,
            secret_to_path_mapping=secret_to_path_mapping,
        )

    def to_dict(self):
        return obj_to_dict(self)


@dataclass
class Resources:
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    use_gpu: bool = DEFAULT_USE_GPU
    accelerator: AcceleratorSpec = field(default_factory=AcceleratorSpec)
    node_count: Optional[int] = None

    @staticmethod
    def from_dict(d):
        cpu = d.get("cpu", DEFAULT_CPU)
        validate_cpu_spec(cpu)
        memory = d.get("memory", DEFAULT_MEMORY)
        validate_memory_spec(memory)
        accelerator = AcceleratorSpec.from_str((d.get("accelerator", None)))
        use_gpu = d.get("use_gpu", DEFAULT_USE_GPU)
        if accelerator.accelerator is not None:
            use_gpu = True

        r = Resources(cpu=cpu, memory=memory, use_gpu=use_gpu, accelerator=accelerator)

        # only add node_count if not None. This helps keep
        # config generated by truss init concise.
        node_count = d.get("node_count")
        validate_node_count(node_count)
        r.node_count = node_count

        return r

    def to_dict(self):
        d = {
            "cpu": self.cpu,
            "memory": self.memory,
            "use_gpu": self.use_gpu,
            "accelerator": self.accelerator.to_str(),
        }
        if self.node_count is not None:
            d["node_count"] = self.node_count
        return d


@dataclass
class ExternalDataItem:
    """A piece of remote data, to be made available to the Truss at serving time.

    Remote data is downloaded and stored under Truss's data directory. Care should be taken
    to avoid conflicts. This will get precedence if there's overlap.
    """

    # Url to download the data from.
    # Currently only files are allowed.
    url: str

    # This should be path relative to data directory. This is where the remote
    # file will be downloaded.
    local_data_path: str

    # The backend used to download files
    backend: str = DEFAULT_BLOB_BACKEND

    # A name can be given to a data item for readability purposes. It's not used
    # in the download process.
    name: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "ExternalDataItem":
        url = d.get("url")
        if url is None or url == "":
            raise ValueError("URL of an external data item cannot be empty")
        local_data_path = d.get("local_data_path")
        if local_data_path is None or local_data_path == "":
            raise ValueError(
                "The `local_data_path` field of an external data item cannot be empty"
            )

        item = ExternalDataItem(
            url=d["url"],
            local_data_path=d["local_data_path"],
            name=d.get("name"),
            backend=d.get("backend", DEFAULT_BLOB_BACKEND),
        )
        return item

    def to_dict(self):
        d = {
            "url": self.url,
            "local_data_path": self.local_data_path,
            "backend": self.backend,
        }
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ExternalData:
    """[Experimental] External data is data that is not contained in the Truss folder.

    Typically this will be data stored remotely. This data is guaranteed to be made
    available under the data directory of the truss.
    """

    items: List[ExternalDataItem]

    @staticmethod
    def from_list(items: List[Dict[str, str]]) -> "ExternalData":
        return ExternalData([ExternalDataItem.from_dict(item) for item in items])

    def to_list(self) -> List[Dict[str, str]]:
        return [item.to_dict() for item in self.items]


class DockerAuthType(Enum):
    """
    This enum will express all of the types of registry
    authentication we support.
    """

    GCP_SERVICE_ACCOUNT_JSON = "GCP_SERVICE_ACCOUNT_JSON"


@dataclass
class DockerAuthSettings:
    """
    Provides information about how to authenticate to the docker registry containing
    the custom base image.
    """

    auth_method: DockerAuthType
    secret_name: str
    registry: Optional[str] = ""

    @staticmethod
    def from_dict(d: Dict[str, str]):
        auth_method = d.get("auth_method")
        secret_name = d.get("secret_name")

        if auth_method:
            # Capitalize the auth method so that we support this field passed
            # as "gcs_service_account".
            auth_method = auth_method.upper()

        if (
            not secret_name
            or not auth_method
            or auth_method not in [auth_type.value for auth_type in DockerAuthType]
        ):
            raise ValueError("Please provide a `secret_name`, and valid `auth_method`")

        return DockerAuthSettings(
            auth_method=DockerAuthType[auth_method],
            secret_name=secret_name,
            registry=d.get("registry"),
        )

    def to_dict(self):
        return {
            "auth_method": self.auth_method.value,
            "secret_name": self.secret_name,
            "registry": self.registry,
        }


@dataclass
class BaseImage:
    image: str = ""
    python_executable_path: str = ""
    docker_auth: Optional[DockerAuthSettings] = None

    @staticmethod
    def from_dict(d):
        image = d.get("image", "")
        python_executable_path = d.get("python_executable_path", "")
        docker_auth = d.get("docker_auth")
        validate_python_executable_path(python_executable_path)
        return BaseImage(
            image=image,
            python_executable_path=python_executable_path,
            docker_auth=(
                DockerAuthSettings.from_dict(docker_auth) if docker_auth else None
            ),
        )

    def to_dict(self):
        return {
            "image": self.image,
            "python_executable_path": self.python_executable_path,
            "docker_auth": transform_optional(
                self.docker_auth, lambda docker_auth: docker_auth.to_dict()
            ),
        }


@dataclass
class DockerServer:
    start_command: str
    server_port: int
    predict_endpoint: str
    readiness_endpoint: str
    liveness_endpoint: str

    @staticmethod
    def from_dict(d) -> "DockerServer":
        return DockerServer(
            start_command=d.get("start_command"),
            server_port=d.get("server_port"),
            predict_endpoint=d.get("predict_endpoint"),
            readiness_endpoint=d.get("readiness_endpoint"),
            liveness_endpoint=d.get("liveness_endpoint"),
        )

    def to_dict(self):
        return {
            "start_command": self.start_command,
            "server_port": self.server_port,
            "readiness_endpoint": self.readiness_endpoint,
            "liveness_endpoint": self.liveness_endpoint,
            "predict_endpoint": self.predict_endpoint,
        }


@dataclass
class TrussConfig:
    """
    `config.yaml` controls Truss config
    Args:
        description (str):
            Describe your model for documentation purposes.
        environment_variables (Dict[str, str]):
            <Warning>
            Do not store secret values directly in environment variables (or anywhere in the
            config file). See the `secrets` arg for information on properly managing secrets.
            </Warning>
            Any environment variables can be provided here as key value pairs and are exposed
            to the environment that the model executes in. Many Python libraries can be
            customized using environment variables, so this field can be quite handy in those
            scenarios.
            ```yaml
            environment_variables:
              ENVIRONMENT: Staging
              DB_URL: https://my_database.example.com/
            ```
        model_metadata (Dict[str, str]):
            Set any additional metadata in this catch-all field. The entire contents of the
            config file are available to the model at runtime, so this is a good place to
            store any custom information that model needs. For example, scikit-learn models
            include a flag here that indicates whether the model supports returning
            probabilities alongside predictions.
            ```yaml
            model_metadata:
              supports_predict_proba: true
            ```
        model_name (str):
            The model's name, for documentation purposes.
        requirements_file (str):
            Path of the requirements file with the required Python dependencies.
        requirements (List[str]):
            List the Python dependencies that the model depends on. The requirements should
            be provided in the
            [pip requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/),
            but as a yaml list. These requirements are installed after the ones from `requirements_file`.
            We strongly recommend pinning versions in your requirements.
            ```yaml
            requirements:
            - scikit-learn==1.0.2
            - threadpoolctl==3.0.0
            - joblib==1.1.0
            - numpy==1.20.3
            - scipy==1.7.3
            ```
        resources (Dict[str, str]):
            Specify model server runtime resources such as CPU, RAM and GPU.
            ```yaml
            resources:
              cpu: "3"
              memory: 14Gi
              use_gpu: true
              accelerator: A10G
              node_count: 2
            ```
        secrets (Dict[str, str]):
            <Warning>
            This field can be used to specify the keys for such secrets and dummy default
            values. ***Never store actual secret values in the config***. Dummy default
            values are instructive of what the actual values look like and thus act as
            documentation of the format.
            </Warning>
            A model may depend on certain secret values that can't be bundled with the model
            and need to be bound securely at runtime. For example, a model may need to download
            information from s3 and may need access to AWS credentials for that.
            ```yaml
            secrets:
              hf_access_token: "ACCESS TOKEN"
            ```
        system_packages (List[str]):
            Specify any system packages that you would typically install using `apt` on a Debian operating system.
            ```yaml
            system_packages:
            - ffmpeg
            - libsm6
            - libxext6
            ```
    Returns:
        `config.yaml` file which can be updated
    """

    model_framework: ModelFrameworkType = DEFAULT_MODEL_FRAMEWORK_TYPE
    model_type: str = DEFAULT_MODEL_TYPE
    model_name: Optional[str] = None

    model_module_dir: str = DEFAULT_MODEL_MODULE_DIR
    model_class_filename: str = DEFAULT_MODEL_CLASS_FILENAME
    model_class_name: str = DEFAULT_MODEL_CLASS_NAME

    data_dir: str = DEFAULT_DATA_DIRECTORY
    external_data: Optional[ExternalData] = None

    # Python types for what the model expects as input
    input_type: str = DEFAULT_MODEL_INPUT_TYPE
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    requirements_file: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    runtime: Runtime = field(default_factory=Runtime)
    build: Build = field(default_factory=Build)
    python_version: str = DEFAULT_PYTHON_VERSION
    examples_filename: str = DEFAULT_EXAMPLES_FILENAME
    secrets: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    bundled_packages_dir: str = DEFAULT_BUNDLED_PACKAGES_DIR
    external_package_dirs: List[str] = field(default_factory=list)
    live_reload: bool = False
    apply_library_patches: bool = True
    # spec_version is a version string
    spec_version: str = DEFAULT_SPEC_VERSION
    base_image: Optional[BaseImage] = None
    docker_server: Optional[DockerServer] = None
    model_cache: ModelCache = field(default_factory=ModelCache)
    trt_llm: Optional[TRTLLMConfiguration] = None
    build_commands: List[str] = field(default_factory=list)
    use_local_src: bool = False
    # internal
    cache_internal: CacheInternal = field(default_factory=CacheInternal)

    @property
    def canonical_python_version(self) -> str:
        return {"py311": "3.11", "py310": "3.10", "py39": "3.9", "py38": "3.8"}[
            self.python_version
        ]

    @property
    def parsed_trt_llm_build_configs(self) -> List[TrussTRTLLMBuildConfiguration]:
        if self.trt_llm:
            if self.trt_llm.build.speculator and self.trt_llm.build.speculator.build:
                return [self.trt_llm.build, self.trt_llm.build.speculator.build]
            return [self.trt_llm.build]
        return []

    @staticmethod
    def from_dict(d):
        config = TrussConfig(
            spec_version=d.get("spec_version", DEFAULT_SPEC_VERSION),
            model_type=d.get("model_type", DEFAULT_MODEL_TYPE),
            model_framework=ModelFrameworkType(
                d.get("model_framework", DEFAULT_MODEL_FRAMEWORK_TYPE.value)
            ),
            model_module_dir=d.get("model_module_dir", DEFAULT_MODEL_MODULE_DIR),
            model_class_filename=d.get(
                "model_class_filename", DEFAULT_MODEL_CLASS_FILENAME
            ),
            model_class_name=d.get("model_class_name", DEFAULT_MODEL_CLASS_NAME),
            data_dir=d.get("data_dir", DEFAULT_DATA_DIRECTORY),
            input_type=d.get("input_type", DEFAULT_MODEL_INPUT_TYPE),
            model_metadata=d.get("model_metadata", {}),
            requirements_file=d.get("requirements_file", None),
            requirements=d.get("requirements", []),
            system_packages=d.get("system_packages", []),
            environment_variables=_handle_env_vars(d.get("environment_variables", {})),
            resources=Resources.from_dict(d.get("resources", {})),
            runtime=Runtime.from_dict(d.get("runtime", {})),
            build=Build.from_dict(d.get("build", {})),
            python_version=d.get("python_version", DEFAULT_PYTHON_VERSION),
            model_name=d.get("model_name", None),
            examples_filename=d.get("examples_filename", DEFAULT_EXAMPLES_FILENAME),
            secrets=d.get("secrets", {}),
            description=d.get("description", None),
            bundled_packages_dir=d.get(
                "bundled_packages_dir", DEFAULT_BUNDLED_PACKAGES_DIR
            ),
            external_package_dirs=d.get("external_package_dirs", []),
            live_reload=d.get("live_reload", False),
            apply_library_patches=d.get("apply_library_patches", True),
            external_data=transform_optional(
                d.get("external_data"), ExternalData.from_list
            ),
            base_image=transform_optional(d.get("base_image"), BaseImage.from_dict),
            docker_server=transform_optional(
                d.get("docker_server"), DockerServer.from_dict
            ),
            model_cache=transform_optional(
                d.get("model_cache") or d.get("hf_cache") or [],  # type: ignore
                ModelCache.from_list,
            ),
            cache_internal=transform_optional(
                d.get("cache_internal") or [],  # type: ignore
                CacheInternal.from_list,
            ),
            trt_llm=transform_optional(
                d.get("trt_llm"), lambda x: (TRTLLMConfiguration(**x))
            ),
            build_commands=d.get("build_commands", []),
            use_local_src=d.get("use_local_src", False),
        )
        config.validate()
        return config

    def load_requirements_from_file(self, truss_dir: Path) -> List[str]:
        if self.requirements_file:
            requirements_path = truss_dir / self.requirements_file
            try:
                requirements = []
                with open(requirements_path) as f:
                    for line in f.readlines():
                        parsed_line = parse_requirement_string(line)
                        if parsed_line:
                            requirements.append(parsed_line)
                return requirements
            except Exception as e:
                logger.exception(
                    f"failed to read requirements file: {self.requirements_file}"
                )
                raise e
        return []

    @staticmethod
    def load_requirements_file_from_filepath(yaml_path: Path) -> List[str]:
        config = TrussConfig.from_yaml(yaml_path)
        return config.load_requirements_from_file(yaml_path.parent)

    @staticmethod
    def from_yaml(yaml_path: Path):
        if not os.path.isfile(yaml_path):
            raise ValueError(f"Expected a truss configuration file at {yaml_path}")
        with yaml_path.open() as yaml_file:
            raw_data = yaml.safe_load(yaml_file) or {}
            if "hf_cache" in raw_data:
                logger.warning(
                    """Warning: `hf_cache` is deprecated in favor of `model_cache`.
                    Everything will run as before, but if you are pulling weights from S3 or GCS, they will be
                    stored at /app/model_cache instead of /app/hf_cache as before."""
                )
            return TrussConfig.from_dict(raw_data)

    def write_to_yaml_file(self, path: Path, verbose: bool = True):
        with path.open("w") as config_file:
            yaml.dump(self.to_dict(verbose=verbose), config_file)

    def to_dict(self, verbose: bool = True):
        return obj_to_dict(self, verbose=verbose)

    def clone(self):
        return TrussConfig.from_dict(self.to_dict())

    def _validate_trt_llm_config(self) -> None:
        if self.trt_llm:
            if self.trt_llm.build.base_model != TrussTRTLLMModel.ENCODER:
                current_tags = self.model_metadata.get("tags", [])
                if (
                    OPENAI_COMPATIBLE_TAG in current_tags
                    and OPENAI_NON_COMPATIBLE_TAG in current_tags
                ):
                    raise ValueError(
                        f"TRT-LLM models should have either model_metadata['tags'] = ['{OPENAI_COMPATIBLE_TAG}'] or ['{OPENAI_NON_COMPATIBLE_TAG}']. "
                        f"Your current tags are both {current_tags}, which is invalid. Please remove one of the tags."
                    )
                elif not (
                    OPENAI_COMPATIBLE_TAG in current_tags
                    or OPENAI_NON_COMPATIBLE_TAG in current_tags
                ):
                    # only check this in engine-builder for catching old truss pushes and force them adopt the new tag.
                    message = f"""TRT-LLM models should have model_metadata['tags'] = ['{OPENAI_COMPATIBLE_TAG}'] (or ['{OPENAI_NON_COMPATIBLE_TAG}']).
                        Your current tags are {current_tags}, which is has neither option. We require a active choice to be made.

                        For making the model compatible with OpenAI clients, we require to add the following to your config.
                        ```yaml
                        model_metadata:
                        tags:
                        - {OPENAI_COMPATIBLE_TAG}
                        # for legacy behavior set above line to the following, which will break OpenAI compatibility explicitly.
                        # This was the old default behaviour if you used Baseten before March 19th 2025 or truss<=0.9.68
                        # `- {OPENAI_NON_COMPATIBLE_TAG}`
                        ```
                        """
                    if ENGINE_BUILDER_TRUSS_RUNTIME_MIGRATION:
                        raise ValueError(message)
                    else:
                        logger.warning(message)
                elif OPENAI_NON_COMPATIBLE_TAG in current_tags:
                    logger.warning(
                        f"Model is marked as {OPENAI_NON_COMPATIBLE_TAG}. This model will not be compatible with OpenAI clients directly. "
                        f"This is the deprecated legacy behavior, please update the tag to {OPENAI_COMPATIBLE_TAG}."
                    )

            if (
                self.trt_llm.build.quantization_type
                is TrussTRTLLMQuantizationType.WEIGHTS_ONLY_INT8
                and self.resources.accelerator.accelerator is Accelerator.A100
            ):
                logger.warning(
                    "Weight only int8 quantization on A100 accelerators is not recommended."
                )
            if self.resources.accelerator.accelerator in [
                Accelerator.T4,
                Accelerator.V100,
            ]:
                raise ValueError(
                    "TRT-LLM is not supported on CUDA_COMPUTE_75 (T4) and CUDA_COMPUTE_70 (V100) GPUs"
                    "the lowest supported CUDA compute capability is CUDA_COMPUTE_80 (A100) or A10G (CUDA_COMPUTE_86)"
                )
            elif self.trt_llm.build.quantization_type in [
                TrussTRTLLMQuantizationType.FP8,
                TrussTRTLLMQuantizationType.FP8_KV,
            ] and self.resources.accelerator.accelerator in [
                Accelerator.A10G,
                Accelerator.A100,
                Accelerator.A100_40GB,
            ]:
                raise ValueError(
                    "FP8 quantization is only supported on L4, H100, H200 accelerators or newer (CUDA_COMPUTE>=89)"
                )
            tensor_parallel_count = self.trt_llm.build.tensor_parallel_count

            if tensor_parallel_count != self.resources.accelerator.count:
                raise ValueError(
                    "Tensor parallelism and GPU count must be the same for TRT-LLM"
                )

    def validate(self):
        if self.python_version not in VALID_PYTHON_VERSIONS:
            raise ValueError(
                f"Please ensure that `python_version` is one of {VALID_PYTHON_VERSIONS}"
            )

        if not isinstance(self.secrets, dict):
            raise ValueError(
                "Please ensure that `secrets` is a mapping of the form:\n"
                "```\n"
                "secrets:\n"
                '  secret1: "some default value"\n'
                '  secret2: "some other default value"\n'
                "```"
            )
        for secret_name in self.secrets:
            validate_secret_name(secret_name)

        if self.requirements and self.requirements_file:
            raise ValueError(
                "Please ensure that only one of `requirements` and `requirements_file` is specified"
            )
        self._validate_trt_llm_config()


def _handle_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    new_env_vars = {}
    for k, v in env_vars.items():
        if isinstance(v, bool):
            new_env_vars[k] = str(v).lower()
        else:
            new_env_vars[k] = str(v)
    return new_env_vars


DATACLASS_TO_REQ_KEYS_MAP = {
    Resources: {"accelerator", "cpu", "memory", "use_gpu"},
    Runtime: {"predict_concurrency"},
    Build: {"model_server"},
    TrussConfig: {
        "environment_variables",
        "external_package_dirs",
        "model_metadata",
        "model_name",
        "python_version",
        "requirements",
        "resources",
        "secrets",
        "system_packages",
        "build_commands",
    },
    BaseImage: {"image", "python_executable_path"},
}


def obj_to_dict(obj, verbose: bool = False):
    """
    This function serializes a given object (usually starting with a TrussConfig) and
    only keeps required keys or ones changed by the user manually. This simplifies the config.yml.
    """
    required_keys = DATACLASS_TO_REQ_KEYS_MAP[type(obj)]
    d = {}
    for f in fields(obj):
        field_name = f.name
        field_default_value = f.default
        field_default_factory = f.default_factory

        field_curr_value = getattr(obj, f.name)

        expected_default_value = None
        if not isinstance(field_default_value, _MISSING_TYPE):
            expected_default_value = field_default_value
        else:
            expected_default_value = field_default_factory()  # type: ignore

        should_add_to_dict = (
            expected_default_value != field_curr_value or field_name in required_keys
        )

        if verbose or should_add_to_dict:
            if isinstance(field_curr_value, tuple(DATACLASS_TO_REQ_KEYS_MAP.keys())):
                d[field_name] = obj_to_dict(field_curr_value, verbose=verbose)
            elif isinstance(field_curr_value, AcceleratorSpec):
                d[field_name] = field_curr_value.to_str()
            elif isinstance(field_curr_value, Enum):
                d[field_name] = field_curr_value.value
            elif isinstance(field_curr_value, ExternalData):
                d["external_data"] = transform_optional(
                    field_curr_value, lambda data: data.to_list()
                )
            elif isinstance(field_curr_value, ModelCache):
                d["model_cache"] = transform_optional(
                    field_curr_value, lambda data: data.to_list(verbose=verbose)
                )
            elif isinstance(field_curr_value, CacheInternal):
                d["cache_internal"] = transform_optional(
                    field_curr_value, lambda data: data.to_list()
                )
            elif isinstance(field_curr_value, TRTLLMConfiguration):
                d["trt_llm"] = transform_optional(
                    field_curr_value, lambda data: data.to_json_dict(verbose=verbose)
                )
            elif isinstance(field_curr_value, BaseImage):
                d["base_image"] = transform_optional(
                    field_curr_value, lambda data: data.to_dict()
                )
            elif isinstance(field_curr_value, DockerServer):
                d["docker_server"] = transform_optional(
                    field_curr_value, lambda data: data.to_dict()
                )
            elif isinstance(field_curr_value, DockerAuthSettings):
                d["docker_auth"] = transform_optional(
                    field_curr_value, lambda data: data.to_dict()
                )
            elif isinstance(field_curr_value, HealthChecks):
                d["health_checks"] = transform_optional(
                    field_curr_value, lambda data: data.to_dict()
                )
            else:
                d[field_name] = field_curr_value

    return d


def _infer_python_version() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def map_local_to_supported_python_version() -> str:
    return _map_to_supported_python_version(_infer_python_version())


def _map_to_supported_python_version(python_version: str) -> str:
    """Map python version to truss supported python version.

    Currently, it maps any versions greater than 3.11 to 3.11.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
          py310.
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    if python_major_version != 3:
        raise NotImplementedError("Only python version 3 is supported")

    if python_minor_version > 11:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.11, the highest version that Truss currently supports."
        )
        return "py311"

    if python_minor_version < 8:
        raise ValueError(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.8, the lowest version that Truss currently supports."
        )

    return python_version
