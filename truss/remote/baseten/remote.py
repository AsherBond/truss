import enum
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import yaml
from requests import ReadTimeout
from truss.constants import PRODUCTION_ENVIRONMENT_NAME

if TYPE_CHECKING:
    from rich import console as rich_console
from truss.local.local_config_handler import LocalConfigHandler
from truss.remote.baseten import custom_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.core import (
    ChainDeploymentHandle,
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionId,
    archive_truss,
    create_chain,
    create_truss_service,
    exists_model,
    get_chain_id_by_name,
    get_dev_version,
    get_dev_version_from_versions,
    get_model_versions,
    get_prod_version_from_versions,
    upload_truss,
)
from truss.remote.baseten.error import ApiError, RemoteError
from truss.remote.baseten.service import BasetenService, URLConfig
from truss.remote.baseten.utils.transfer import base64_encoded_json_str
from truss.remote.truss_remote import TrussRemote
from truss.truss_config import ModelServer
from truss.truss_handle import TrussHandle
from truss.util.path import is_ignored, load_trussignore_patterns_from_truss_dir
from watchfiles import watch


class PatchStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILED = enum.auto()
    SKIPPED = enum.auto()


class PatchResult(NamedTuple):
    status: PatchStatus
    message: str


class BasetenRemote(TrussRemote):
    def __init__(self, remote_url: str, api_key: str, **kwargs):
        super().__init__(remote_url, **kwargs)
        self._auth_service = AuthService(api_key=api_key)
        self._api = BasetenApi(remote_url, self._auth_service)

    @property
    def api(self) -> BasetenApi:
        return self._api

    def create_chain(
        self,
        chain_name: str,
        chainlets: List[custom_types.ChainletData],
        publish: bool = False,
        environment: Optional[str] = None,
    ) -> ChainDeploymentHandle:
        if environment:
            # If we are promoting a model to an environment after deploy, it must be published.
            # Draft models cannot be promoted.
            publish = True
        # Returns tuple of (chain_id, chain_deployment_id)
        chain_id = get_chain_id_by_name(self._api, chain_name)
        return create_chain(
            self._api,
            chain_id=chain_id,
            chain_name=chain_name,
            chainlets=chainlets,
            is_draft=not publish,
            environment=environment,
        )

    def get_chainlets(
        self, chain_deployment_id: str
    ) -> List[custom_types.DeployedChainlet]:
        return [
            custom_types.DeployedChainlet(
                name=chainlet["name"],
                is_entrypoint=chainlet["is_entrypoint"],
                is_draft=chainlet["oracle_version"]["is_draft"],
                status=chainlet["oracle_version"]["current_model_deployment_status"][
                    "status"
                ],
                logs_url=URLConfig.chainlet_logs_url(
                    self.remote_url,
                    chainlet["chain"]["id"],
                    chain_deployment_id,
                    chainlet["id"],
                ),
                oracle_predict_url=URLConfig.invocation_url(
                    self._api.rest_api_url,
                    URLConfig.MODEL,
                    chainlet["oracle"]["id"],
                    chainlet["oracle_version"]["id"],
                    chainlet["oracle_version"]["is_draft"],
                ),
                oracle_name=chainlet["oracle"]["name"],
            )
            for chainlet in self._api.get_chainlets_by_deployment_id(
                chain_deployment_id
            )
        ]

    def push(  # type: ignore
        self,
        truss_handle: TrussHandle,
        model_name: str,
        publish: bool = True,
        trusted: bool = False,
        promote: bool = False,
        preserve_previous_prod_deployment: bool = False,
        deployment_name: Optional[str] = None,
        origin: Optional[custom_types.ModelOrigin] = None,
        environment: Optional[str] = None,
        chain_environment: Optional[str] = None,
        chainlet_name: Optional[str] = None,
        chain_name: Optional[str] = None,
    ) -> BasetenService:
        if model_name.isspace():
            raise ValueError("Model name cannot be empty")

        model_id = exists_model(self._api, model_name)

        gathered_truss = TrussHandle(truss_handle.gather())
        if gathered_truss.spec.model_server != ModelServer.TrussServer:
            publish = True

        if promote:
            environment = PRODUCTION_ENVIRONMENT_NAME

        if environment:
            # If there is a target environment, it must be published.
            # Draft models cannot be promoted.
            publish = True

        if not publish and deployment_name:
            raise ValueError(
                "Deployment name cannot be used for development deployment"
            )

        if not promote and preserve_previous_prod_deployment:
            raise ValueError(
                "preserve-previous-production-deployment can only be used "
                "with the '--promote' option"
            )

        if deployment_name and not re.match(r"^[0-9a-zA-Z_\-\.]*$", deployment_name):
            raise ValueError(
                "Deployment name must only contain alphanumeric, -, _ and . characters"
            )

        encoded_config_str = base64_encoded_json_str(
            gathered_truss._spec._config.to_dict()
        )

        temp_file = archive_truss(gathered_truss)
        s3_key = upload_truss(self._api, temp_file)

        model_id, model_version_id = create_truss_service(
            api=self._api,
            model_name=model_name,
            s3_key=s3_key,
            config=encoded_config_str,
            is_draft=not publish,
            model_id=model_id,
            is_trusted=trusted,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            deployment_name=deployment_name,
            origin=origin,
            environment=environment,
            chain_environment=chain_environment,
            chainlet_name=chainlet_name,
            chain_name=chain_name,
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not publish,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}/model_versions/{model_version_id}",
            truss_handle=truss_handle,
            api=self._api,
        )

    @staticmethod
    def _get_matching_version(model_versions: List[dict], published: bool) -> dict:
        if not published:
            # Return the development model version.
            dev_version = get_dev_version_from_versions(model_versions)
            if not dev_version:
                raise RemoteError(
                    "No development model found. Run `truss push` then try again."
                )
            return dev_version

        # Return the production deployment version.
        prod_version = get_prod_version_from_versions(model_versions)
        if not prod_version:
            raise RemoteError(
                "No production model found. Run `truss push --publish` then try again."
            )
        return prod_version

    @staticmethod
    def _get_service_url_path_and_model_ids(
        api: BasetenApi, model_identifier: ModelIdentifier, published: bool
    ) -> Tuple[str, str, str]:
        if isinstance(model_identifier, ModelVersionId):
            try:
                model_version = api.get_model_version_by_id(model_identifier.value)
            except ApiError:
                raise RemoteError(f"Model version {model_identifier.value} not found.")
            model_version_id = model_version["model_version"]["id"]
            model_id = model_version["model_version"]["oracle"]["id"]
            service_url_path = f"/model_versions/{model_version_id}"
            return service_url_path, model_id, model_version_id

        if isinstance(model_identifier, ModelName):
            model_id, model_versions = get_model_versions(api, model_identifier)
            model_version = BasetenRemote._get_matching_version(
                model_versions, published
            )
            model_version_id = model_version["id"]
            service_url_path = f"/model_versions/{model_version_id}"
        elif isinstance(model_identifier, ModelId):
            # TODO(helen): consider making this consistent with getting the
            # service via model_name / respect --published in service_url_path.
            try:
                model = api.get_model_by_id(model_identifier.value)
            except ApiError:
                raise RemoteError(f"Model {model_identifier.value} not found.")
            model_id = model["model"]["id"]
            model_version_id = model["model"]["primary_version"]["id"]
            service_url_path = f"/models/{model_id}"
        else:
            # Model identifier is of invalid type.
            raise RemoteError(
                "You must either be inside of a Truss directory, or provide "
                "--model-deployment or --model options."
            )

        return service_url_path, model_id, model_version_id

    def get_service(self, **kwargs) -> BasetenService:
        try:
            model_identifier = kwargs["model_identifier"]
        except KeyError:
            raise ValueError("Baseten Service requires a model_identifier")

        published = kwargs.get("published", False)
        (
            service_url_path,
            model_id,
            model_version_id,
        ) = self._get_service_url_path_and_model_ids(
            self._api, model_identifier, published
        )

        return BasetenService(
            model_id=model_id,
            model_version_id=model_version_id,
            is_draft=not published,
            api_key=self._auth_service.authenticate().value,
            service_url=f"{self._remote_url}{service_url_path}",
            api=self._api,
        )

    def sync_truss_to_dev_version_by_name(
        self,
        model_name: str,
        target_directory: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ) -> None:
        # verify that development deployment exists for given model name
        dev_version = get_dev_version(self._api, model_name)  # pylint: disable=protected-access
        if not dev_version:
            raise RemoteError(
                "No development model found. Run `truss push` then try again."
            )

        watch_path = Path(target_directory)
        truss_ignore_patterns = load_trussignore_patterns_from_truss_dir(watch_path)

        def watch_filter(_, path):
            return not is_ignored(
                Path(path),
                truss_ignore_patterns,
            )

        # disable watchfiles logger
        logging.getLogger("watchfiles.main").disabled = True

        console.print(f"🚰 Attempting to sync truss at '{watch_path}' with remote")
        self.patch(watch_path, truss_ignore_patterns, console, error_console)

        console.print(f"👀 Watching for changes to truss at '{watch_path}' ...")
        for _ in watch(watch_path, watch_filter=watch_filter, raise_interrupt=False):
            self.patch(watch_path, truss_ignore_patterns, console, error_console)

    def _patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: Optional["rich_console.Console"] = None,
    ) -> PatchResult:
        try:
            truss_handle = TrussHandle(watch_path)
        except yaml.parser.ParserError as e:
            return PatchResult(PatchStatus.FAILED, f"Unable to parse config file. {e}")
        except ValueError as e:
            return PatchResult(
                PatchStatus.FAILED,
                f"Error when reading truss from directory {watch_path}. {e}",
            )

        model_name = truss_handle.spec.config.model_name
        dev_version = get_dev_version(self._api, model_name)  # type: ignore
        if not dev_version:
            return PatchResult(
                PatchStatus.FAILED,
                f"No development deployment found for model: {model_name}.",
            )

        truss_hash = dev_version.get("truss_hash", None)
        truss_signature = dev_version.get("truss_signature", None)
        if not (truss_hash and truss_signature):
            return PatchResult(
                PatchStatus.FAILED,
                (
                    "Failed to inspect a running remote deployment to watch for "
                    "changes.  Ensure that there exists a running remote deployment"
                    " before attempting to watch for changes."
                ),
            )

        LocalConfigHandler.add_signature(truss_hash, truss_signature)
        try:
            patch_request = truss_handle.calc_patch(truss_hash, truss_ignore_patterns)
        except Exception as e:
            return PatchResult(PatchStatus.FAILED, f"Failed to calculate patch. {e}")
        if not patch_request:
            return PatchResult(
                PatchStatus.FAILED,
                "Failed to calculate patch. Change type might not be supported.",
            )

        if (
            patch_request.prev_hash == patch_request.next_hash
            or len(patch_request.patch_ops) == 0
        ):
            return PatchResult(
                PatchStatus.SKIPPED, "No changes observed, skipping patching."
            )
        try:
            if console:
                with console.status("Applying patch..."):
                    resp = self._api.patch_draft_truss(model_name, patch_request)
            else:
                resp = self._api.patch_draft_truss(model_name, patch_request)

        except ReadTimeout:
            return PatchResult(
                PatchStatus.FAILED, "Read Timeout when attempting to patch remote."
            )
        except Exception as e:
            return PatchResult(
                PatchStatus.FAILED, f"Failed to patch draft deployment. {e}"
            )
        if not resp["succeeded"]:
            needs_full_deploy = resp.get("needs_full_deploy", None)
            if needs_full_deploy:
                message = (
                    f"Model {model_name} is not able to be patched, "
                    f"use `truss push` to deploy."
                )
            else:
                message = (
                    f"Failed to patch. Server error: `{resp['error']}`. "
                    "Model left in original state."
                )
            return PatchResult(PatchStatus.FAILED, message)
        else:
            return PatchResult(
                PatchStatus.SUCCESS,
                resp.get(
                    "success_message",
                    f"Model {model_name} patched successfully.",
                ),
            )

    def patch(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ):
        result = self._patch(watch_path, truss_ignore_patterns)
        if result.status in (PatchStatus.SUCCESS, PatchStatus.SKIPPED):
            console.print(result.message, style="green")
        else:
            error_console.print(result.message)

    def patch_for_chainlet(
        self,
        watch_path: Path,
        truss_ignore_patterns: List[str],
    ) -> PatchResult:
        return self._patch(watch_path, truss_ignore_patterns, console=None)
