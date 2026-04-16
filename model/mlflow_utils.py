from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow  # pyright: ignore[reportMissingImports]
import mlflow.sklearn  # pyright: ignore[reportMissingImports]
from mlflow.tracking import MlflowClient  # pyright: ignore[reportMissingImports]

import common


@dataclass(frozen=True)
class RegisteredModelSnapshot:
    model_name: str
    version: str
    run_id: str
    model_uri: str
    creation_timestamp: int | None


def _to_sqlite_uri(path: str) -> str:
    return f"sqlite:///{Path(path).resolve().as_posix()}"


def get_mlflow_tracking_uri() -> str:
    return _to_sqlite_uri(common.CONFIG["paths"]["mlflow_tracking_db"])


def get_mlflow_experiment_name() -> str:
    return common.CONFIG["mlflow"]["experiment_name"]


def get_registered_model_name() -> str:
    return common.CONFIG["mlflow"]["registered_model_name"]


def ensure_mlflow_storage() -> None:
    Path(common.CONFIG["paths"]["mlflow_dir"]).mkdir(parents=True, exist_ok=True)
    Path(common.CONFIG["paths"]["mlflow_artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    Path(common.CONFIG["paths"]["mlflow_tracking_db"]).parent.mkdir(parents=True, exist_ok=True)


def configure_mlflow() -> None:
    ensure_mlflow_storage()
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = get_mlflow_experiment_name()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = Path(common.CONFIG["paths"]["mlflow_artifacts_dir"]).resolve().as_uri()
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def get_mlflow_client() -> MlflowClient:
    configure_mlflow()
    return MlflowClient(tracking_uri=get_mlflow_tracking_uri())


def _snapshot_from_version(model_version) -> RegisteredModelSnapshot:
    return RegisteredModelSnapshot(
        model_name=model_version.name,
        version=str(model_version.version),
        run_id=model_version.run_id,
        model_uri=f"models:/{model_version.name}/{model_version.version}",
        creation_timestamp=model_version.creation_timestamp,
    )


def get_registered_model_snapshot(model_name: str, model_version: str | None = None) -> RegisteredModelSnapshot:
    client = get_mlflow_client()
    if model_version is None:
        versions = client.search_model_versions(f"name = '{model_name}'")
        if not versions:
            raise RuntimeError(f"Aucune version enregistrée pour le modèle {model_name}.")
        latest = max(versions, key=lambda version: int(version.version))
        return _snapshot_from_version(latest)

    version = client.get_model_version(model_name, model_version)
    return _snapshot_from_version(version)


def snapshot_to_metadata(model_name: str, snapshot: RegisteredModelSnapshot) -> dict[str, Any]:
    if snapshot.creation_timestamp is not None:
        model_mtime = datetime.fromtimestamp(snapshot.creation_timestamp / 1000, tz=timezone.utc).isoformat()
    else:
        model_mtime = datetime.now(timezone.utc).isoformat()

    return {
        "model_name": model_name,
        "model_version": snapshot.version,
        "model_path": snapshot.model_uri,
        "model_file_mtime": model_mtime,
    }


def load_registered_model(model_name: str, model_version: str | None = None):
    configure_mlflow()
    snapshot = get_registered_model_snapshot(model_name, model_version)
    model = mlflow.sklearn.load_model(snapshot.model_uri)
    return model, snapshot