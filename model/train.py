from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlflow  # pyright: ignore[reportMissingImports]
import mlflow.sklearn  # pyright: ignore[reportMissingImports]
from mlflow.models.signature import infer_signature  # pyright: ignore[reportMissingImports]
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import common
from model.features import build_model_features
from model.load_data import load_test_data, load_train_data
from model.mlflow_utils import (
    configure_mlflow,
    get_registered_model_name,
    get_registered_model_snapshot,
    snapshot_to_metadata,
)

MODEL_PATH = common.CONFIG["paths"]["model_path"]
RANDOM_STATE = common.CONFIG["ml"]["random_state"]

FULL_NUM_FEATURES = [
    "log_distance_haversine",
    "hour",
    "abnormal_period",
    "is_high_traffic_trip",
    "is_high_speed_trip",
    "is_rare_pickup_point",
    "is_rare_dropoff_point",
    "vendor_id",
    "store_and_fwd_flag",
    "passenger_count",
]
TRIMMED_NUM_FEATURES = [
    "log_distance_haversine",
    "hour",
    "is_high_traffic_trip",
    "is_high_speed_trip",
    "vendor_id",
    "store_and_fwd_flag",
    "passenger_count",
]
CAT_FEATURES = ["weekday", "month"]


@dataclass(frozen=True)
class CandidateRun:
    run_name: str
    estimator_name: str
    estimator_factory: Callable[[], object]
    numeric_features: list[str]
    scale_numeric: bool
    alpha: float | None = None


def build_pipeline(candidate: CandidateRun) -> Pipeline:
    numeric_transformer = StandardScaler() if candidate.scale_numeric else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("numeric", numeric_transformer, candidate.numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(build_model_features, validate=False)),
            ("preprocess", preprocessor),
            ("regression", candidate.estimator_factory()),
        ]
    )


def evaluate_model(trained_model: Pipeline, X, y, label: str) -> float:
    predictions = trained_model.predict(X)
    score = root_mean_squared_error(y, predictions)
    print(f"RMSE on {label} data: {score:.4f}")
    return score


def persist_model(trained_model: Pipeline, path: str) -> None:
    print(f"Persisting the model to {path}")
    model_dir = Path(path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(trained_model, file)
    print("Done")


def _build_candidate_runs() -> list[CandidateRun]:
    return [
        CandidateRun(
            run_name="ridge_scaled_full",
            estimator_name="ridge",
            estimator_factory=lambda: Ridge(alpha=1.0),
            numeric_features=FULL_NUM_FEATURES,
            scale_numeric=True,
            alpha=1.0,
        ),
        CandidateRun(
            run_name="ridge_scaled_trimmed",
            estimator_name="ridge",
            estimator_factory=lambda: Ridge(alpha=0.5),
            numeric_features=TRIMMED_NUM_FEATURES,
            scale_numeric=True,
            alpha=0.5,
        ),
        CandidateRun(
            run_name="ridge_passthrough_full",
            estimator_name="ridge",
            estimator_factory=lambda: Ridge(alpha=2.0),
            numeric_features=FULL_NUM_FEATURES,
            scale_numeric=False,
            alpha=2.0,
        ),
        CandidateRun(
            run_name="sgd_scaled_full",
            estimator_name="sgd_regressor",
            estimator_factory=lambda: SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=0.0001,
                max_iter=3000,
                tol=1e-3,
                random_state=RANDOM_STATE,
            ),
            numeric_features=FULL_NUM_FEATURES,
            scale_numeric=True,
        ),
    ]


def _log_candidate_run(candidate: CandidateRun, X_train, X_valid, y_train, y_valid):
    pipeline = build_pipeline(candidate)
    pipeline.fit(X_train, y_train)

    train_rmse = evaluate_model(pipeline, X_train, y_train, "train")
    valid_rmse = evaluate_model(pipeline, X_valid, y_valid, "validation")

    with mlflow.start_run(run_name=candidate.run_name) as run:
        params = {
            "estimator": candidate.estimator_name,
            "scale_numeric": candidate.scale_numeric,
            "feature_set": "full" if candidate.numeric_features == FULL_NUM_FEATURES else "trimmed",
        }
        if candidate.alpha is not None:
            params["alpha"] = candidate.alpha

        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "train_rmse": train_rmse,
                "validation_rmse": valid_rmse,
            }
        )
        signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(1),
        )
        return {
            "run_id": run.info.run_id,
            "run_name": candidate.run_name,
            "validation_rmse": valid_rmse,
            "candidate": candidate,
        }


def train_and_register_best_model() -> tuple[Pipeline, dict[str, str]]:
    configure_mlflow()
    X, y = load_train_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    candidate_results = []
    for candidate in _build_candidate_runs():
        print(f"Running experiment: {candidate.run_name}")
        candidate_results.append(_log_candidate_run(candidate, X_train, X_valid, y_train, y_valid))

    best_candidate_result = min(candidate_results, key=lambda result: result["validation_rmse"])
    best_candidate = best_candidate_result["candidate"]
    print(
        f"Best validation RMSE: {best_candidate_result['validation_rmse']:.4f} "
        f"with {best_candidate.run_name}"
    )

    final_pipeline = build_pipeline(best_candidate)
    final_pipeline.fit(X, y)

    X_test, y_test = load_test_data()
    test_rmse = evaluate_model(final_pipeline, X_test, y_test, "test")

    with mlflow.start_run(run_name=f"{best_candidate.run_name}_final") as run:
        params = {
            "estimator": best_candidate.estimator_name,
            "scale_numeric": best_candidate.scale_numeric,
            "feature_set": "full" if best_candidate.numeric_features == FULL_NUM_FEATURES else "trimmed",
        }
        if best_candidate.alpha is not None:
            params["alpha"] = best_candidate.alpha

        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "validation_rmse": best_candidate_result["validation_rmse"],
                "test_rmse": test_rmse,
            }
        )
        signature = infer_signature(X.head(5), final_pipeline.predict(X.head(5)))
        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X.head(1),
        )
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=get_registered_model_name(),
        )

    snapshot = get_registered_model_snapshot(get_registered_model_name(), str(registered_model.version))
    model_metadata = snapshot_to_metadata("main", snapshot)
    persist_model(final_pipeline, MODEL_PATH)
    return final_pipeline, model_metadata


def main() -> None:
    _, model_metadata = train_and_register_best_model()
    print(f"Registered model version: {model_metadata['model_version']}")


if __name__ == "__main__":
    main()
