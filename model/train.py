import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import common
from model.features import build_model_features
from model.load_data import load_test_data, load_train_data

MODEL_PATH = common.CONFIG["paths"]["model_path"]

NUM_FEATURES = [
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
CAT_FEATURES = ["weekday", "month"]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("scaling", StandardScaler(), NUM_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(build_model_features, validate=False)),
            ("preprocess", preprocessor),
            ("regression", Ridge()),
        ]
    )


def train_model() -> Pipeline:
    print("Building a model")
    X_train, y_train = load_train_data()
    trained_model = build_pipeline()
    trained_model.fit(X_train, y_train)

    y_pred = trained_model.predict(X_train)
    score = root_mean_squared_error(y_train, y_pred)
    print(f"RMSE on train data {score:.4f}")
    return trained_model


def evaluate_model(trained_model: Pipeline) -> float:
    print("Evaluating the model")
    X_test, y_test = load_test_data()
    y_pred = trained_model.predict(X_test)
    score = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE on test data {score:.4f}")
    return score


def persist_model(trained_model: Pipeline, path: str) -> None:
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(trained_model, file)
    print("Done")


def main() -> None:
    fitted_model = train_model()
    evaluate_model(fitted_model)
    persist_model(fitted_model, MODEL_PATH)


if __name__ == "__main__":
    main()
