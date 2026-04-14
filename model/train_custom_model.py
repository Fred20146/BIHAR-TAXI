import os
import dill
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

import common
from model.features import build_model_features
from model.load_data import load_test_data, load_train_data

MODEL_PATH = common.CONFIG["paths"]["model_custom_path"]

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


class TaxiModel:
    def __init__(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("scaling", StandardScaler(), NUM_FEATURES),
            ]
        )
        self.model = Pipeline(
            steps=[
                ("feature_engineering", FunctionTransformer(build_model_features, validate=False)),
                ("preprocess", preprocessor),
                ("regression", Ridge()),
            ]
        )

    def __preprocess_target(self, y):
        return np.log1p(y)

    def __postprocess_target(self, raw_output):
        output = np.expm1(raw_output)
        return np.round(output)

    def fit(self, X, y):
        y_processed = self.__preprocess_target(y)
        self.model.fit(X, y_processed)
        return self

    def predict(self, X):
        try:
            check_is_fitted(self.model)
            raw_output = self.model.predict(X)
            output = self.__postprocess_target(raw_output)
        except NotFittedError:
            print("Model is not fitted yet.")
            raise
        return output


def train_model() -> TaxiModel:
    print("Building a custom model")
    X_train, y_train = load_train_data()

    trained_model = TaxiModel()
    trained_model.fit(X_train, y_train)

    y_pred = trained_model.predict(X_train)
    score = root_mean_squared_error(y_train, y_pred)
    print(f"RMSE on train data {score:.4f}")
    return trained_model


def evaluate_model(trained_model: TaxiModel) -> float:
    print("Evaluating the custom model")
    X_test, y_test = load_test_data()
    y_pred = trained_model.predict(X_test)
    score = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE on test data {score:.4f}")
    return score


def persist_model(trained_model: TaxiModel, path: str) -> None:
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        dill.settings["recurse"] = True
        dill.dump(trained_model, file)
    print("Done")


def main() -> None:
    fitted_model = train_model()
    evaluate_model(fitted_model)
    persist_model(fitted_model, MODEL_PATH)


if __name__ == "__main__":
    main()
