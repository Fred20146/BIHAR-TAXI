import dill
import pickle

from model.load_data import load_random_test_data

import common

MODEL_PATH = common.CONFIG["paths"]["model_path"]
MODEL_CUSTOM_PATH = common.CONFIG["paths"]["model_custom_path"]


def load_pickle_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        loaded_model = pickle.load(file)
    print("Done")
    return loaded_model


def load_dill_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        loaded_model = dill.load(file)
    print("Done")
    return loaded_model


def test_model(loaded_model, model_name: str):
    print(f"Test the model: {model_name}")
    X, y = load_random_test_data()
    y_pred = loaded_model.predict(X)
    df = X.copy()
    df["y_true"] = y
    df["y_pred"] = y_pred
    print(df.head())


if __name__ == "__main__":
    base_model = load_pickle_model(MODEL_PATH)
    test_model(base_model, "base")

    custom_model = load_dill_model(MODEL_CUSTOM_PATH)
    test_model(custom_model, "custom")
