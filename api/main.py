import sqlite3
import os
import sys
import dill
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from . import config

DB_PATH = config.CONFIG["paths"]["db_path"]
MODEL_PATH = config.CONFIG["paths"]["model_path"]
MODEL_CUSTOM_PATH = config.CONFIG["paths"]["model_custom_path"]

app = FastAPI()

print(f"Loading the model from {MODEL_PATH}")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print(f"Loading the custom model from {MODEL_CUSTOM_PATH}")
with open(MODEL_CUSTOM_PATH, "rb") as file:
    model_custom = dill.load(file)


class Trip(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


@app.get("/")
def root():
    return {"status": "ok", "service": "bihar-taxi-api"}


@app.post("/predict")
def predict(trip: Trip):
    input_data = pd.DataFrame([trip.model_dump()])
    result = model.predict(input_data)[0]
    return {"result": float(result)}


@app.post("/predict_custom")
def predict_custom(trip: Trip):
    input_data = pd.DataFrame([trip.model_dump()])
    result = model_custom.predict(input_data)[0]
    return {"result": int(result)}


@app.get("/trips/randomtest")
def get_random_test_trip():
    print(f"Reading random test data from the database: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    try:
        data_test = pd.read_sql("SELECT * FROM test ORDER BY RANDOM() LIMIT 1", con)
    finally:
        con.close()

    X = data_test.drop(columns=["trip_duration"])
    y = data_test["trip_duration"].iloc[0]
    return {"x": X.iloc[0].to_dict(), "y": float(y)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
