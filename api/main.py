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


def _ensure_predictions_table():
    """Crée la table de logs de prédiction si elle n'existe pas déjà."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                vendor_id INTEGER NOT NULL,
                pickup_datetime TEXT NOT NULL,
                passenger_count INTEGER NOT NULL,
                pickup_longitude REAL NOT NULL,
                pickup_latitude REAL NOT NULL,
                dropoff_longitude REAL NOT NULL,
                dropoff_latitude REAL NOT NULL,
                store_and_fwd_flag TEXT NOT NULL,
                prediction REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _save_prediction(trip, prediction: float, model_name: str):
    """Enregistre une prédiction et les features d'entrée dans SQLite."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            INSERT INTO prediction_logs (
                model_name,
                vendor_id,
                pickup_datetime,
                passenger_count,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                store_and_fwd_flag,
                prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                trip.vendor_id,
                trip.pickup_datetime,
                trip.passenger_count,
                trip.pickup_longitude,
                trip.pickup_latitude,
                trip.dropoff_longitude,
                trip.dropoff_latitude,
                trip.store_and_fwd_flag,
                float(prediction),
            ),
        )
        con.commit()
    finally:
        con.close()

print(f"Loading the model from {MODEL_PATH}")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print(f"Loading the custom model from {MODEL_CUSTOM_PATH}")
with open(MODEL_CUSTOM_PATH, "rb") as file:
    model_custom = dill.load(file)

_ensure_predictions_table()


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
    """Vérifie que l'API est disponible et renvoie un statut de santé."""
    return {"status": "ok", "service": "bihar-taxi-api"}


@app.post("/predict")
def predict(trip: Trip):
    """Prédit la durée du trajet avec le modèle principal à partir des données d'entrée."""
    input_data = pd.DataFrame([trip.model_dump()])
    result = model.predict(input_data)[0]
    _save_prediction(trip, float(result), "main")
    return {"result": float(result)}


@app.post("/predict_custom")
def predict_custom(trip: Trip):
    """Prédit la durée avec le modèle custom, puis convertit le résultat en entier."""
    input_data = pd.DataFrame([trip.model_dump()])
    result = model_custom.predict(input_data)[0]
    _save_prediction(trip, float(result), "custom")
    return {"result": int(result)}


@app.get("/predictions/recent")
def get_recent_predictions(limit: int = 20, model_name: str | None = None):
    """Retourne les dernières prédictions persistées, avec filtre optionnel par modèle."""
    safe_limit = max(1, min(limit, 200))
    safe_model = model_name.strip().lower() if model_name else None

    if safe_model and safe_model not in {"main", "custom"}:
        return {
            "count": 0,
            "items": [],
            "error": "model_name doit être 'main' ou 'custom'",
        }

    con = sqlite3.connect(DB_PATH)
    try:
        if safe_model:
            df = pd.read_sql(
                """
                SELECT
                    id,
                    model_name,
                    vendor_id,
                    pickup_datetime,
                    passenger_count,
                    pickup_longitude,
                    pickup_latitude,
                    dropoff_longitude,
                    dropoff_latitude,
                    store_and_fwd_flag,
                    prediction,
                    created_at
                FROM prediction_logs
                WHERE model_name = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                con,
                params=(safe_model, safe_limit),
            )
        else:
            df = pd.read_sql(
                """
                SELECT
                    id,
                    model_name,
                    vendor_id,
                    pickup_datetime,
                    passenger_count,
                    pickup_longitude,
                    pickup_latitude,
                    dropoff_longitude,
                    dropoff_latitude,
                    store_and_fwd_flag,
                    prediction,
                    created_at
                FROM prediction_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                con,
                params=(safe_limit,),
            )
    finally:
        con.close()

    return {
        "count": int(len(df)),
        "items": df.to_dict(orient="records"),
    }


@app.delete("/predictions")
def delete_predictions(model_name: str | None = None, confirm: bool = False):
    """Supprime les logs de prédictions (tous ou filtrés par modèle)."""
    safe_model = model_name.strip().lower() if model_name else None

    if safe_model and safe_model not in {"main", "custom"}:
        return {
            "deleted": 0,
            "error": "model_name doit être 'main' ou 'custom'",
        }

    # Protection anti-suppression massive non intentionnelle.
    if not safe_model and not confirm:
        return {
            "deleted": 0,
            "error": "Pour supprimer tous les logs, utilisez confirm=true",
        }

    con = sqlite3.connect(DB_PATH)
    try:
        if safe_model:
            cur = con.execute("DELETE FROM prediction_logs WHERE model_name = ?", (safe_model,))
        else:
            cur = con.execute("DELETE FROM prediction_logs")
        con.commit()
        deleted = int(cur.rowcount) if cur.rowcount is not None else 0
    finally:
        con.close()

    return {
        "deleted": deleted,
        "scope": safe_model or "all",
    }


@app.get("/trips/randomtest")
def get_random_test_trip():
    """Récupère un trajet aléatoire depuis la table test avec sa cible réelle."""
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
