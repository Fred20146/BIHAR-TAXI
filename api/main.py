import sqlite3
import os
import sys
import dill
import pickle
import hashlib
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator, model_validator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from . import config
from mlflow.exceptions import MlflowException  # pyright: ignore[reportMissingImports]
from model.mlflow_utils import (
    configure_mlflow,
    get_registered_model_name,
    load_registered_model,
    snapshot_to_metadata,
)

DB_PATH = config.CONFIG["paths"]["db_path"]
MODEL_PATH = config.CONFIG["paths"]["model_path"]
MODEL_CUSTOM_PATH = config.CONFIG["paths"]["model_custom_path"]
REGISTERED_MODEL_NAME = get_registered_model_name()

MIN_TRIP_DISTANCE_METERS = 50.0
NYC_MIN_LONGITUDE = -74.30
NYC_MAX_LONGITUDE = -73.65
NYC_MIN_LATITUDE = 40.45
NYC_MAX_LATITUDE = 41.05

app = FastAPI()


def _ensure_model_registry_table():
    """Crée la table des métadonnées de modèles si elle n'existe pas déjà."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_name TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                model_file_mtime TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_version_path
            ON model_registry(model_version, model_path)
            """
        )
        con.commit()
    finally:
        con.close()


def _ensure_predictions_table():
    """Crée la table de logs de prédiction si elle n'existe pas déjà et applique les migrations utiles."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL DEFAULT 'unknown',
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

        # Migration douce: ajoute model_version si la table existait déjà sans cette colonne.
        cols = con.execute("PRAGMA table_info(prediction_logs)").fetchall()
        col_names = {col[1] for col in cols}
        if "model_version" not in col_names:
            con.execute(
                "ALTER TABLE prediction_logs ADD COLUMN model_version TEXT NOT NULL DEFAULT 'unknown'"
            )

        con.commit()
    finally:
        con.close()


def _hash_file(path: str) -> str:
    """Calcule un hash court du fichier modèle pour versionner les artefacts locaux."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as model_bin:
        for chunk in iter(lambda: model_bin.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]


def _build_model_metadata(model_name: str, model_path: str) -> dict:
    """Construit les métadonnées persistées d'un modèle local."""
    file_mtime = os.path.getmtime(model_path)
    file_mtime_iso = pd.to_datetime(file_mtime, unit="s", utc=True).isoformat()
    file_hash = _hash_file(model_path)
    model_version = f"{model_name}-{file_hash}"
    return {
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "model_file_mtime": file_mtime_iso,
    }


def _upsert_model_metadata(model_metadata: dict):
    """Insère ou met à jour les métadonnées du modèle dans model_registry."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            INSERT INTO model_registry (
                model_name,
                model_version,
                model_path,
                model_file_mtime,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(model_name) DO UPDATE SET
                model_version=excluded.model_version,
                model_path=excluded.model_path,
                model_file_mtime=excluded.model_file_mtime,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                model_metadata["model_name"],
                model_metadata["model_version"],
                model_metadata["model_path"],
                model_metadata["model_file_mtime"],
            ),
        )
        con.commit()
    finally:
        con.close()


def _save_prediction(trip, prediction: float, model_name: str, model_version: str):
    """Enregistre une prédiction et les features d'entrée dans SQLite."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """
            INSERT INTO prediction_logs (
                model_name,
                model_version,
                vendor_id,
                pickup_datetime,
                passenger_count,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                store_and_fwd_flag,
                prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                model_version,
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


def _load_primary_model():
    """Charge le dernier modèle enregistré dans MLflow, avec repli local si nécessaire."""
    try:
        configure_mlflow()
        loaded_model, snapshot = load_registered_model(REGISTERED_MODEL_NAME)
        metadata = snapshot_to_metadata("main", snapshot)
        print(f"Loaded MLflow model {REGISTERED_MODEL_NAME} version {snapshot.version}")
        return loaded_model, metadata
    except (MlflowException, RuntimeError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Falling back to local model file {MODEL_PATH}: {error}")
        with open(MODEL_PATH, "rb") as model_file:
            fallback_model = pickle.load(model_file)
        return fallback_model, _build_model_metadata("main", MODEL_PATH)


model, primary_model_metadata = _load_primary_model()

print(f"Loading the custom model from {MODEL_CUSTOM_PATH}")
with open(MODEL_CUSTOM_PATH, "rb") as custom_model_file:
    model_custom = dill.load(custom_model_file)

_ensure_model_registry_table()
_ensure_predictions_table()

MODEL_METADATA = {
    "main": primary_model_metadata,
    "custom": _build_model_metadata("custom", MODEL_CUSTOM_PATH),
}
for model_meta in MODEL_METADATA.values():
    _upsert_model_metadata(model_meta)

LOADED_MODELS_BY_VERSION = {
    MODEL_METADATA["main"]["model_version"]: {"model_name": "main", "model": model},
    MODEL_METADATA["custom"]["model_version"]: {
        "model_name": "custom",
        "model": model_custom,
    },
}
LATEST_MODEL_VERSION = MODEL_METADATA["main"]["model_version"]


def _resolve_model_by_version(model_version: str | None):
    """Résout le modèle à utiliser à partir d'une version explicite ou de la version la plus récente."""
    version = model_version or LATEST_MODEL_VERSION
    selected = LOADED_MODELS_BY_VERSION.get(version)
    if selected is None:
        try:
            loaded_model, snapshot = load_registered_model(REGISTERED_MODEL_NAME, version)
        except (MlflowException, RuntimeError, FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "model_version introuvable",
                    "requested_model_version": version,
                    "available_model_versions": sorted(LOADED_MODELS_BY_VERSION.keys()),
                },
            ) from exc
        metadata = snapshot_to_metadata("main", snapshot)
        _upsert_model_metadata(metadata)
        LOADED_MODELS_BY_VERSION[metadata["model_version"]] = {
            "model_name": "main",
            "model": loaded_model,
        }
        return LOADED_MODELS_BY_VERSION[metadata["model_version"]], metadata["model_version"]
    return selected, version


class Trip(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str

    @field_validator(
        "pickup_longitude",
        "dropoff_longitude",
    )
    @classmethod
    def validate_longitude(cls, value: float):
        if not NYC_MIN_LONGITUDE <= value <= NYC_MAX_LONGITUDE:
            raise ValueError(
                f"La longitude doit être dans la zone de New York [{NYC_MIN_LONGITUDE}, {NYC_MAX_LONGITUDE}]."
            )
        return value

    @field_validator(
        "pickup_latitude",
        "dropoff_latitude",
    )
    @classmethod
    def validate_latitude(cls, value: float):
        if not NYC_MIN_LATITUDE <= value <= NYC_MAX_LATITUDE:
            raise ValueError(
                f"La latitude doit être dans la zone de New York [{NYC_MIN_LATITUDE}, {NYC_MAX_LATITUDE}]."
            )
        return value

    @model_validator(mode="after")
    def validate_trip_distance(self):
        """Vérifie que le trajet n'est pas trop court pour être exploitable."""
        earth_radius_meters = 6371000.0
        lat1 = radians(self.pickup_latitude)
        lon1 = radians(self.pickup_longitude)
        lat2 = radians(self.dropoff_latitude)
        lon2 = radians(self.dropoff_longitude)

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        haversine = 2 * earth_radius_meters * asin(
            sqrt(
                sin(delta_lat / 2) ** 2
                + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
            )
        )

        if haversine <= MIN_TRIP_DISTANCE_METERS:
            raise ValueError("La distance entre pickup et dropoff doit être supérieure à 50 mètres.")
        return self


class BatchTripRequest(BaseModel):
    trips: list[Trip]


@app.get("/")
def root():
    """Vérifie que l'API est disponible et renvoie un statut de santé."""
    return {"status": "ok", "service": "bihar-taxi-api"}


@app.post("/predict")
def predict(
    trip: Trip = Body(
        ...,
        example={
            "vendor_id": 1,
            "pickup_datetime": "2016-06-01 11:07:08",
            "passenger_count": 1,
            "pickup_longitude": -73.97777557373047,
            "pickup_latitude": 40.76396560668945,
            "dropoff_longitude": -73.96023559570312,
            "dropoff_latitude": 40.77887725830078,
            "store_and_fwd_flag": "N",
        },
    ),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle à utiliser. Si absent, la dernière version disponible est utilisée.",
        examples=["1", "2"],
    ),
):
    """Prédit la durée d'un trajet et renvoie aussi la version du modèle utilisée."""
    selected, resolved_model_version = _resolve_model_by_version(model_version)
    input_data = pd.DataFrame([trip.model_dump()])
    result = selected["model"].predict(input_data)[0]
    _save_prediction(
        trip,
        float(result),
        selected["model_name"],
        resolved_model_version,
    )
    return {
        "result": float(result),
        "model_version": resolved_model_version,
    }


@app.post("/predict_batch")
def predict_batch(
    batch_request: BatchTripRequest = Body(
        ...,
        example={
            "trips": [
                {
                    "vendor_id": 1,
                    "pickup_datetime": "2016-06-01 11:07:08",
                    "passenger_count": 1,
                    "pickup_longitude": -73.97777557373047,
                    "pickup_latitude": 40.76396560668945,
                    "dropoff_longitude": -73.96023559570312,
                    "dropoff_latitude": 40.77887725830078,
                    "store_and_fwd_flag": "N",
                },
                {
                    "vendor_id": 2,
                    "pickup_datetime": "2016-06-04 17:14:35",
                    "passenger_count": 1,
                    "pickup_longitude": -73.99095153808594,
                    "pickup_latitude": 40.73477935791016,
                    "dropoff_longitude": -73.98129272460938,
                    "dropoff_latitude": 40.74727249145508,
                    "store_and_fwd_flag": "N",
                },
            ]
        },
    ),
    model_version: str | None = Query(
        default=None,
        description="Version du modèle à utiliser. Si absent, la dernière version disponible est utilisée.",
        examples=["1", "2"],
    ),
):
    """Prédit la durée d'une liste de trajets et renvoie la version du modèle utilisée."""
    selected, resolved_model_version = _resolve_model_by_version(model_version)

    if not batch_request.trips:
        return {
            "predictions": [],
            "model_version": resolved_model_version,
        }

    input_data = pd.DataFrame([trip.model_dump() for trip in batch_request.trips])
    predictions = selected["model"].predict(input_data)
    predictions_float = [float(value) for value in predictions]

    for trip, prediction in zip(batch_request.trips, predictions_float):
        _save_prediction(
            trip,
            prediction,
            selected["model_name"],
            resolved_model_version,
        )

    return {
        "predictions": predictions_float,
        "model_version": resolved_model_version,
    }


@app.post("/predict_custom")
def predict_custom(trip: Trip):
    """Prédit la durée avec le modèle custom, puis convertit le résultat en entier."""
    input_data = pd.DataFrame([trip.model_dump()])
    result = model_custom.predict(input_data)[0]
    _save_prediction(
        trip,
        float(result),
        "custom",
        MODEL_METADATA["custom"]["model_version"],
    )
    return {"result": int(result)}


@app.get("/models/metadata")
def get_models_metadata():
    """Retourne les métadonnées persistées des modèles chargés."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            """
            SELECT
                model_name,
                model_version,
                model_path,
                model_file_mtime,
                created_at,
                updated_at
            FROM model_registry
            ORDER BY model_name ASC
            """,
            con,
        )
    finally:
        con.close()

    return {
        "count": int(len(df)),
        "items": df.to_dict(orient="records"),
    }


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
                    model_version,
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
                    model_version,
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
