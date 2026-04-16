# BIHAR-TAXI

Recreation of the same MLOps structure as the reference repository, adapted to the NYC Taxi Trip Duration data and workflow from the provided Colab notebook.

## Files for the basic training pipeline

- `data/download_data.py`: downloads and stores the dataset in SQLite (`data/taxi.db`)
- `model/load_data.py`: data access layer for train/test/random samples
- `model/features.py`: feature engineering inspired by the Colab tasks
- `model/train.py`: compares several MLflow runs, registers the best model, and saves the selected base model (`models/taxi.model`)
- `model/train_custom_model.py`: trains and saves the custom wrapped model (`models/taxi_custom.model`)
- `model/test_model.py`: loads and tests both models on random test rows
- `api/main.py`: FastAPI service for inference
- `common.py`: shared config loader from `config.yml`
- `requirements.txt`: Python dependencies

## Run the basic training pipeline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Load data into SQLite

```bash
python -m data.download_data
```

### 3. Train and save the base model

```bash
python -m model.train
```

This step now creates several MLflow runs with different regressors and preprocessing variants, then registers the best-performing model in the local MLflow registry.

### 4. Train and save the custom model

```bash
python -m model.train_custom_model
```

### 5. Test inference from saved models

```bash
python -m model.test_model
```

## FastAPI inference service

Start API:

```bash
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Run API in background (keep terminal available):

```bash
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8001 > api.log 2>&1 &
```

Stop API (example):

```bash
lsof -i :8001
kill <PID>
```

### Endpoints

- `POST /predict`: returns trip duration prediction + `model_version`
- `POST /predict_batch`: returns list of predictions + `model_version`
- `POST /predict_custom`: custom model prediction (legacy endpoint)
- `GET /trips/randomtest`: returns one random test trip + target value
- `GET /models/metadata`: returns available model metadata
- `GET /predictions/recent`: recent prediction logs (optional `model_name`, `limit`)
- `DELETE /predictions`: delete logs (`confirm=true` required for full deletion)

Validation rules applied at inference time:

- Coordinates must be in NYC bounds:
  - longitude in `[-74.30, -73.65]`
  - latitude in `[40.45, 41.05]`
- Haversine distance between pickup and dropoff must be `> 50m`

### MLflow artifacts

The training script stores tracking data and model artifacts locally under `data/mlflow/`.

To inspect the runs, start the MLflow UI from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000
```

The FastAPI service now loads the latest registered version of the main model by name from MLflow on startup. If the registry is not available, it falls back to the local pickle file so the service stays usable during setup.

The main prediction endpoints accept an optional `model_version` query parameter. When omitted, the API uses the latest registered version of the main model.

## Streamlit frontend

Launch the Streamlit UI, which is fully independent from the training pipeline and only consumes the FastAPI inference service:

```bash
streamlit run ui/app.py
```

By default, the UI calls the API at `http://127.0.0.1:8001`. You can change the API URL from the sidebar.

UI highlights:

- Single prediction form with Streamlit widgets (`date_input`, selectboxes, sliders)
- Error handling for invalid inputs and API failures
- Result card with:
  - predicted duration (`HH:mm:ss`)
  - estimated distance
  - `model_version` used by the API
- Interactive map visualization of pickup/dropoff and current route segment (driven by sliders)

Request body example for prediction endpoints:

```json
{
  "vendor_id": 1,
  "pickup_datetime": "2016-03-14 17:24:55",
  "passenger_count": 1,
  "pickup_longitude": -73.982154,
  "pickup_latitude": 40.767937,
  "dropoff_longitude": -73.96463,
  "dropoff_latitude": 40.765602,
  "store_and_fwd_flag": "N"
}
```

## Quick Validation Checklist

1. Start API on port `8001`
2. Start UI with `streamlit run ui/app.py`
3. Make one valid prediction and verify:
   - duration displayed in `HH:mm:ss`
   - model version displayed
4. Try a too-short trip (`<= 50m`) and verify user-friendly error message
5. Move sliders and verify route segment updates on map
