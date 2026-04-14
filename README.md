# BIHAR-TAXI

Recreation of the same MLOps structure as the reference repository, adapted to the NYC Taxi Trip Duration data and workflow from the provided Colab notebook.

## Files for the basic training pipeline

- `data/download_data.py`: downloads and stores the dataset in SQLite (`data/taxi.db`)
- `model/load_data.py`: data access layer for train/test/random samples
- `model/features.py`: feature engineering inspired by the Colab tasks
- `model/train.py`: trains and saves the base model (`models/taxi.model`)
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
cd api
python main.py
```

### Endpoints

- `POST /predict`: returns base model prediction (seconds, float)
- `POST /predict_custom`: returns custom model prediction (seconds, rounded int)
- `GET /trips/randomtest`: returns one random test trip + target value

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
