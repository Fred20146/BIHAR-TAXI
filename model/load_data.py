import sqlite3

import pandas as pd

import common

DB_PATH = common.CONFIG["paths"]["db_path"]


def _read_sql(query: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(query, con, parse_dates=["pickup_datetime"])
    finally:
        con.close()


def load_train_data():
    print(f"Reading train data from the database: {DB_PATH}")
    data_train = _read_sql("SELECT * FROM train")
    X = data_train.drop(columns=["trip_duration"])
    y = data_train["trip_duration"]
    return X, y


def load_test_data():
    print(f"Reading test data from the database: {DB_PATH}")
    data_test = _read_sql("SELECT * FROM test")
    X = data_test.drop(columns=["trip_duration"])
    y = data_test["trip_duration"]
    return X, y


def load_random_test_data(n_samples: int = 5):
    print(f"Reading random test data from the database: {DB_PATH}")
    data_test = _read_sql(f"SELECT * FROM test ORDER BY RANDOM() LIMIT {n_samples}")
    X = data_test.drop(columns=["trip_duration"])
    y = data_test["trip_duration"]
    return X, y
