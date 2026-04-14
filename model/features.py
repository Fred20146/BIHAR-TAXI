import numpy as np
import pandas as pd


def haversine_array(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    avg_earth_radius_km = 6371
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    return 2 * avg_earth_radius_km * np.arcsin(np.sqrt(d))


def is_high_traffic_trip(df: pd.DataFrame) -> pd.Series:
    weekday = df["weekday"]
    hour = df["hour"]
    return (
        ((hour >= 8) & (hour <= 19) & (weekday >= 0) & (weekday <= 4))
        | ((hour >= 13) & (hour <= 20) & (weekday == 5))
        | ((hour >= 13) & (hour <= 20) & (weekday == 6))
    )


def is_high_speed_trip(df: pd.DataFrame) -> pd.Series:
    weekday = df["weekday"]
    hour = df["hour"]
    return (
        ((hour >= 2) & (hour <= 5) & (weekday >= 0) & (weekday <= 4))
        | ((hour >= 4) & (hour <= 7) & (weekday == 5))
        | ((hour >= 5) & (hour <= 7) & (weekday == 6))
    )


def _is_rare_point(df: pd.DataFrame, lat_col: str, lon_col: str, q=(0.01, 0.995, 0.0, 0.95)) -> pd.Series:
    qmin_lat, qmax_lat, qmin_lon, qmax_lon = q
    lat_min = df[lat_col].quantile(qmin_lat)
    lat_max = df[lat_col].quantile(qmax_lat)
    lon_min = df[lon_col].quantile(qmin_lon)
    lon_max = df[lon_col].quantile(qmax_lon)
    return (df[lat_col] < lat_min) | (df[lat_col] > lat_max) | (df[lon_col] < lon_min) | (df[lon_col] > lon_max)


def build_model_features(X: pd.DataFrame) -> pd.DataFrame:
    res = X.copy()
    res["pickup_datetime"] = pd.to_datetime(res["pickup_datetime"], errors="coerce")

    res["weekday"] = res["pickup_datetime"].dt.weekday
    res["month"] = res["pickup_datetime"].dt.month
    res["hour"] = res["pickup_datetime"].dt.hour

    # Known abnormal low-trip periods from the source notebook.
    abnormal_dates = pd.to_datetime(["2016-01-23", "2016-01-24"]).date
    res["abnormal_period"] = res["pickup_datetime"].dt.date.isin(abnormal_dates).astype(int)

    distance_haversine = haversine_array(
        res["pickup_latitude"],
        res["pickup_longitude"],
        res["dropoff_latitude"],
        res["dropoff_longitude"],
    )
    res["log_distance_haversine"] = np.log1p(distance_haversine)

    res["is_high_traffic_trip"] = is_high_traffic_trip(res).astype(int)
    res["is_high_speed_trip"] = is_high_speed_trip(res).astype(int)
    res["is_rare_pickup_point"] = _is_rare_point(res, "pickup_latitude", "pickup_longitude").astype(int)
    res["is_rare_dropoff_point"] = _is_rare_point(res, "dropoff_latitude", "dropoff_longitude").astype(int)

    res["vendor_id"] = res["vendor_id"].map({1: 0, 2: 1}).fillna(0).astype(int)
    res["store_and_fwd_flag"] = res["store_and_fwd_flag"].map({"N": 0, "Y": 1}).fillna(0).astype(int)
    res["passenger_count"] = res["passenger_count"].clip(lower=0, upper=6)

    return res
