import json
import os
import sqlite3
from datetime import datetime
from math import log2
from math import asin, cos, radians, sin, sqrt

import pydeck as pdk
import requests
import streamlit as st
import numpy as np
import pandas as pd
import yaml


CANDIDATE_API_URLS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8015",
    "http://127.0.0.1:8014",
]

NYC_MIN_LONGITUDE = -74.30
NYC_MAX_LONGITUDE = -73.65
NYC_MIN_LATITUDE = 40.45
NYC_MAX_LATITUDE = 41.05
NYC_VIEWBOX = (-74.2591, 40.9176, -73.7004, 40.4774)
NYC_ADDRESS_SUGGESTIONS = [
    "Times Square, Manhattan, New York, NY",
    "Central Park, Manhattan, New York, NY",
    "Brooklyn Bridge, New York, NY",
    "Penn Station, Manhattan, New York, NY",
    "JFK Airport, Queens, New York, NY",
    "Wall Street, Manhattan, New York, NY",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ROOT_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yml")

CSV_REQUIRED_COLUMNS = [
    "vendor_id",
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
]


st.set_page_config(page_title="BIHAR-TAXI Predictor", page_icon="🚕", layout="wide")

# URL directe du pont de Brooklyn illuminé de nuit
hero_bg_url = 'url("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Pont_de_Brooklyn_de_nuit_-_Octobre_2008_edit.jpg/1920px-Pont_de_Brooklyn_de_nuit_-_Octobre_2008_edit.jpg")'

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 10% 10%, rgba(14, 116, 144, 0.16), transparent 42%),
            radial-gradient(circle at 90% 0%, rgba(16, 185, 129, 0.11), transparent 34%),
            linear-gradient(180deg, #0b1220 0%, #101a2d 55%, #16253d 100%);
        color: #e5e7eb;
    }}
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stCaptionContainer"] {{
        color: #e5e7eb;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    [data-testid="stSidebar"] {{
        border-right: 1px solid rgba(255, 255, 255, 0.18);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-image:
            linear-gradient(180deg, rgba(2, 6, 23, 0.70) 0%, rgba(15, 23, 42, 0.75) 40%, rgba(30, 41, 59, 0.80) 100%),
            url("https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg");
        background-size: cover;
        background-position: center;
        color: #f8fafc;
    }}
    [data-testid="stSidebar"] * {{
        color: #f8fafc;
    }}
    [data-testid="stSidebar"] .stTextInput > div > div,
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] .stButton > button {{
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(203, 213, 225, 0.25);
        border-radius: 12px;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: rgba(125, 211, 252, 0.8);
        box-shadow: 0 0 0 1px rgba(125, 211, 252, 0.3), 0 10px 22px rgba(2, 132, 199, 0.25);
    }}
    .hero {{
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.55) 0%, rgba(30, 41, 59, 0.50) 55%, rgba(51, 65, 85, 0.50) 100%),
            {hero_bg_url};
        background-size: cover;
        background-position: center;
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2rem;
    }}
    .hero p {{
        margin: 0.35rem 0 0;
        opacity: 0.88;
        font-size: 0.98rem;
    }}
    .metric-card {{
        background: linear-gradient(150deg, #0b1220 0%, #172235 65%, #22304a 100%);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
    }}
    .metric-card h3,
    .metric-card p,
    .metric-card strong {{
        color: #f8fafc;
    }}
    .metric-value {{
        margin: 0;
        font-size: 1.05rem;
        color: #f8fafc;
    }}
    .metric-label {{
        margin: 0.4rem 0 0;
        color: #cbd5e1;
    }}
    .small-note {{
        color: #cbd5e1;
        font-size: 0.9rem;
    }}
    .validation-card {{
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: #fee2e2;
        border: 1px solid #fecaca;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 24px rgba(127, 29, 29, 0.25);
    }}
    .validation-card strong {{
        color: #ffffff;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 10px 24px rgba(14, 165, 233, 0.35);
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(base_url: str, endpoint_path: str, method: str = "GET", body_payload: dict | None = None):
    """Exécute un appel HTTP JSON vers l'API FastAPI via la librairie requests."""
    url = base_url.rstrip("/") + endpoint_path
    http_response = requests.request(
        method=method,
        url=url,
        json=body_payload,
        timeout=30,
    )
    http_response.raise_for_status()
    return http_response.json()


def format_seconds_hms(seconds: float) -> str:
    """Convertit une durée en secondes vers le format HH:mm:ss."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcule la distance haversine entre deux points GPS en mètres."""
    earth_radius_meters = 6371000.0
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_meters * asin(sqrt(a))


def format_distance(distance_meters: float) -> str:
    """Affiche la distance en m si < 1000, sinon en km et m."""
    meters = max(0, int(round(distance_meters)))
    if meters < 1000:
        return f"{meters} m"
    km = meters // 1000
    remaining_m = meters % 1000
    if remaining_m == 0:
        return f"{km} km"
    return f"{km} km {remaining_m} m"


def build_trip_deck(pickup_lat: float, pickup_lon: float, dropoff_lat: float, dropoff_lon: float) -> pdk.Deck:
    """Construit une carte pydeck non éditable avec pickup/dropoff et segment du trajet."""
    midpoint_lat = (pickup_lat + dropoff_lat) / 2
    midpoint_lon = (pickup_lon + dropoff_lon) / 2

    # Zoom adaptatif pour garder tout le segment visible, même pour des trajets très courts.
    lat_span = abs(dropoff_lat - pickup_lat)
    lon_span = abs(dropoff_lon - pickup_lon)
    max_span = max(lat_span, lon_span, 0.001)
    padded_span = max_span * 1.8
    dynamic_zoom = max(9.0, min(16.0, log2(360.0 / padded_span) - 1.0))

    point_data = [
        {
            "name": "pickup",
            "lat": pickup_lat,
            "lon": pickup_lon,
            "color": [22, 163, 74],
        },
        {
            "name": "dropoff",
            "lat": dropoff_lat,
            "lon": dropoff_lon,
            "color": [239, 68, 68],
        },
    ]

    line_data = [
        {
            "path": [[pickup_lon, pickup_lat], [dropoff_lon, dropoff_lat]],
            "color": [59, 130, 246],
        }
    ]

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=90,
            radius_scale=1,
            radius_min_pixels=6,
            radius_max_pixels=16,
            pickable=True,
        ),
        pdk.Layer(
            "PathLayer",
            data=line_data,
            get_path="path",
            get_color="color",
            width_scale=1,
            width_min_pixels=3,
            get_width=6,
            pickable=False,
        ),
    ]

    view_state = pdk.ViewState(
        latitude=midpoint_lat,
        longitude=midpoint_lon,
        zoom=dynamic_zoom,
        pitch=0,
        bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
    )


@st.cache_data(ttl=30)
def fetch_model_metadata(api_base_url: str):
    try:
        metadata_payload = api_request(api_base_url, "/models/metadata")
        return metadata_payload, None
    except requests.exceptions.HTTPError as http_error:
        metadata_error_response = http_error.response
        error_status_code = metadata_error_response.status_code if metadata_error_response is not None else "?"
        details = metadata_error_response.text if metadata_error_response is not None else ""
        return None, f"HTTP {error_status_code} sur /models/metadata {details}".strip()
    except (requests.exceptions.RequestException, ValueError):
        return None, "Impossible de joindre l'API ou réponse JSON invalide."


def autodetect_api_url(candidates: list[str]):
    """Retourne la première URL API joignable exposant /models/metadata."""
    for candidate in candidates:
        metadata, _ = fetch_model_metadata(candidate)
        if metadata and metadata.get("items") is not None:
            return candidate, metadata, None
        if metadata and metadata.get("count") == 0:
            return candidate, metadata, None
    # Si aucune candidate ne passe, on renvoie la première avec son erreur pour diagnostic.
    first = candidates[0]
    _, first_error = fetch_model_metadata(first)
    return None, None, first_error


def predict_with_fallback(api_base_url: str, endpoint_path: str, request_payload: dict):
    """Essaie l'URL configurée puis des URLs locales connues si l'API est indisponible."""
    try:
        return api_request(api_base_url, endpoint_path, method="POST", body_payload=request_payload), api_base_url
    except requests.exceptions.RequestException:
        fallback_candidates = [url for url in CANDIDATE_API_URLS if url != api_base_url]
        for fallback_url in fallback_candidates:
            try:
                fallback_response = api_request(fallback_url, endpoint_path, method="POST", body_payload=request_payload)
                return fallback_response, fallback_url
            except requests.exceptions.RequestException:
                continue
        raise


@st.cache_data(ttl=60)
def get_db_path() -> str:
    """Résout le chemin de la base SQLite à partir du fichier de configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    rel_path = config_data["paths"]["db_path"]
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))


@st.cache_data(ttl=60)
def load_distribution_data() -> tuple[pd.Series, pd.Series, str]:
    """Charge la distribution train et la distribution des prédictions persistées."""
    sqlite_db_path = get_db_path()
    con = sqlite3.connect(sqlite_db_path)
    try:
        train_series = pd.read_sql("SELECT trip_duration FROM train", con)["trip_duration"].astype(float)
        try:
            pred_series = pd.read_sql("SELECT prediction FROM prediction_logs", con)["prediction"].astype(float)
        except (sqlite3.OperationalError, KeyError, ValueError):
            pred_series = pd.Series(dtype=float)
    finally:
        con.close()

    return train_series, pred_series, sqlite_db_path


def build_histogram(values: pd.Series, bin_edges: np.ndarray) -> pd.DataFrame:
    """Construit une table histogramme compatible avec st.bar_chart."""
    if values.empty:
        return pd.DataFrame({"count": []})

    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return pd.DataFrame({"count": []})

    counts, edges = np.histogram(clean, bins=bin_edges)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"count": counts}, index=labels)


def validate_csv_schema(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    """Vérifie la présence des colonnes obligatoires pour la prédiction batch."""
    missing_columns = [col for col in CSV_REQUIRED_COLUMNS if col not in frame.columns]
    return len(missing_columns) == 0, missing_columns


def normalize_csv_payload(frame: pd.DataFrame) -> list[dict]:
    """Convertit un DataFrame CSV en payload JSON prêt pour /predict_batch."""
    normalized = frame[CSV_REQUIRED_COLUMNS].copy()
    normalized["vendor_id"] = normalized["vendor_id"].astype(int)
    normalized["passenger_count"] = normalized["passenger_count"].astype(int)
    normalized["pickup_longitude"] = normalized["pickup_longitude"].astype(float)
    normalized["pickup_latitude"] = normalized["pickup_latitude"].astype(float)
    normalized["dropoff_longitude"] = normalized["dropoff_longitude"].astype(float)
    normalized["dropoff_latitude"] = normalized["dropoff_latitude"].astype(float)
    normalized["pickup_datetime"] = normalized["pickup_datetime"].astype(str)
    normalized["store_and_fwd_flag"] = normalized["store_and_fwd_flag"].astype(str)
    return normalized.to_dict(orient="records")


def geocode_nyc_address(address_text: str | None):
    """Géocode une adresse NYC en limitant la recherche au périmètre new-yorkais."""
    cleaned_address = (address_text or "").strip()
    if not cleaned_address:
        return None, "Adresse vide"

    try:
        geocode_response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": cleaned_address,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 5,
                "bounded": 1,
                "viewbox": f"{NYC_VIEWBOX[0]},{NYC_VIEWBOX[1]},{NYC_VIEWBOX[2]},{NYC_VIEWBOX[3]}",
            },
            headers={"User-Agent": "BIHAR-TAXI-Streamlit/1.0"},
            timeout=20,
        )
        geocode_response.raise_for_status()
        candidates = geocode_response.json()
    except requests.exceptions.RequestException:
        return None, "Géocodage indisponible"
    except ValueError:
        return None, "Réponse de géocodage invalide"

    if not candidates:
        return None, "Aucun résultat trouvé"

    best_match = candidates[0]
    try:
        latitude = float(best_match["lat"])
        longitude = float(best_match["lon"])
    except (KeyError, TypeError, ValueError):
        return None, "Résultat de géocodage incomplet"

    if not (NYC_MIN_LONGITUDE <= longitude <= NYC_MAX_LONGITUDE and NYC_MIN_LATITUDE <= latitude <= NYC_MAX_LATITUDE):
        return None, "Adresse hors zone New York"

    return {
        "latitude": latitude,
        "longitude": longitude,
        "display_name": best_match.get("display_name", cleaned_address),
    }, None


st.markdown(
    """
    <div class="hero">
        <h1>BIHAR-TAXI Predictor</h1>
        <p>Interface Streamlit autonome qui appelle l'API FastAPI pour obtenir une prédiction de durée de trajet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    current_page = st.radio(
        "Navigation",
        options=["Prédiction", "Statistiques"],
        index=0,
    )

    st.header("Configuration")
    default_api_url = os.getenv("BIHAR_TAXI_API_URL", "http://127.0.0.1:8001")
    api_url = st.text_input("URL de l'API FastAPI", value=default_api_url)

    if st.button("Auto-détecter API locale", use_container_width=True):
        detected_url, _, _ = autodetect_api_url([api_url] + CANDIDATE_API_URLS)
        if detected_url:
            st.session_state["detected_api_url"] = detected_url

    if "detected_api_url" in st.session_state:
        api_url = st.session_state["detected_api_url"]
        st.caption(f"API détectée: {api_url}")

    model_metadata, metadata_error = fetch_model_metadata(api_url)

    st.caption("Ce frontend ne dépend pas du pipeline d'entraînement. Il consomme uniquement l'API de prédiction.")

    if model_metadata and model_metadata.get("items"):
        version_options = [item["model_version"] for item in model_metadata["items"]]
        model_version = st.selectbox(
            "model_version",
            options=["latest"] + version_options,
            index=0,
            help="Si vous gardez 'latest', l'API sélectionne la version la plus récente disponible.",
        )
        model_name_by_version = {
            item.get("model_version"): item.get("model_name")
            for item in model_metadata["items"]
            if item.get("model_version")
        }
    else:
        model_version = st.text_input(
            "model_version",
            value="",
            help="Laissez vide pour utiliser la dernière version disponible.",
        )
        model_version = model_version.strip() or None
        model_name_by_version = {}

    st.divider()
    if not model_metadata:
        st.caption("Métadonnées modèles indisponibles actuellement.")

if current_page == "Prédiction":
    if "pickup_geo" not in st.session_state:
        st.session_state["pickup_geo"] = None
    if "dropoff_geo" not in st.session_state:
        st.session_state["dropoff_geo"] = None
    if "pickup_address_raw" not in st.session_state:
        st.session_state["pickup_address_raw"] = ""
    if "dropoff_address_raw" not in st.session_state:
        st.session_state["dropoff_address_raw"] = ""
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "adresse"

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("Entrée de prédiction")
        vendor_id = st.selectbox("vendor_id", options=[1, 2], index=0)
        vendor_id_value = int(vendor_id or 1)

        input_mode = st.radio(
            "Mode de saisie",
            options=["adresse", "coordonnées"],
            index=0 if st.session_state["input_mode"] == "adresse" else 1,
            horizontal=True,
            help="Choisis soit des adresses guidées, soit la saisie manuelle des coordonnées GPS.",
        )
        st.session_state["input_mode"] = input_mode

        now = datetime.now()
        st.markdown("**pickup_datetime**")
        date_col, hour_col, minute_col = st.columns([2.6, 1, 1])
        with date_col:
            pickup_date = st.date_input("Date", value=now.date(), label_visibility="collapsed")
        with hour_col:
            pickup_hour = st.selectbox(
                "Heure",
                options=list(range(24)),
                index=now.hour,
                format_func=lambda value: f"{value:02d}",
                label_visibility="collapsed",
            )
        with minute_col:
            pickup_minute = st.selectbox(
                "Minute",
                options=list(range(60)),
                index=now.minute,
                format_func=lambda value: f"{value:02d}",
                label_visibility="collapsed",
            )

        pickup_datetime = f"{pickup_date:%Y-%m-%d} {pickup_hour:02d}:{pickup_minute:02d}:00"
        passenger_count = st.slider("passenger_count", min_value=0, max_value=9, value=1, step=1)
        pickup_geo = None
        dropoff_geo = None
        pickup_coords_ready = False
        dropoff_coords_ready = False

        if input_mode == "adresse":
            st.markdown("**Adresse de départ**")
            pickup_address_suggestion = st.selectbox(
                "Adresse de départ courante",
                options=NYC_ADDRESS_SUGGESTIONS,
                index=0,
                help="Choisis un point courant pour éviter les erreurs de frappe.",
            )
            pickup_address = st.text_input(
                "Adresse de départ",
                value=pickup_address_suggestion,
                placeholder="Ex: 11 Wall St, New York, NY",
            )

            st.markdown("**Adresse d'arrivée**")
            dropoff_address_suggestion = st.selectbox(
                "Adresse d'arrivée courante",
                options=NYC_ADDRESS_SUGGESTIONS,
                index=2,
                help="Choisis un point courant pour éviter les erreurs de frappe.",
            )
            dropoff_address = st.text_input(
                "Adresse d'arrivée",
                value=dropoff_address_suggestion,
                placeholder="Ex: 1 Centre St, New York, NY",
            )

            if st.session_state["pickup_address_raw"] != pickup_address:
                st.session_state["pickup_geo"] = None
                st.session_state["pickup_address_raw"] = pickup_address
            if st.session_state["dropoff_address_raw"] != dropoff_address:
                st.session_state["dropoff_geo"] = None
                st.session_state["dropoff_address_raw"] = dropoff_address

            geocode_cols = st.columns(2)
            with geocode_cols[0]:
                resolve_pickup = st.button("Valider départ", use_container_width=True)
            with geocode_cols[1]:
                resolve_dropoff = st.button("Valider arrivée", use_container_width=True)

            if resolve_pickup:
                pickup_geo, pickup_error = geocode_nyc_address(pickup_address)
                st.session_state["pickup_geo"] = pickup_geo
                if pickup_error:
                    st.error(f"Départ: {pickup_error}")
                elif pickup_geo:
                    st.success(f"Départ validé: {pickup_geo['display_name']}")

            if resolve_dropoff:
                dropoff_geo, dropoff_error = geocode_nyc_address(dropoff_address)
                st.session_state["dropoff_geo"] = dropoff_geo
                if dropoff_error:
                    st.error(f"Arrivée: {dropoff_error}")
                elif dropoff_geo:
                    st.success(f"Arrivée validée: {dropoff_geo['display_name']}")

            pickup_geo = st.session_state.get("pickup_geo")
            dropoff_geo = st.session_state.get("dropoff_geo")

            if pickup_geo:
                st.caption(f"Départ: {pickup_geo['display_name']} ({pickup_geo['latitude']:.5f}, {pickup_geo['longitude']:.5f})")
            if dropoff_geo:
                st.caption(f"Arrivée: {dropoff_geo['display_name']} ({dropoff_geo['latitude']:.5f}, {dropoff_geo['longitude']:.5f})")

            pickup_coords_ready = bool(pickup_geo)
            dropoff_coords_ready = bool(dropoff_geo)

        else:
            pickup_longitude = st.slider(
                "pickup_longitude",
                min_value=NYC_MIN_LONGITUDE,
                max_value=NYC_MAX_LONGITUDE,
                value=-73.98,
                step=0.0001,
                format="%.4f",
            )
            pickup_latitude = st.slider(
                "pickup_latitude",
                min_value=NYC_MIN_LATITUDE,
                max_value=NYC_MAX_LATITUDE,
                value=40.75,
                step=0.0001,
                format="%.4f",
            )
            dropoff_longitude = st.slider(
                "dropoff_longitude",
                min_value=NYC_MIN_LONGITUDE,
                max_value=NYC_MAX_LONGITUDE,
                value=-73.96,
                step=0.0001,
                format="%.4f",
            )
            dropoff_latitude = st.slider(
                "dropoff_latitude",
                min_value=NYC_MIN_LATITUDE,
                max_value=NYC_MAX_LATITUDE,
                value=40.77,
                step=0.0001,
                format="%.4f",
            )

            pickup_geo = {
                "latitude": float(pickup_latitude),
                "longitude": float(pickup_longitude),
                "display_name": "Coordonnées pickup",
            }
            dropoff_geo = {
                "latitude": float(dropoff_latitude),
                "longitude": float(dropoff_longitude),
                "display_name": "Coordonnées dropoff",
            }
            pickup_coords_ready = True
            dropoff_coords_ready = True

        store_and_fwd_flag = st.selectbox("store_and_fwd_flag", options=["N", "Y"], index=0)
        submitted = st.button("Prédire", type="primary")

        if input_mode == "adresse":
            st.markdown(
                '<p class="small-note">Astuce: sélectionne une adresse connue puis clique sur “Résoudre”. Les adresses hors New York sont refusées automatiquement.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="small-note">Astuce: reste dans la zone New York et garde une distance supérieure à 50 m pour éviter la validation API.</p>',
                unsafe_allow_html=True,
            )

    with col2:
        st.subheader("Résultat")
        if pickup_coords_ready and dropoff_coords_ready:
            assert pickup_geo is not None and dropoff_geo is not None
            trip_distance_m = haversine_meters(
                float(pickup_geo["latitude"]),
                float(pickup_geo["longitude"]),
                float(dropoff_geo["latitude"]),
                float(dropoff_geo["longitude"]),
            )
            trip_distance_display = format_distance(trip_distance_m)
        else:
            trip_distance_m = 0.0
            trip_distance_display = "en attente de validation"

        if submitted:
            if not pickup_coords_ready or not dropoff_coords_ready:
                st.markdown(
                    """
                    <div class="validation-card">
                        <strong>Validation adresse</strong><br>
                        Valides d'abord l'adresse de départ et l'adresse d'arrivée avant de lancer la prédiction.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                assert pickup_geo is not None and dropoff_geo is not None
                request_body = {
                    "vendor_id": vendor_id_value,
                    "pickup_datetime": pickup_datetime,
                    "passenger_count": int(passenger_count),
                    "pickup_longitude": float(pickup_geo["longitude"]),
                    "pickup_latitude": float(pickup_geo["latitude"]),
                    "dropoff_longitude": float(dropoff_geo["longitude"]),
                    "dropoff_latitude": float(dropoff_geo["latitude"]),
                    "store_and_fwd_flag": store_and_fwd_flag,
                }

                query = {}
                if model_version and model_version != "latest":
                    query["model_version"] = model_version

                if trip_distance_m <= 50:
                    st.markdown(
                        """
                        <div class="validation-card">
                            <strong>Validation entrée</strong><br>
                            La distance de la course ne peut être inférieure à 50 m.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Distance calculée: {trip_distance_display}")
                else:
                    path = "/predict"
                    if query:
                        from urllib.parse import urlencode

                        path = f"{path}?{urlencode(query)}"

                    try:
                        api_response, used_api_url = predict_with_fallback(api_url, path, request_body)
                        if used_api_url != api_url:
                            st.session_state["detected_api_url"] = used_api_url
                            st.info(f"API indisponible sur {api_url}. Bascule automatique sur {used_api_url}.")
                        display_model_version = api_response.get("model_version", "non fourni par l'API")
                        display_model_name = model_name_by_version.get(display_model_version)
                        if not display_model_name and isinstance(display_model_version, str) and "-" in display_model_version:
                            display_model_name = display_model_version.split("-", 1)[0]
                        if not display_model_name:
                            display_model_name = "non fourni par l'API"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3 style="margin-top:0; margin-bottom:0.4rem;">Prédiction reçue</h3>
                                <p class="metric-value"><strong>Durée estimée:</strong> {format_seconds_hms(api_response['result'])}</p>
                                <p class="metric-label"><strong>Distance estimée:</strong> {trip_distance_display}</p>
                                <p class="metric-label"><strong>Nom du modèle:</strong> {display_model_name}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.caption("Réponse brute de l'API")
                        st.json(api_response)
                    except requests.exceptions.HTTPError as error:
                        error_response = error.response
                        status_code = error_response.status_code if error_response is not None else 500
                        detail_message = "Entrée invalide. Vérifiez les adresses et la distance minimale de 50 m."
                        if error_response is not None:
                            try:
                                payload = error_response.json()
                                if isinstance(payload, dict) and payload.get("detail"):
                                    detail_message = str(payload.get("detail"))
                            except ValueError:
                                if error_response.text:
                                    detail_message = error_response.text
                        st.error(f"Erreur API ({status_code})")
                        st.warning(detail_message)
                    except requests.exceptions.RequestException as error:
                        st.error("Impossible de joindre l'API configurée et aucune API locale n'a été trouvée.")
                        st.code(str(error))

        st.divider()
        st.subheader("Carte du trajet sélectionné")
        st.caption(f"Distance actuelle: {trip_distance_display}")
        if pickup_coords_ready and dropoff_coords_ready:
            assert pickup_geo is not None and dropoff_geo is not None
            trip_deck = build_trip_deck(
                float(pickup_geo["latitude"]),
                float(pickup_geo["longitude"]),
                float(dropoff_geo["latitude"]),
                float(dropoff_geo["longitude"]),
            )
            st.pydeck_chart(trip_deck, use_container_width=True)
            if input_mode == "adresse":
                st.caption("La carte se base sur les adresses résolues. Les points ne sont pas déplaçables.")
            else:
                st.caption("La carte se base sur les coordonnées saisies manuellement.")
        else:
            st.info("Valides les deux adresses pour afficher la carte du trajet.")

    st.divider()
    st.subheader("Exemple de charge utile")
    st.code(
        json.dumps(
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
            indent=2,
        ),
        language="json",
    )

    st.divider()
    st.subheader("Prédiction batch par CSV")
    st.caption("Importez un CSV avec les mêmes colonnes que le payload single prediction.")

    uploaded_csv = st.file_uploader(
        "Fichier CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="batch_csv_uploader",
    )

    if uploaded_csv is not None:
        try:
            csv_frame = pd.read_csv(uploaded_csv)
        except pd.errors.ParserError as csv_error:
            st.error("Le fichier CSV n'a pas pu être lu.")
            st.code(str(csv_error))
            csv_frame = None

        if csv_frame is not None:
            is_valid_schema, missing_cols = validate_csv_schema(csv_frame)
            if not is_valid_schema:
                st.error("Le CSV ne contient pas toutes les colonnes requises.")
                st.write("Colonnes manquantes:", missing_cols)
                st.write("Colonnes attendues:", CSV_REQUIRED_COLUMNS)
            elif csv_frame.empty:
                st.warning("Le CSV est vide.")
            else:
                st.write(f"Lignes détectées: {len(csv_frame)}")
                st.dataframe(csv_frame.head(10), use_container_width=True)

                if st.button("Lancer la prédiction batch", key="run_batch_csv"):
                    try:
                        batch_trips = normalize_csv_payload(csv_frame)
                        batch_query = {}
                        if model_version and model_version != "latest":
                            batch_query["model_version"] = model_version

                        batch_path = "/predict_batch"
                        if batch_query:
                            from urllib.parse import urlencode

                            batch_path = f"{batch_path}?{urlencode(batch_query)}"

                        batch_payload = {"trips": batch_trips}
                        batch_response, used_batch_api_url = predict_with_fallback(api_url, batch_path, batch_payload)
                        if used_batch_api_url != api_url:
                            st.session_state["detected_api_url"] = used_batch_api_url
                            st.info(f"API indisponible sur {api_url}. Bascule automatique sur {used_batch_api_url}.")

                        predictions = batch_response.get("predictions", [])
                        batch_model_version = batch_response.get("model_version", "non fourni par l'API")

                        if len(predictions) != len(csv_frame):
                            st.warning("Le nombre de prédictions retournées ne correspond pas au nombre de lignes du CSV.")

                        result_frame = csv_frame.copy()
                        result_frame["predicted_trip_duration_sec"] = pd.Series(predictions)
                        result_frame["predicted_trip_duration_hms"] = result_frame["predicted_trip_duration_sec"].apply(
                            lambda seconds: format_seconds_hms(float(seconds)) if pd.notna(seconds) else ""
                        )

                        st.success("Prédiction batch terminée.")
                        st.write(f"Version du modèle: {batch_model_version}")
                        st.dataframe(result_frame.head(50), use_container_width=True)

                        csv_output_bytes = result_frame.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Télécharger les résultats CSV",
                            data=csv_output_bytes,
                            file_name="predictions_batch_results.csv",
                            mime="text/csv",
                            key="download_batch_csv",
                        )
                    except requests.exceptions.HTTPError as batch_http_error:
                        response = batch_http_error.response
                        status_code = response.status_code if response is not None else 500
                        detail_message = "Erreur lors de la prédiction batch. Vérifiez le format des données."
                        if response is not None:
                            try:
                                payload = response.json()
                                if isinstance(payload, dict) and payload.get("detail"):
                                    detail_message = str(payload.get("detail"))
                            except ValueError:
                                if response.text:
                                    detail_message = response.text
                        st.error(f"Erreur API ({status_code})")
                        st.warning(detail_message)
                    except requests.exceptions.RequestException as batch_req_error:
                        st.error("Impossible de joindre l'API pour la prédiction batch.")
                        st.code(str(batch_req_error))

else:
    st.subheader("Statistiques des distributions")
    train_values, pred_values, db_path = load_distribution_data()

    st.caption(f"Source des données: {db_path}")
    st.caption("Comparaison entre la distribution des `trip_duration` d'entraînement et les valeurs prédites persistées.")

    if train_values.empty:
        st.warning("Aucune donnée d'entraînement trouvée dans la base SQLite.")
    else:
        train_clean = train_values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        pred_clean = pred_values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)

        combined = train_clean.copy()
        if not pred_clean.empty:
            combined = pd.concat([combined, pred_clean], ignore_index=True)

        upper = float(np.percentile(combined, 99)) if not combined.empty else 1000.0
        upper = max(upper, 60.0)
        bins = np.linspace(0.0, upper, 36)

        train_hist = build_histogram(train_clean, bins)
        pred_hist = build_histogram(pred_clean, bins)

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Samples train", f"{len(train_clean):,}".replace(",", " "))
        metrics_col2.metric("Samples prédits", f"{len(pred_clean):,}".replace(",", " "))
        mean_pred = float(pred_clean.mean()) if not pred_clean.empty else 0.0
        metrics_col3.metric("Moyenne prédictions", f"{mean_pred:.1f} sec")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Distribution d'entraînement (`train.trip_duration`)**")
            st.bar_chart(train_hist)
        with chart_col2:
            st.markdown("**Distribution des valeurs prédites (`prediction_logs.prediction`)**")
            if pred_hist.empty:
                st.info("Aucune prédiction persistée pour l'instant. Lancez quelques inférences pour alimenter l'histogramme.")
            else:
                st.bar_chart(pred_hist)
