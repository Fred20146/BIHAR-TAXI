import json
import os
from datetime import datetime
from math import log2
from math import asin, cos, radians, sin, sqrt

import pydeck as pdk
import requests
import streamlit as st


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


st.set_page_config(page_title="BIHAR-TAXI Predictor", page_icon="🚕", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.35rem 0 0;
        opacity: 0.88;
        font-size: 0.98rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
    }
    .metric-card h3,
    .metric-card p,
    .metric-card strong {
        color: #f8fafc;
    }
    .metric-value {
        margin: 0;
        font-size: 1.05rem;
        color: #f8fafc;
    }
    .metric-label {
        margin: 0.4rem 0 0;
        color: #cbd5e1;
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    .validation-card {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: #fee2e2;
        border: 1px solid #fecaca;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 24px rgba(127, 29, 29, 0.25);
    }
    .validation-card strong {
        color: #ffffff;
    }
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
        tooltip={"text": "{name}"},
    )


@st.cache_data(ttl=30)
def fetch_model_metadata(api_base_url: str):
    try:
        metadata_payload = api_request(api_base_url, "/models/metadata")
        return metadata_payload, None
    except requests.exceptions.HTTPError as http_error:
        error_response = http_error.response
        error_status_code = error_response.status_code if error_response is not None else "?"
        details = error_response.text if error_response is not None else ""
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
    else:
        model_version = st.text_input(
            "model_version",
            value="",
            help="Laissez vide pour utiliser la dernière version disponible.",
        )
        model_version = model_version.strip() or None

    st.divider()
    if not model_metadata:
        st.caption("Métadonnées modèles indisponibles actuellement.")

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("Entrée de prédiction")
    vendor_id = st.selectbox("vendor_id", options=[1, 2], index=0)

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
    store_and_fwd_flag = st.selectbox("store_and_fwd_flag", options=["N", "Y"], index=0)

    submitted = st.button("Prédire", type="primary")

    st.markdown('<p class="small-note">Astuce: gardez une distance pickup/dropoff supérieure à 50 m pour passer la validation API.</p>', unsafe_allow_html=True)

with col2:
    st.subheader("Résultat")
    trip_distance_m = haversine_meters(
        float(pickup_latitude),
        float(pickup_longitude),
        float(dropoff_latitude),
        float(dropoff_longitude),
    )
    trip_distance_display = format_distance(trip_distance_m)

    if submitted:
        request_body = {
            "vendor_id": int(vendor_id),
            "pickup_datetime": pickup_datetime,
            "passenger_count": int(passenger_count),
            "pickup_longitude": float(pickup_longitude),
            "pickup_latitude": float(pickup_latitude),
            "dropoff_longitude": float(dropoff_longitude),
            "dropoff_latitude": float(dropoff_latitude),
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
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3 style="margin-top:0; margin-bottom:0.4rem;">Prédiction reçue</h3>
                        <p class="metric-value"><strong>Durée estimée:</strong> {format_seconds_hms(api_response['result'])}</p>
                        <p class="metric-label"><strong>Distance estimée:</strong> {trip_distance_display}</p>
                        <p class="metric-label"><strong>Version du modèle:</strong> {display_model_version}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Réponse brute de l'API")
                st.json(api_response)
            except requests.exceptions.HTTPError as error:
                response = error.response
                status_code = response.status_code if response is not None else 500
                detail_message = "Entrée invalide. Vérifiez les coordonnées et la distance minimale de 50 m."
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
            except requests.exceptions.RequestException as error:
                st.error("Impossible de joindre l'API configurée et aucune API locale n'a été trouvée.")
                st.code(str(error))
                st.caption("Conseil: vérifiez le port de l'API dans la sidebar ou cliquez sur 'Auto-détecter API locale'.")
    else:
        st.info("Renseignez les champs: la carte ci-dessous se met à jour en temps réel. Cliquez sur Prédire pour lancer l'inférence.")

    st.divider()
    st.subheader("Carte du trajet sélectionné")
    st.caption(f"Distance actuelle: {trip_distance_display}")
    trip_deck = build_trip_deck(
        float(pickup_latitude),
        float(pickup_longitude),
        float(dropoff_latitude),
        float(dropoff_longitude),
    )
    st.pydeck_chart(trip_deck, use_container_width=True)
    st.caption("La carte suit en direct les valeurs des sliders (pickup/dropoff). Points non déplaçables.")

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
