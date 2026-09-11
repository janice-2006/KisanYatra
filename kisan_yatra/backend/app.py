"""
KisanYatra - Crop Recommendation System with User Authentication
Multi-page Streamlit app with agriculture theme
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import json
import hashlib
import math
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from i18n import LANGUAGES, translate, translate_term, translate_trader
from voice import text_to_speech

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None

# Import disease detection module (handles cv2 internally)
try:
    from disease_detection import (
        draw_prediction_label,
        load_disease_model, predict_disease,
        DISEASE_REMEDIES
    )
except ImportError as e:
    st.warning(f"Disease detection module not available: {e}")
    draw_prediction_label = None
    load_disease_model = None
    predict_disease = None
    DISEASE_REMEDIES = {}

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="KisanYatra - Smart Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agriculture-themed color scheme
COLORS = {
    "primary": "#2d5016",      # Dark green
    "secondary": "#6ba547",    # Light green
    "accent": "#f4a460",       # Sandy brown
    "success": "#27ae60",      # Green
    "danger": "#e74c3c",       # Red
    "light_bg": "#f0f4e8",     # Light cream
}


def t(key, default=""):
    return str(translate(st.session_state.get("language", "English"), key, default or key))


def tr(key):
    return translate_trader(st.session_state.get("language", "English"), key)


def render_voice_output(text, key):
    if LANGUAGES[st.session_state.language] == "or":
        render_browser_voice(text, key)
        return
    if st.button(t("listen"), key=key):
        try:
            audio = text_to_speech(text, LANGUAGES[st.session_state.language])
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.warning("Text-to-speech is unavailable. Install gTTS to enable audio.")
        except Exception as error:
            st.warning(f"Text-to-speech failed: {error}")


def render_browser_voice(text, key):
    """Use the browser's installed speech voice when no server voice exists."""
    safe_text = json.dumps(text, ensure_ascii=False)
    speech_locale = {
        "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
        "ml": "ml-IN", "pa": "pa-IN", "bn": "bn-IN", "or": "or-IN",
    }.get(LANGUAGES[st.session_state.language], "en-IN")
    components.html(
        f"""
        <button id="listen-{key}" style="background:#6ba547;color:white;border:0;border-radius:6px;padding:0.6rem 1rem;font-size:16px;cursor:pointer;">
            {t('listen')}
        </button>
        <script>
        const button = document.getElementById("listen-{key}");
        button.onclick = () => {{
            window.speechSynthesis.cancel();
            const text = {safe_text};
            const chunks = text.match(/[^.!?।]+[.!?।]*/g) || [text];
            let index = 0;
            const speakNext = () => {{
                if (index >= chunks.length) return;
                const utterance = new SpeechSynthesisUtterance(chunks[index++].trim());
                utterance.lang = "{speech_locale}";
                utterance.onend = speakNext;
                window.speechSynthesis.speak(utterance);
            }};
            speakNext();
        }};
        </script>
        """,
        height=48,
    )


def page_narration(page_key):
    """Build a complete narration from the current page and saved app state."""
    if page_key == "dashboard":
        crop_name = st.session_state.get("harvested_crop")
        crop_text = translate_term(st.session_state.language, crop_name) if crop_name else t("recommended", "No crop recommendation yet")
        soil_text = st.session_state.get("predicted_soil", "No soil image analyzed yet")
        values = st.session_state.get("dashboard_values", {})
        values_text = ". ".join(
            f"{label}: {values.get(key, 'not entered')}"
            for key, label in [
                ("nitrogen", "Nitrogen"), ("phosphorus", "Phosphorus"),
                ("potassium", "Potassium"), ("temperature", "Temperature"),
                ("humidity", "Humidity"), ("ph", "pH"), ("rainfall", "Rainfall"),
                ("land_area", "Land area"),
            ]
        )
        trader_text = t("recommend", "Use the recommendation button to begin")
        if crop_name:
            trader_text = f"{tr('sell')}. {tr('sell_subtitle')} {tr('permission')} {tr('sorted')} {tr('details')}, {tr('directions')}, {tr('call')}."
        return " ".join([
            t("crop_recommender"), t("analyze_soil"), t("soil_help"), t("upload_soil"),
            f"Predicted soil type: {soil_text}.",
            t("enter_conditions"), "Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH value, Rainfall, Land Area.",
            values_text, t("recommend"), f"{t('recommended')}: {crop_text}.", trader_text,
            t("market_price_qtl"), t("market_price_kg"), t("profit_analysis"),
        ])
    if page_key == "disease":
        return " ".join([
            t("disease_detection"), t("disease_help"), t("upload_crop"),
            t("upload_crop_help"), t("detection_settings"), t("confidence"),
            t("model_status"), t("detect"), t("description"), t("treatment"),
            t("prevention"), t("healthy_guide"), t("no_detection"),
        ])
    if page_key == "data":
        return f"{t('data')}. Explore crop and nutrient requirements, climate patterns, feature correlations, and data visualizations."
    history = get_user_history(st.session_state.user_email) if st.session_state.get("user_email") else []
    history_text = f"There are {len(history)} saved crop recommendations." if history else "There are no saved recommendations yet."
    return f"{t('profile')}. Farmer account and recommendation history. {history_text}"


def render_page_voice(page_key):
    page_content = page_narration(page_key)
    if st.button(t("listen_page"), key="listen_current_page", use_container_width=True):
        try:
            audio = text_to_speech(
                page_content,
                LANGUAGES[st.session_state.language],
            )
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
            else:
                st.warning("Text-to-speech is unavailable. Install gTTS to enable audio.")
        except Exception as error:
            st.error(f"Text-to-speech failed: {error}")


# Custom CSS for agriculture theme
st.markdown(f"""
<style>
    * {{
        margin: 0;
        padding: 0;
    }}
    
    body {{
        background-color: {COLORS['light_bg']};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    .main {{
        background-color: {COLORS['light_bg']};
    }}
    
    .stTabs [data-baseweb="tab-list"] button {{
        background-color: {COLORS['light_bg']};
        color: {COLORS['primary']};
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLORS['primary']};
        border-bottom: 3px solid {COLORS['secondary']};
    }}
    
    .stButton > button {{
        background-color: {COLORS['secondary']};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['primary']};
        box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
    }}
    
    .header-green {{
        color: {COLORS['primary']};
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .section-title {{
        color: {COLORS['primary']};
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {COLORS['secondary']};
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .success-box {{
        background-color: {COLORS['light_bg']};
        border-left: 5px solid {COLORS['success']};
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    .warning-box {{
        background-color: {COLORS['light_bg']};
        border-left: 5px solid {COLORS['danger']};
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    .input-field {{
        border-radius: 8px;
        border: 2px solid {COLORS['secondary']};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & DATA
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
USERS_DB = BASE_DIR / "users_db.json"
USER_HISTORY = BASE_DIR / "user_history"
USER_HISTORY.mkdir(exist_ok=True)

NPK_ESTIMATES = {
    'Alluvial soil': {'N': 85, 'P': 45, 'K': 45},
    'Black Soil': {'N': 50, 'P': 55, 'K': 65},
    'Clay soil': {'N': 60, 'P': 65, 'K': 70},
    'Red soil': {'N': 40, 'P': 40, 'K': 40},
    'Laterite soil': {'N': 100, 'P': 80, 'K': 50},
    'Micaceous soil': {'N': 70, 'P': 60, 'K': 40},
    'Sandy soil': {'N': 35, 'P': 30, 'K': 25}
}

PROFIT_DATA = {
    'rice': {'yield_kg_per_hectare': 4000, 'cost_per_hectare': 17000},
    'maize': {'yield_kg_per_hectare': 2500, 'cost_per_hectare': 15000},
    'jute': {'yield_kg_per_hectare': 2000, 'cost_per_hectare': 12000},
    'cotton': {'yield_kg_per_hectare': 500, 'cost_per_hectare': 30000},
    'coconut': {'yield_kg_per_hectare': 8000, 'cost_per_hectare': 40000},
    'papaya': {'yield_kg_per_hectare': 30000, 'cost_per_hectare': 50000},
    'orange': {'yield_kg_per_hectare': 15000, 'cost_per_hectare': 50000},
    'apple': {'yield_kg_per_hectare': 20000, 'cost_per_hectare': 120000},
    'muskmelon': {'yield_kg_per_hectare': 20000, 'cost_per_hectare': 25000},
    'watermelon': {'yield_kg_per_hectare': 25000, 'cost_per_hectare': 30000},
    'grapes': {'yield_kg_per_hectare': 15000, 'cost_per_hectare': 90000},
    'mango': {'yield_kg_per_hectare': 10000, 'cost_per_hectare': 70000},
    'banana': {'yield_kg_per_hectare': 40000, 'cost_per_hectare': 60000},
    'pomegranate': {'yield_kg_per_hectare': 10000, 'cost_per_hectare': 80000},
    'lentil': {'yield_kg_per_hectare': 1200, 'cost_per_hectare': 10000},
    'blackgram': {'yield_kg_per_hectare': 900, 'cost_per_hectare': 8000},
    'mungbean': {'yield_kg_per_hectare': 1000, 'cost_per_hectare': 9000},
    'mothbeans': {'yield_kg_per_hectare': 800, 'cost_per_hectare': 7500},
    'pigeonpeas': {'yield_kg_per_hectare': 1500, 'cost_per_hectare': 11000},
    'kidneybeans': {'yield_kg_per_hectare': 1300, 'cost_per_hectare': 10500},
    'chickpea': {'yield_kg_per_hectare': 1600, 'cost_per_hectare': 12000},
    'coffee': {'yield_kg_per_hectare': 750, 'cost_per_hectare': 80000}
}

FALLBACK_MARKET_PRICES = {
    'rice': 2500, 'maize': 2000, 'jute': 5500, 'cotton': 5800,
    'coconut': 9000, 'papaya': 3500, 'orange': 4500, 'apple': 6500,
    'muskmelon': 4000, 'watermelon': 3000, 'grapes': 8000, 'mango': 4500,
    'banana': 2800, 'pomegranate': 8500, 'lentil': 5500, 'blackgram': 6800,
    'mungbean': 5900, 'mothbeans': 5200, 'pigeonpeas': 6500, 'kidneybeans': 7200,
    'chickpea': 5500, 'coffee': 12000
}

DEFAULT_FARMER_LOCATION = {"latitude": 12.9716, "longitude": 77.5946}
TRADER_DIRECTORY = [
    {
        "name": "Sri Lakshmi Traders", "type": "Crop trader", "latitude": 12.9990,
        "longitude": 77.6010, "crops": ["rice", "maize", "wheat"], "price": 2250,
        "address": "Avenue Road Wholesale Market, Bengaluru", "hours": "6:00 AM - 8:00 PM",
        "phone": "+918012345678",
    },
    {
        "name": "Green Agro Traders", "type": "Agricultural buyer", "latitude": 12.9352,
        "longitude": 77.6245, "crops": ["rice", "maize", "chickpea", "cotton"], "price": 2200,
        "address": "BTM Agricultural Yard, Bengaluru", "hours": "7:00 AM - 7:00 PM",
        "phone": "+918087654321",
    },
    {
        "name": "Farmers Market / Wholesale Market", "type": "Wholesale market", "latitude": 13.0358,
        "longitude": 77.5970, "crops": ["rice", "maize", "banana", "tomato"], "price": 2180,
        "address": "Yeshwanthpur APMC Market, Bengaluru", "hours": "5:00 AM - 6:00 PM",
        "phone": "+918076543210",
    },
    {
        "name": "Namma Harvest Buyers", "type": "Direct buyer", "latitude": 12.9050,
        "longitude": 77.5850, "crops": ["rice", "ragi", "maize", "pomegranate"], "price": 2140,
        "address": "Jigani Industrial Area, Bengaluru", "hours": "8:00 AM - 6:00 PM",
        "phone": "+918098765432",
    },
]


def get_google_maps_key():
    try:
        return st.secrets.get("GOOGLE_MAPS_API_KEY")
    except Exception:
        return os.getenv("GOOGLE_MAPS_API_KEY")


def fetch_live_traders(crop_name, latitude, longitude):
    """Search real nearby buyers through Google Places Text Search."""
    api_key = get_google_maps_key()
    if not api_key:
        return []

    try:
        response = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": (
                    "places.displayName,places.formattedAddress,places.location,"
                    "places.nationalPhoneNumber,places.regularOpeningHours,places.googleMapsUri"
                ),
            },
            json={
                "textQuery": f"{crop_name} trader agricultural buyer wholesale market",
                "languageCode": "en",
                "maxResultCount": 10,
                "locationBias": {
                    "circle": {
                        "center": {"latitude": latitude, "longitude": longitude},
                        "radius": 50000,
                    }
                },
            },
            timeout=10,
        )
        response.raise_for_status()
        live_traders = []
        for place in response.json().get("places", []):
            location = place.get("location", {})
            trader_latitude = location.get("latitude")
            trader_longitude = location.get("longitude")
            if trader_latitude is None or trader_longitude is None:
                continue
            hours = place.get("regularOpeningHours", {}).get("weekdayDescriptions", [])
            live_traders.append({
                "name": place.get("displayName", {}).get("text", "Local buyer"),
                "type": "Nearby agricultural buyer or market",
                "latitude": trader_latitude,
                "longitude": trader_longitude,
                "crops": [crop_name],
                "price": get_market_price(crop_name),
                "address": place.get("formattedAddress", "Address unavailable"),
                "hours": "; ".join(hours) if hours else "Hours unavailable",
                "phone": place.get("nationalPhoneNumber", "Not available"),
                "maps_uri": place.get("googleMapsUri"),
            })
        return live_traders
    except (requests.RequestException, ValueError, TypeError):
        return []


def distance_km(latitude_one, longitude_one, latitude_two, longitude_two):
    """Return great-circle distance between two GPS coordinates."""
    earth_radius_km = 6371
    lat_one, lat_two = math.radians(latitude_one), math.radians(latitude_two)
    delta_lat = math.radians(latitude_two - latitude_one)
    delta_lon = math.radians(longitude_two - longitude_one)
    value = (math.sin(delta_lat / 2) ** 2
             + math.cos(lat_one) * math.cos(lat_two) * math.sin(delta_lon / 2) ** 2)
    return earth_radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def nearby_traders(crop_name, latitude, longitude):
    live_traders = fetch_live_traders(crop_name, latitude, longitude)
    source_traders = live_traders or [trader.copy() for trader in TRADER_DIRECTORY if crop_name in trader["crops"]]
    traders = []
    for trader in source_traders:
        trader = trader.copy()
        trader["distance_km"] = distance_km(latitude, longitude, trader["latitude"], trader["longitude"])
        trader["travel_minutes"] = max(4, round(trader["distance_km"] * 2.5))
        traders.append(trader)
    return sorted(traders, key=lambda trader: trader["distance_km"])


def render_sell_harvest(crop_name):
    """Show the location-gated trader search for the predicted crop."""
    st.markdown("---")
    st.markdown(f"### {tr('sell')}")
    st.caption(tr("sell_subtitle"))

    if not st.session_state.get("trader_search_started"):
        st.info(f"{tr('find')}\n\n{tr('permission')}")
        if streamlit_geolocation is None:
            st.warning(tr("location_missing"))
            if st.button(tr("allow_fallback"), key="use_default_location"):
                st.session_state.trader_search_started = True
                st.session_state.farmer_location = DEFAULT_FARMER_LOCATION
                st.rerun()
        else:
            location = streamlit_geolocation()
            if location and location.get("latitude") and location.get("longitude"):
                st.session_state.trader_search_started = True
                st.session_state.farmer_location = {
                    "latitude": location["latitude"], "longitude": location["longitude"]
                }
                st.rerun()
        if st.button(tr("later"), key="maybe_later_traders"):
            st.session_state.trader_search_started = False
            st.info(tr("later"))
        return

    farmer_location = st.session_state.get("farmer_location", DEFAULT_FARMER_LOCATION)
    traders = nearby_traders(crop_name, farmer_location["latitude"], farmer_location["longitude"])
    display_crop_name = translate_term(st.session_state.language, crop_name)
    st.markdown(f"### {tr('nearby')} {display_crop_name}")
    if get_google_maps_key():
        st.caption(f"{tr('sorted')} Live nearby places from Google Maps. Buying prices may need confirmation with the trader.")
    else:
        st.caption(f"{tr('sorted')} {tr('prototype_note')}")

    if not traders:
        st.warning(f"{tr('no_traders')} {display_crop_name}.")
        return

    map_points = pd.DataFrame([
        {"latitude": farmer_location["latitude"], "longitude": farmer_location["longitude"]}
    ] + [
        {"latitude": trader["latitude"], "longitude": trader["longitude"]} for trader in traders
    ])
    st.map(map_points, latitude="latitude", longitude="longitude", zoom=11, use_container_width=True)

    for index, trader in enumerate(traders):
        label = tr("nearest") if index == 0 else tr("nearby_trader")
        st.markdown(f"#### {label} · {trader['name']}")
        crop_labels = ", ".join(str(translate_term(st.session_state.language, crop)) for crop in trader["crops"])
        st.write(f"{tr('buys')}: {crop_labels}")
        st.write(f"📍 **{trader['distance_km']:.1f} {tr('away')}** · 🚜 {tr('approx')} **{trader['travel_minutes']} {tr('minutes')}**")
        price_text = f"₹{trader['price']:,} / {tr('quintal')}" if trader.get("price") else "Contact trader for current price"
        st.write(f"{tr('price')}: **{price_text}**")
        detail_col, direction_col = st.columns(2)
        with detail_col:
            with st.expander(tr("details"), expanded=index == 0):
                st.write(f"**{tr('type')}:** {trader['type']}")
                st.write(f"**{tr('address')}:** {trader['address']}")
                st.write(f"**{tr('hours')}:** {trader['hours']}")
                st.write(f"**{tr('contact')}:** {trader['phone']}")
                st.markdown(f"[{tr('call')}](tel:{trader['phone']})")
        with direction_col:
            directions_url = (
                "https://www.google.com/maps/dir/?api=1"
                f"&origin={farmer_location['latitude']},{farmer_location['longitude']}"
                f"&destination={trader['latitude']},{trader['longitude']}"
            )
            st.link_button(tr("directions"), directions_url, use_container_width=True)

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users_db():
    """Load users database"""
    if USERS_DB.exists():
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users_db(users):
    """Save users database"""
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=4)

def register_user(email, password, name):
    """Register a new user"""
    users = load_users_db()
    if email in users:
        return False, "Email already registered"
    
    users[email] = {
        "name": name,
        "password": hash_password(password),
        "registered_at": datetime.now().isoformat()
    }
    save_users_db(users)
    return True, "Registration successful"

def authenticate_user(email, password):
    """Authenticate user"""
    users = load_users_db()
    if email not in users:
        return False, "Email not found"
    
    if users[email]["password"] != hash_password(password):
        return False, "Incorrect password"
    
    return True, users[email]["name"]

def save_user_history(email, recommendation):
    """Save user's crop recommendation"""
    user_file = USER_HISTORY / f"{email}.json"
    history = []
    
    if user_file.exists():
        with open(user_file, 'r') as f:
            history = json.load(f)
    
    recommendation['timestamp'] = datetime.now().isoformat()
    history.append(recommendation)
    
    with open(user_file, 'w') as f:
        json.dump(history, f, indent=4)

def get_user_history(email):
    """Get user's recommendation history"""
    user_file = USER_HISTORY / f"{email}.json"
    if user_file.exists():
        with open(user_file, 'r') as f:
            return json.load(f)
    return []

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_data
def load_visualization_data():
    try:
        csv_path = BASE_DIR / "Crop_recommendation_cleaned.csv"
        df = pd.read_csv(csv_path)
        summary = pd.pivot_table(df, index=['label'], aggfunc='mean')
        return df, summary
    except Exception:
        return None, None

@st.cache_resource(show_spinner=True)
def load_soil_model():
    try:
        classes_path = BASE_DIR / "models" / "soil_classes.pkl"
        model_path = BASE_DIR / "models" / "efficientnet_b0_model.pth"
        with open(classes_path, 'rb') as f:
            soil_classes = pickle.load(f)
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(soil_classes))
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, soil_classes
    except FileNotFoundError as e:
        st.error(f"Error loading soil model: {e}")
        return None, None

@st.cache_resource
def load_crop_model():
    try:
        model_path = BASE_DIR / "models" / "crop_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: 'models/crop_model.pkl' not found.")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_soil_type(image_uploader, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_uploader).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    return class_names[predicted_idx.item()]


def rank_crop_recommendations(crop_model, sample, limit=3):
    """Return the model's top crop candidates instead of hiding uncertainty."""
    if hasattr(crop_model, "predict_proba"):
        probabilities = crop_model.predict_proba(sample)[0]
        labels = crop_model.classes_
        ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
        return [
            {"crop": str(label).lower(), "confidence": float(probability)}
            for label, probability in ranked[:limit]
        ]
    prediction = crop_model.predict(sample)[0]
    return [{"crop": str(prediction).lower(), "confidence": None}]

def get_api_key():
    try:
        return st.secrets["API_KEY"]
    except Exception:
        pass
    env_key = os.getenv("API_KEY")
    if env_key:
        return env_key
    secrets_path = BASE_DIR / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
            with open(secrets_path, 'rb') as f:
                data = tomllib.load(f)
            return data.get("API_KEY")
        except Exception:
            pass
    return None

def get_market_price(crop_name):
    commodity_name_map = {
        'apple': 'Apple', 'banana': 'Banana', 'blackgram': 'Black Gram (Urd Beans)(Whole)',
        'chickpea': 'Bengal Gram(Gram)(Whole)', 'coconut': 'Coconut', 'coffee': 'Coffee',
        'cotton': 'Cotton', 'grapes': 'Grapes', 'jute': 'Jute',
        'kidneybeans': 'Rajmah(Kabulichana)(Whole)', 'lentil': 'Lentil (Masur)(Whole)',
        'maize': 'Maize', 'mango': 'Mango (Raw-Ripe)', 'mungbean': 'Green Gram (Moong)(Whole)',
        'orange': 'Orange', 'papaya': 'Papaya', 'pigeonpeas': 'Arhar (Tur/Red Gram)(Whole)',
        'pomegranate': 'Pomegranate', 'rice': 'Rice', 'watermelon': 'Water Melon',
    }
    official_crop_name = commodity_name_map.get(crop_name.lower(), crop_name.title())
    api_key = get_api_key()
    
    if api_key:
        try:
            BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            params = {'api-key': api_key, 'format': 'json', 'limit': 5, 'filters[commodity]': official_crop_name}
            response = requests.get(BASE_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get('records'):
                for record in data['records']:
                    price_str = record.get('modal_price', '0')
                    if price_str.replace('.', '', 1).isdigit():
                        return float(price_str)
        except (requests.exceptions.RequestException, ValueError, TypeError):
            pass
    
    fallback_price = FALLBACK_MARKET_PRICES.get(crop_name.lower())
    if fallback_price:
        return fallback_price
    return None

# ============================================================================
# PAGE: LOGIN
# ============================================================================

def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='header-green'>🌾 KisanYatra</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Smart Crop Recommendation System</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        # LOGIN TAB
        with tab1:
            st.markdown("### Welcome Back, Farmer! 🚜")
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔐 Password", type="password", placeholder="Enter your password")
            
            if st.button("Login", use_container_width=True):
                if email and password:
                    success, message = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.user_name = message
                        st.success(f"Welcome back, {message}! 🎉")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
        
        # SIGNUP TAB
        with tab2:
            st.markdown("### Join Our Community 👨‍🌾")
            name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            email = st.text_input("📧 Email", placeholder="Enter your email", key="signup_email")
            password = st.text_input("🔐 Password", type="password", placeholder="Create a password", key="signup_pass")
            confirm_password = st.text_input("🔐 Confirm Password", type="password", placeholder="Confirm password")
            
            if st.button("Sign Up", use_container_width=True):
                if name and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        success, message = register_user(email, password, name)
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")

# ============================================================================
# PAGE: DASHBOARD (CROP RECOMMENDER)
# ============================================================================

def page_dashboard():
    st.markdown(f"<h2 class='section-title'>{t('crop_recommender')}</h2>", unsafe_allow_html=True)
    
    soil_model, soil_classes = load_soil_model()
    crop_model = load_crop_model()
    
    col1, col2 = st.columns([1, 1.5], gap="medium")
    
    estimated_npk = {'N': 90, 'P': 42, 'K': 43}
    
    with col1:
        st.markdown(f"#### {t('analyze_soil')}")
        st.markdown(t("soil_help"))
        
        uploaded_file = st.file_uploader(t("upload_soil"), type=["jpg", "png", "jpeg"], key="soil_upload")
        if uploaded_file and soil_model and soil_classes:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            with st.spinner('Analyzing soil type...'):
                predicted_soil = predict_soil_type(uploaded_file, soil_model, soil_classes)
                st.session_state.predicted_soil = predicted_soil
                st.markdown(f'<div class="success-box"><b>🌱 Predicted Soil Type:</b> <span style="color: #2d5016; font-size: 1.2rem;">{predicted_soil}</span></div>', unsafe_allow_html=True)
                estimated_npk = NPK_ESTIMATES.get(predicted_soil, estimated_npk)
        elif not uploaded_file:
            st.session_state.pop("harvested_crop", None)
            st.session_state.pop("recommended_crops", None)
    
    with col2:
        st.markdown(f"#### {t('enter_conditions')}")
        
        col_npk1, col_npk2, col_npk3 = st.columns(3)
        with col_npk1:
            N = st.number_input("Nitrogen (N)", value=estimated_npk['N'], min_value=0, max_value=150)
        with col_npk2:
            P = st.number_input("Phosphorus (P)", value=estimated_npk['P'], min_value=0, max_value=150)
        with col_npk3:
            K = st.number_input("Potassium (K)", value=estimated_npk['K'], min_value=0, max_value=150)
        
        col_cond1, col_cond2 = st.columns(2)
        with col_cond1:
            temperature = st.number_input("🌡️ Temperature (°C)", value=25.0, format="%.2f")
            humidity = st.number_input("💧 Humidity (%)", value=75.0, format="%.2f", max_value=100.0)
        with col_cond2:
            ph = st.number_input("pH value", value=6.5, format="%.2f", min_value=0.0, max_value=14.0)
            rainfall = st.number_input("☔ Rainfall (mm)", value=150.0, format="%.2f")
        
        land_area = st.number_input("🏞️ Land Area (hectares)", min_value=0.1, value=1.0, format="%.2f")

    st.session_state.dashboard_values = {
        "nitrogen": N, "phosphorus": P, "potassium": K,
        "temperature": temperature, "humidity": humidity, "ph": ph,
        "rainfall": rainfall, "land_area": land_area,
    }

    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )

    soil_detected = bool(uploaded_file and soil_model and soil_classes and st.session_state.get("predicted_soil"))
    if soil_detected and crop_model:
        automatic_recommendations = rank_crop_recommendations(crop_model, sample)
        st.session_state.recommended_crops = automatic_recommendations
        st.session_state.harvested_crop = automatic_recommendations[0]["crop"]
        st.markdown("### 🌱 Top Crop Recommendations")
        for index, recommendation in enumerate(automatic_recommendations, 1):
            display_name = translate_term(st.session_state.language, recommendation["crop"])
            confidence = recommendation["confidence"]
            confidence_text = f" ({confidence:.1%} model score)" if confidence is not None else ""
            st.write(f"**{index}. {display_name}**{confidence_text}")
        st.caption("These are model rankings based on the detected soil profile and entered climate conditions. Confirm local agronomy before planting.")
    
    st.markdown("---")
    
    if st.button(t("recommend"), type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Upload a soil image first so the crop recommendation is based on detected soil.")
        elif not soil_detected:
            st.warning("Upload a soil image and wait for soil type detection before requesting crop recommendations.")
        elif crop_model:
            recommendations = rank_crop_recommendations(crop_model, sample)
            st.session_state.recommended_crops = recommendations
            crop_name = recommendations[0]["crop"]
            st.session_state.harvested_crop = crop_name
            st.session_state.trader_search_started = False
            
            # Save to history
            recommendation = {
                "crop": crop_name,
                "npk": {"N": N, "P": P, "K": K},
                "conditions": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                    "land_area": land_area
                }
            }
            save_user_history(st.session_state.user_email, recommendation)
            
            # Display results
            display_crop_name = translate_term(st.session_state.language, crop_name)
            st.markdown(f'<div class="success-box"><h3>{t("recommended")}: <span style="color: {COLORS["secondary"]};">{display_crop_name}</span></h3></div>', unsafe_allow_html=True)
            render_voice_output(
                f"{t('recommended')}: {display_crop_name}",
                "speak_crop_recommendation",
            )
            
            with st.spinner("Fetching market price..."):
                price_per_qtl = get_market_price(crop_name)
            
            if price_per_qtl:
                price_per_kg = price_per_qtl / 100.0
                
                col_price1, col_price2 = st.columns(2)
                with col_price1:
                    st.metric(t("market_price_qtl"), f"₹{price_per_qtl:,.2f}")
                with col_price2:
                    st.metric(t("market_price_kg"), f"₹{price_per_kg:,.2f}")
                
                if crop_name in PROFIT_DATA:
                    data = PROFIT_DATA[crop_name]
                    total_yield = data['yield_kg_per_hectare'] * land_area
                    total_cost = data['cost_per_hectare'] * land_area
                    total_revenue = total_yield * price_per_kg
                    total_profit = total_revenue - total_cost
                    
                    st.markdown(f"<h3 style='color: {COLORS['primary']}; text-align: center;'>{t('profit_analysis')} ({land_area} hectares)</h3>", unsafe_allow_html=True)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("📊 Total Yield", f"{total_yield:,.0f} kg")
                    with col_m2:
                        st.metric("💸 Total Cost", f"₹{total_cost:,.2f}")
                    with col_m3:
                        st.metric("💵 Total Profit", f"₹{total_profit:,.2f}", delta=f"{(total_profit/total_cost*100):.1f}%" if total_cost > 0 else None)

    harvested_crop = st.session_state.get("harvested_crop")
    if harvested_crop:
        render_sell_harvest(harvested_crop)

# ============================================================================
# PAGE: DATA EXPLORATION
# ============================================================================

def page_data_exploration():
    st.markdown("<h2 class='section-title'>📊 Data Analysis & Insights</h2>", unsafe_allow_html=True)
    st.markdown("Explore comprehensive crop and nutrient requirements data")
    
    cropdf, crop_summary = load_visualization_data()
    
    if cropdf is None or crop_summary is None:
        st.error("Could not load crop data")
        return
    
    # Nutrient requirements
    st.markdown("### 🥗 Nutrient Requirements by Crop")
    col_n, col_p, col_k = st.columns(3)
    
    with col_n:
        st.markdown("#### Nitrogen (N)")
        fig_n = px.bar(crop_summary.sort_values('N', ascending=False).head(10), 
                       y='N', title="Top 10 Crops", color_discrete_sequence=[COLORS['secondary']])
        st.plotly_chart(fig_n, use_container_width=True)
    
    with col_p:
        st.markdown("#### Phosphorus (P)")
        fig_p = px.bar(crop_summary.sort_values('P', ascending=False).head(10), 
                       y='P', title="Top 10 Crops", color_discrete_sequence=[COLORS['accent']])
        st.plotly_chart(fig_p, use_container_width=True)
    
    with col_k:
        st.markdown("#### Potassium (K)")
        fig_k = px.bar(crop_summary.sort_values('K', ascending=False).head(10), 
                       y='K', title="Top 10 Crops", color_discrete_sequence=[COLORS['primary']])
        st.plotly_chart(fig_k, use_container_width=True)
    
    # NPK Comparison
    st.markdown("### 📈 NPK Comparison Across All Crops")
    fig_npk = go.Figure()
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['N'], name='Nitrogen', marker_color=COLORS['primary']))
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['P'], name='Phosphorous', marker_color=COLORS['accent']))
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['K'], name='Potash', marker_color=COLORS['secondary']))
    fig_npk.update_layout(plot_bgcolor='white', barmode='group', xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_npk, use_container_width=True)
    
    # Climate Conditions
    st.markdown("### 🌍 Climate Conditions by Crop")
    fig_climate = px.bar(crop_summary, x=crop_summary.index, y=["rainfall", "temperature", "humidity"],
                         color_discrete_map={"rainfall": COLORS['primary'], "temperature": COLORS['accent'], "humidity": COLORS['secondary']})
    fig_climate.update_layout(plot_bgcolor='white', height=400)
    st.plotly_chart(fig_climate, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("### 🔗 Feature Correlation Heatmap")
    fig_heatmap, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cropdf.drop('label', axis=1).corr(), annot=True, cmap='YlGn', ax=ax, cbar_kws={'label': 'Correlation'})
    st.pyplot(fig_heatmap)

# ============================================================================
# PAGE: USER PROFILE & HISTORY
# ============================================================================

def page_profile():
    st.markdown(f"<h2 class='section-title'>👤 Your Profile</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### Welcome, {st.session_state.user_name}! 👋")
        st.markdown(f"**📧 Email:** {st.session_state.user_email}")
        st.markdown(f"**🌾 Farmer Account**")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.rerun()
    
    with col2:
        st.markdown("### 📋 Your Recommendations History")
        history = get_user_history(st.session_state.user_email)
        
        if history:
            for i, record in enumerate(reversed(history[-10:])):  # Show last 10
                with st.expander(f"🌱 {record['crop'].upper()} - {record['timestamp'][:10]}"):
                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        st.markdown(f"**NPK Values:**")
                        st.text(f"N: {record['npk']['N']}\nP: {record['npk']['P']}\nK: {record['npk']['K']}")
                    with col_h2:
                        st.markdown(f"**Conditions:**")
                        cond = record['conditions']
                        st.text(f"Temp: {cond['temperature']}°C\nHumidity: {cond['humidity']}%\nRainfall: {cond['rainfall']}mm")
        else:
            st.info("No recommendations yet. Start exploring the recommender tool!")

# ============================================================================
# PAGE: DISEASE DETECTION
# ============================================================================

def page_disease_detection():
    st.markdown(f"<h2 class='section-title'>{t('disease_detection')}</h2>", unsafe_allow_html=True)
    st.markdown(t("disease_help"))
    
    st.info("💡 **Disease Detection Feature**: Powered by your locally trained EfficientNet model.")
    
    model, class_names, model_error = load_disease_model() if load_disease_model else (None, None, "Module not available")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### {t('upload_crop')}")
        uploaded_file = st.file_uploader(
            t("upload_crop_help"),
            type=["jpg", "png", "jpeg", "webp"],
            key="disease_upload",
            disabled=bool(model_error),
        )

        if model_error:
            st.caption(f"⚠️ {model_error}")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.markdown(f"### {t('model_status')}")
        if not model_error and model is not None:
            st.success("✅ Local classification model is ready!")
            st.caption(f"Loaded {len(class_names)} disease categories.")
        else:
            st.warning("⚠️ Local model not loaded. Ensure `disease_efficientnet_b0.pth` and `disease_classes.pkl` are inside your `models/` directory.")
    
    # Run prediction if image is uploaded and model is loaded
    if uploaded_file and not model_error and model is not None:
        st.markdown("---")
        
        if st.button(t("detect"), type="primary", use_container_width=True):
            with st.spinner("🤔 Analyzing image for diseases..."):
                try:
                    predicted_class, confidence = predict_disease(image, model, class_names)
                    disease_key = predicted_class.lower().replace(" ", "_")
                    display_disease_name = translate_term(st.session_state.language, disease_key)
                    
                    # Get disease info from dictionary
                    disease_info = DISEASE_REMEDIES.get(disease_key, DISEASE_REMEDIES.get(predicted_class))
                    
                    with st.expander(f"🔴 {display_disease_name} (Confidence: {confidence:.1%})", expanded=True):
                        col_d1, col_d2 = st.columns([1, 1])
                        
                        with col_d1:
                            st.markdown(f"**{t('description')}:**")
                            st.write(disease_info['description'] if disease_info else "Disease class predicted by the local model.")
                            st.markdown(f"**Confidence Score:** `{confidence:.1%}`")
                        
                        with col_d2:
                            if disease_info:
                                st.markdown(f"**{t('treatment')}:**")
                                for treatment in disease_info['treatment']:
                                    st.write(treatment)
                            else:
                                st.info("Add this class key to DISEASE_REMEDIES for customized treatment guidance.")
                        
                        if disease_info:
                            st.markdown(f"**{t('prevention')}:** {disease_info['prevention']}")
                            render_voice_output(
                                f"{display_disease_name}. {disease_info.get('description', '')}. "
                                f"{t('treatment')}: {'; '.join(disease_info.get('treatment', []))}",
                                "speak_disease_result",
                            )
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
# ============================================================================
# MAIN APP LOGIC
# ============================================================================

def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.user_name = None
    
    if "language" not in st.session_state:
        st.session_state.language = "English"

    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    elif st.session_state.current_page not in {"dashboard", "disease", "data", "profile"}:
        st.session_state.current_page = "dashboard"
    
    # Sidebar
    if st.session_state.logged_in:
        with st.sidebar:
            st.markdown(f"<h3 style='color: {COLORS['primary']};'>🌾 KisanYatra</h3>", unsafe_allow_html=True)
            st.markdown(f"**👤 {st.session_state.user_name}**")
            st.markdown("---")

            st.session_state.language = st.selectbox(
                t("language"), list(LANGUAGES), key="language_selector"
            )

            page_labels = {
                "dashboard": t("dashboard"),
                "disease": t("disease"),
                "data": t("data"),
                "profile": t("profile"),
            }
            
            st.session_state.current_page = st.radio(
                t("navigation"),
                list(page_labels),
                format_func=lambda page: page_labels[page],
                label_visibility="collapsed"
            )

            render_page_voice(st.session_state.current_page)
            
            st.markdown("---")
            if st.button(t("logout")):
                st.session_state.logged_in = False
                st.rerun()
    
    # Router
    if not st.session_state.logged_in:
        page_login()
    else:
        if st.session_state.current_page == "dashboard":
            page_dashboard()
        elif st.session_state.current_page == "disease":
            page_disease_detection()
        elif st.session_state.current_page == "data":
            page_data_exploration()
        elif st.session_state.current_page == "profile":
            page_profile()

if __name__ == "__main__":
    main()