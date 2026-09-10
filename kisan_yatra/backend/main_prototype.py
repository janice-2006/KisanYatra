import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

try:
    from disease_detection import (
        DISEASE_REMEDIES,
        detect_disease_roboflow,
        draw_detections,
        load_roboflow_client,
    )
except ImportError:
    DISEASE_REMEDIES = {}
    detect_disease_roboflow = None
    draw_detections = None
    load_roboflow_client = None

BASE_DIR = Path(__file__).resolve().parent

NPK_ESTIMATES = {
    'Alluvial soil': {'N': 85, 'P': 45, 'K': 45},
    'Black Soil': {'N': 50, 'P': 55, 'K': 65},
    'Clay soil': {'N': 60, 'P': 65, 'K': 70},
    'Red soil': {'N': 40, 'P': 40, 'K': 40},
    'laterite soil': {'N': 100, 'P': 80, 'K': 50},
    'micaceous soil': {'N': 70, 'P': 60, 'K': 40}
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
        
        # Load state dict and ensure model is updated
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        st.success(f"✓ Soil model loaded. Classes: {soil_classes}")
        return model, soil_classes
    except FileNotFoundError as e:
        st.error(f"Error loading soil model: {e}. Ensure model files are in the 'models' folder.")
        return None, None

@st.cache_resource
def load_crop_model():
    try:
        model_path = BASE_DIR / "models" / "crop_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: 'models/crop_model.pkl' not found. Please run the training script first.")
        return None

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

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
            api_key = data.get("API_KEY")
            if api_key:
                return api_key
        except Exception:
            pass

    return None


# Fallback market prices (in Rs per quintal) for crops when API is unavailable
FALLBACK_MARKET_PRICES = {
    'rice': 2500, 'maize': 2000, 'jute': 5500, 'cotton': 5800,
    'coconut': 9000, 'papaya': 3500, 'orange': 4500, 'apple': 6500,
    'muskmelon': 4000, 'watermelon': 3000, 'grapes': 8000, 'mango': 4500,
    'banana': 2800, 'pomegranate': 8500, 'lentil': 5500, 'blackgram': 6800,
    'mungbean': 5900, 'mothbeans': 5200, 'pigeonpeas': 6500, 'kidneybeans': 7200,
    'chickpea': 5500, 'coffee': 12000
}

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
    
    # Try to fetch from API first
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
            # Silently fall back to hardcoded prices
            pass
    
    # Use fallback price if available
    fallback_price = FALLBACK_MARKET_PRICES.get(crop_name.lower())
    if fallback_price:
        st.info(f"Using reference market price: ₹{fallback_price} per Quintal")
        return fallback_price
    
    return None

def create_nutrient_chart(summary_df, nutrient, title, colors, key_prefix):
    summary_sorted = summary_df.sort_values(by=nutrient, ascending=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Highest Requirement", "Lowest Requirement"))
    top_10 = summary_sorted.head(10).sort_values(by=nutrient)
    last_10 = summary_sorted.tail(10)
    fig.add_trace(go.Bar(y=top_10.index, x=top_10[nutrient], name="Most required", marker_color=colors[0], orientation='h'), row=1, col=1)
    fig.add_trace(go.Bar(y=last_10.index, x=last_10[nutrient], name="Least required", marker_color=colors[1], orientation='h'), row=1, col=2)
    fig.update_traces(texttemplate='%{x:.2f}', textposition='inside')
    fig.update_layout(title_text=title, plot_bgcolor='white', font_size=12, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

def render_disease_detection():
    st.header("Crop Disease Detection")
    st.write("Upload a crop image to analyze it with the Roboflow computer-vision model.")

    roboflow_client, roboflow_error = load_roboflow_client()

    uploaded_image = st.file_uploader(
        "Upload a crop or leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="prototype_disease_upload",
        disabled=bool(roboflow_error),
    )
    if roboflow_error:
        st.info("Disease detection is unavailable until the deployment secret is configured.")
        return
    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="prototype_disease_confidence",
    )

    if not uploaded_image:
        return

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded crop image", use_container_width=True)

    if st.button("Detect Disease", type="primary", key="prototype_detect_disease"):
        if load_roboflow_client is None or detect_disease_roboflow is None:
            st.error("Disease detection module is not available.")
            return

        with st.spinner("Analyzing image..."):
            if roboflow_error:
                st.error(roboflow_error)
                return

            detections, error = detect_disease_roboflow(
                np.asarray(image), roboflow_client, confidence
            )
            if error:
                st.error(error)
                return

        if not detections:
            st.info("No disease region was detected. This does not prove the crop is healthy.")
            return

        st.subheader("Detection Results")
        for detection in detections:
            class_name = detection["class"]
            confidence_score = detection["confidence"]
            disease_key = class_name.lower().replace(" ", "_")
            disease_info = DISEASE_REMEDIES.get(disease_key)
            st.write(f"**{class_name}** ({confidence_score:.1%})")
            if disease_info:
                st.write(disease_info["description"])

        annotated_image = draw_detections(np.asarray(image), detections)
        st.image(annotated_image, caption="Annotated detection result", use_container_width=True)
# --------------------------------------------------------------------------
# INITIALIZE MODELS AND DATA
# --------------------------------------------------------------------------

soil_model, soil_classes = load_soil_model()
crop_model = load_crop_model()
cropdf, crop_summary = load_visualization_data()

# --------------------------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------------------------

st.set_page_config(page_title="KisanYatra", page_icon="🌾", layout="wide")
st.title("🌾KisanYatra: Crop Recommendation & Analysis")

if cropdf is None or crop_summary is None:
    st.warning("Could not load crop data. Please ensure 'Crop_recommendation_cleaned.csv' is in the correct folder.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧑‍🌾 Crop Recommender Tool",
    "🦠 Disease Detection",
])

# --- TAB 1: DATA EXPLORATION ---
with tab1:
    st.header("Analysis of Crop and Nutrient Data")
    st.write("Overview of nutrient and climate requirements for various crops.")
    create_nutrient_chart(crop_summary, 'N', "Nitrogen (N) Requirement", ['#0592D0', '#E97451'], key_prefix="nitrogen")
    create_nutrient_chart(crop_summary, 'P', "Phosphorus (P) Requirement", ['#Cd7f32', '#Bdb76b'], key_prefix="phosphorus")
    create_nutrient_chart(crop_summary, 'K', "Potassium (K) Requirement", ['#954535', '#C2b280'], key_prefix="potassium")
    st.subheader("NPK Values Comparison")
    fig_npk = go.Figure()
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['N'], name='Nitrogen', marker_color='indianred'))
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['P'], name='Phosphorous', marker_color='lightsalmon'))
    fig_npk.add_trace(go.Bar(x=crop_summary.index, y=crop_summary['K'], name='Potash', marker_color='crimson'))
    fig_npk.update_layout(plot_bgcolor='white', barmode='group', xaxis_tickangle=-45)
    st.plotly_chart(fig_npk, use_container_width=True, key="npk_comparison_chart")
    st.subheader("Climate Conditions Comparison")
    fig_climate = px.bar(crop_summary, x=crop_summary.index, y=["rainfall", "temperature", "humidity"])
    fig_climate.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig_climate, use_container_width=True, key="climate_comparison_chart")
    st.subheader("Feature Correlation Heatmap")
    fig_heatmap, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(cropdf.drop('label', axis=1).corr(), annot=True, cmap='Wistia', ax=ax)
    st.pyplot(fig_heatmap)

# --- TAB 2: CROP RECOMMENDER TOOL ---
with tab2:
    st.header("Get a Personalized Crop Recommendation")
    col1, col2 = st.columns([1, 1.5])
    estimated_npk = {'N': 90, 'P': 42, 'K': 43}
    with col1:
        st.subheader("1. Predict Soil Nutrients (Optional)")
        uploaded_file = st.file_uploader("Upload a soil image for an NPK estimate.", type=["jpg", "png", "jpeg"])
        if uploaded_file and soil_model and soil_classes:
            st.image(uploaded_file, caption='Uploaded Image.', width=200)
            with st.spinner('Analyzing soil type...'):
                predicted_soil = predict_soil_type(uploaded_file, soil_model, soil_classes)
                st.success(f"Predicted Soil Type: **{predicted_soil}**")
                estimated_npk = NPK_ESTIMATES.get(predicted_soil, estimated_npk)
        elif uploaded_file:
            st.error("Soil analysis model is not loaded. Cannot predict.")
    with col2:
        st.subheader("2. Enter Conditions & Get Recommendation")
        st.write("NPK values are estimated from the image. Adjust them or enter manually.")
        N = st.number_input("Nitrogen (N)", value=estimated_npk['N'])
        P = st.number_input("Phosphorus (P)", value=estimated_npk['P'])
        K = st.number_input("Potassium (K)", value=estimated_npk['K'])
        temperature = st.number_input("Temperature (°C)", value=25.0, format="%.2f")
        humidity = st.number_input("Humidity (%)", value=75.0, format="%.2f")
        ph = st.number_input("pH value", value=6.5, format="%.2f")
        rainfall = st.number_input("Rainfall (mm)", value=150.0, format="%.2f")
        land_area = st.number_input("Land Area (hectares)", min_value=0.1, value=1.0, format="%.2f")
    if st.button("Recommend Crop & Calculate Profit", type="primary", use_container_width=True):
        if crop_model:
            sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=['N','P','K','temperature','humidity','ph','rainfall'])
            prediction = crop_model.predict(sample)
            crop_name = prediction[0].lower()
            st.success(f"**Recommended Crop: {crop_name.title()}**")
            with st.spinner("Fetching live market price..."):
                price_per_qtl = get_market_price(crop_name)
            if price_per_qtl:
                price_per_kg = price_per_qtl / 100.0
                st.write(f"**Live Market Price:** ₹{price_per_qtl:,.2f} per Quintal (₹{price_per_kg:,.2f} per kg)")
                if crop_name in PROFIT_DATA:
                    data = PROFIT_DATA[crop_name]
                    total_yield = data['yield_kg_per_hectare'] * land_area
                    total_cost = data['cost_per_hectare'] * land_area
                    total_revenue = total_yield * price_per_kg
                    total_profit = total_revenue - total_cost
                    st.subheader(f"Estimated Profit Analysis for {land_area} hectares:")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
                    m_col2.metric("Total Cost", f"₹{total_cost:,.2f}")
                    m_col3.metric("Total Profit", f"₹{total_profit:,.2f}", delta=f"{total_profit:,.2f}")
                else:
                    st.warning(f"Profit data not available for '{crop_name}'.")
            else:
                st.error(f"Could not fetch market price for '{crop_name}'.")
        else:
            st.error("Crop recommendation model not loaded.")

# --- TAB 3: DISEASE DETECTION ---
with tab3:
    render_disease_detection()