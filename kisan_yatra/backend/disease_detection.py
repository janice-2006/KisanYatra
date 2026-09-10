"""Crop disease detection through the Roboflow hosted model."""

import os
import cv2
import numpy as np
import base64
import requests

try:
    import streamlit as st
except ImportError:
    st = None

ROBOFLOW_MODEL_ID = "crop-disease-axhjj/1"

# ============================================================================
# DISEASE & TREATMENT DATABASE
# ============================================================================

DISEASE_REMEDIES = {
    "powdery_mildew": {
        "name": "Powdery Mildew",
        "description": "White powdery coating on leaves",
        "treatment": [
            "🌿 Spray sulfur fungicide (0.5%)",
            "💨 Improve air circulation",
            "💧 Reduce humidity levels",
            "🧹 Remove infected leaves",
            "🧪 Use potassium bicarbonate spray"
        ],
        "prevention": "Maintain proper spacing, avoid overhead watering"
    },
    "early_blight": {
        "name": "Early Blight",
        "description": "Brown spots with concentric rings on leaves",
        "treatment": [
            "🌿 Apply chlorothalonil fungicide",
            "🧹 Remove affected leaves immediately",
            "💧 Water at soil level only",
            "☀️ Increase sunlight exposure",
            "🧪 Use copper fungicide"
        ],
        "prevention": "Proper spacing, avoid wetting foliage, crop rotation"
    },
    "late_blight": {
        "name": "Late Blight",
        "description": "Water-soaked spots, white mold on undersides",
        "treatment": [
            "🌿 Apply metalaxyl fungicide",
            "🧹 Remove and destroy infected plants",
            "💨 Improve air circulation",
            "💧 Avoid overhead irrigation",
            "🧪 Use copper fungicide"
        ],
        "prevention": "Crop rotation, resistant varieties, proper spacing"
    },
    "leaf_spot": {
        "name": "Leaf Spot",
        "description": "Small dark spots with yellow halos",
        "treatment": [
            "🌿 Apply mancozeb fungicide",
            "🧹 Remove infected leaves",
            "💨 Increase ventilation",
            "💧 Water only at base of plant",
            "🧪 Use copper-based fungicide"
        ],
        "prevention": "Crop rotation, resistant varieties, proper spacing"
    },
    "rust": {
        "name": "Rust",
        "description": "Orange/brown powder-like pustules on leaves",
        "treatment": [
            "🌿 Apply sulfur dust",
            "🧹 Remove heavily infected leaves",
            "💨 Improve air circulation",
            "💧 Reduce overhead watering",
            "🧪 Use tebuconazole fungicide"
        ],
        "prevention": "Resistant varieties, avoid crowding, proper air flow"
    },
    "bacterial_wilt": {
        "name": "Bacterial Wilt",
        "description": "Sudden wilting without visible lesions",
        "treatment": [
            "🚫 No chemical cure - remove infected plants",
            "🧹 Destroy affected plants entirely",
            "🧼 Disinfect tools between cuts",
            "🦟 Control insect vectors",
            "🔄 Practice crop rotation (3 years)"
        ],
        "prevention": "Use resistant varieties, control insects, crop rotation"
    },
    "anthracnose": {
        "name": "Anthracnose",
        "description": "Circular lesions with dark borders and sunken centers",
        "treatment": [
            "🌿 Apply mancozeb fungicide",
            "🧹 Remove and destroy infected fruit/leaves",
            "💨 Improve air circulation",
            "💧 Avoid overhead watering",
            "🧪 Use chlorothalonil"
        ],
        "prevention": "Resistant varieties, crop rotation, proper pruning"
    },
    "healthy": {
        "name": "Healthy Crop",
        "description": "No disease detected",
        "treatment": [
            "✅ Continue regular monitoring",
            "💚 Maintain good cultural practices",
            "🌱 Ensure proper nutrition",
            "💧 Adequate watering",
            "☀️ Sufficient sunlight"
        ],
        "prevention": "Maintain preventive measures"
    }
}

def load_roboflow_client(api_key=None):
    """Create hosted-inference configuration from a parameter, env var, or secret."""
    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key and st is not None:
        try:
            key = st.secrets.get("ROBOFLOW_API_KEY")
        except Exception:
            key = None

    if not key:
        return None, "ROBOFLOW_API_KEY is not configured"
    return {"api_key": key}, None

def detect_disease_roboflow(image, client, confidence_threshold=0.5):
    """Run Roboflow inference and normalize predictions for the existing UI."""
    if client is None:
        return None, "Roboflow client not configured"

    try:
        image_array = np.asarray(image)
        success, encoded_image = cv2.imencode(
            ".jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        )
        if not success:
            return None, "Could not encode uploaded image"

        response = requests.post(
            f"https://serverless.roboflow.com/{ROBOFLOW_MODEL_ID}",
            data=base64.b64encode(encoded_image.tobytes()),
            headers={
                "Authorization": f"Bearer {client['api_key']}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=60,
        )
        response.raise_for_status()
        response = response.json()

        image_height, image_width = image_array.shape[:2]
        detections = []
        for prediction in response.get("predictions", []):
            confidence = float(prediction.get("confidence", 0))
            if confidence < confidence_threshold:
                continue

            center_x = float(prediction["x"])
            center_y = float(prediction["y"])
            box_width = float(prediction["width"])
            box_height = float(prediction["height"])
            detections.append({
                "class": prediction.get("class", "unknown"),
                "confidence": confidence,
                "coordinates": [[
                    max(0, center_x - box_width / 2),
                    max(0, center_y - box_height / 2),
                    min(image_width, center_x + box_width / 2),
                    min(image_height, center_y + box_height / 2),
                ]],
            })

        return detections, None
    except Exception as e:
        return None, f"Roboflow detection error: {str(e)}"

def draw_detections(image, detections):
    """
    Draw bounding boxes on image
    """
    img_copy = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["coordinates"][0])
        conf = detection["confidence"]
        class_name = detection["class"]
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} ({conf:.2f})"
        cv2.putText(img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_copy

