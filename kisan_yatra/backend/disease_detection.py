"""Crop disease detection through the local trained EfficientNet model."""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pickle

try:
    import streamlit as st
except ImportError:
    st = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ============================================================================
# DISEASE & TREATMENT DATABASE
# ============================================================================
# (Keep your existing DISEASE_REMEDIES dictionary here unchanged)
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

# Cache resource so model loads only once
if st is not None:
    load_cache = st.cache_resource
else:
    load_cache = lambda f: f

@load_cache
def load_disease_model():
    """Load the locally trained EfficientNet-B0 disease classification model."""
    try:
        model_path = MODEL_DIR / "disease_efficientnet_b0.pth"
        classes_path = MODEL_DIR / "disease_classes.pkl"
        
        if not model_path.exists() or not classes_path.exists():
            return None, None, "Model files ('disease_efficientnet_b0.pth' or 'disease_classes.pkl') not found in 'models/' folder."
        
        with open(classes_path, "rb") as f:
            disease_classes = pickle.load(f)
            
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(disease_classes))
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model, disease_classes, None
    except Exception as e:
        return None, None, f"Error loading local model: {str(e)}"

def predict_disease(image, model, class_names):
    """Run local inference on the uploaded PIL image and return predicted class and confidence."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    return predicted_class, confidence.item()

def draw_prediction_label(image, label, confidence):
    """Legacy helper compatibility function for classification."""
    return image