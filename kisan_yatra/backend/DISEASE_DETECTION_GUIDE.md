# 🦠 KisanYatra - Crop Disease Detection Integration Guide

## ✨ What's New

Your KisanYatra app now includes **AI-powered Crop Disease Detection** using YOLOv8 computer vision models!

## 🚀 Quick Start

### 1. **Install YOLOv8**

```bash
# Open terminal and run:
pip install ultralytics opencv-python

# Or if you're using conda:
conda install -c conda-forge ultralytics opencv-python
```

### 2. **Using the Disease Detection Feature**

1. Go to **http://localhost:8502**
2. Login to your account
3. Click **🦠 Disease Detection** in the sidebar
4. Upload a crop/leaf image (JPG, PNG, JPEG, WEBP)
5. Select model size (Nano for speed, Medium for accuracy)
6. Click **🔍 Detect Disease**
7. Get instant disease diagnosis with treatment recommendations

## 🎯 Features

### ✅ Real-time Detection
- Analyze crop images instantly
- Supports multiple image formats
- Adjustable confidence thresholds

### 💊 Treatment Database
The system includes remedies for common crop diseases:
- **Powdery Mildew** - Fungal infection with white coating
- **Early Blight** - Brown spots with concentric rings
- **Late Blight** - Water-soaked spots with mold
- **Leaf Spot** - Small dark spots with yellow halos
- **Rust** - Orange/brown powder-like pustules
- **Bacterial Wilt** - Sudden wilting without lesions
- **Anthracnose** - Circular lesions with dark borders

Each disease includes:
- 📝 Description
- 💊 Multiple treatment options
- 🛡️ Prevention strategies
- 🎯 Confidence score from AI model

### 🎨 Annotated Results
- Visual bounding boxes on detected diseases
- Confidence percentages
- Side-by-side comparison

## 📊 Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **Nano** | ⚡ Very Fast | Good | Real-time, Mobile |
| **Small** | 🚀 Fast | Better | Production |
| **Medium** | 🎯 Balanced | Best | High Accuracy Needed |

## 🔧 Advanced Setup

### Option 1: Using Default Pre-trained Model
```python
# Automatically downloads YOLOv8 nano model
model = YOLO('yolov8n.pt')  # ~6MB
model = YOLO('yolov8s.pt')  # ~22MB
model = YOLO('yolov8m.pt')  # ~49MB
```

### Option 2: Using Custom Crop Disease Model

**Step 1:** Download a trained crop disease model from Roboflow

```bash
# Visit: https://universe.roboflow.com
# Search for "crop disease" or "plant disease"
# Download YOLOv8 format model
```

**Step 2:** Place the model in your project

```
models/
├── crop_disease_model.pt
└── efficientnet_b0_model.pth
```

**Step 3:** Update the disease_detection.py

```python
def load_yolo_model(model_path=None):
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO('models/crop_disease_model.pt')
    return model
```

### Option 3: Train Your Own Model

**Step 1:** Prepare your dataset

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**Step 2:** Create dataset.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 7  # number of classes
names: ['powdery_mildew', 'early_blight', 'late_blight', 'leaf_spot', 'rust', 'bacterial_wilt', 'anthracnose']
```

**Step 3:** Train the model

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0  # GPU device ID, use -1 for CPU
)

# Export model
model.export(format='pt')
```

## 📚 Dataset Resources

### Free Crop Disease Datasets

1. **Roboflow Universe** - https://universe.roboflow.com
   - Search: "crop disease", "plant disease", "rice disease"
   - Pre-labeled datasets ready for training

2. **Kaggle Datasets**
   - [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/plantvillage-dataset)
   - [Tomato Disease Dataset](https://www.kaggle.com/datasets/karakaggle/tomato-disease-detection-dataset)

3. **GitHub Repositories**
   - [PlantDoc](https://github.com/pratikmp/plantdoc-object-detection)
   - [Plant Pathology](https://github.com/davisking/dlib)

## 🎓 Working Workflow

```
User uploads image
        ↓
YOLOv8 model analyzes
        ↓
Detects diseases/objects
        ↓
Matches with disease database
        ↓
Displays diagnosis + treatment
        ↓
Provides prevention tips
```

## ⚙️ Troubleshooting

### Issue: "YOLO not installed"
```bash
pip install ultralytics
```

### Issue: Model download fails
```bash
# Manually download from:
# https://github.com/ultralytics/assets/releases
# Place in: ~/.yolov8/
```

### Issue: Out of memory
```python
# Use smaller model
model = YOLO('yolov8n.pt')  # Nano is most efficient

# Or run on CPU (slower but uses less RAM)
results = model(image, device='cpu')
```

### Issue: Slow detection
- Use "Nano" model for speed
- Reduce image size
- Use GPU if available

## 📊 Performance Tips

1. **For Real-time Use**: Use Nano model
2. **For Accuracy**: Use Medium model
3. **For Mobile**: Use Nano + image compression
4. **For Batch Processing**: Use GPU acceleration

## 🔮 Future Enhancements

- [ ] Multi-disease detection in single image
- [ ] Disease severity rating
- [ ] Fertilizer recommendations
- [ ] Regional disease occurrence data
- [ ] Weather integration for disease risk
- [ ] Historical disease tracking per farm
- [ ] Mobile app with offline detection

## 📞 Support Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com
- **Roboflow Docs**: https://docs.roboflow.com
- **OpenCV Docs**: https://docs.opencv.org

## 🌾 Happy Farming! 🚜

Your crop disease detection system is now ready to help farmers make better decisions!
