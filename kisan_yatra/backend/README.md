# 🌾 KisanYatra - Smart Farming Assistant

**A comprehensive AI-powered agricultural platform for crop recommendations and disease detection**

## 🎯 Features

### 🌱 Crop Recommendation Engine
- Analyze soil nutrients using soil image analysis (EfficientNet)
- Input environmental conditions (temperature, humidity, rainfall, pH)
- Get personalized crop recommendations based on ML model
- View profit analysis for recommended crops
- Check live market prices

### 🦠 Disease Detection System
- Upload crop/leaf images for instant disease analysis
- AI-powered detection using YOLOv8 computer vision
- Comprehensive treatment recommendations
- Prevention strategies for each disease
- Confidence scores for reliability

### 📊 Data Analytics
- Explore crop nutrient requirements
- Analyze climate conditions for different crops
- View correlations between environmental factors
- Interactive visualizations and charts

### 👤 User Management
- Secure registration and authentication
- Personal account dashboard
- History of all crop recommendations
- Profile management

## 📋 Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for YOLO)
- GPU optional (for faster disease detection)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Automatic installation (recommended)
python install_dependencies.py

# OR Manual installation
pip install streamlit pandas numpy scikit-learn lightgbm torch torchvision
pip install pillow matplotlib seaborn plotly ultralytics opencv-python requests
pip install gTTS
pip install streamlit-geolocation
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Access in Browser

```
http://localhost:8502
```

### 4. Sign Up & Explore

- Create an account with your email
- Login with your credentials
- Start using all features!

### 🌐 Languages and Audio

Use the language selector in the sidebar to switch the interface between English,
 हिन्दी, தமிழ், తెలుగు, മലയാളം, ਪੰਜਾਬી, and বাংলা. Recommendation and disease results can be read aloud
as text in the selected language. Use the **Listen** button to hear translated
recommendations and disease guidance. Audio requires an internet connection;
microphone input is not used. Odia uses the browser's built-in speech voice
because the server-side TTS providers do not currently offer an Odia voice.

## 📁 Project Structure

```
kisan_yatra/backend/
├── app.py                           # Main Streamlit application
├── disease_detection.py             # Disease detection module (YOLO)
├── train_model.py                   # Crop recommendation model training
├── train_soil_model.py              # Soil classification model training
├── install_dependencies.py          # Dependency installation script
│
├── models/                          # Pre-trained models
│   ├── crop_model.pkl              # Crop recommendation model
│   ├── efficientnet_b0_model.pth   # Soil classification model
│   └── soil_classes.pkl            # Soil type classes
│
├── soil\ data\ pics/               # Training images for soil types
│   ├── alluvial soil 1.jpg
│   ├── black soil 1.jpg
│   ├── clay soil 1.jpg
│   └── ...
│
├── Crop_recommendation_cleaned.csv # Crop dataset
├── soil_values_npk.csv            # NPK values reference
├── secrets.toml                    # API keys
├── users_db.json                   # User database
├── user_history/                   # User recommendation history
│
├── DISEASE_DETECTION_GUIDE.md      # Disease detection documentation
└── README.md                        # This file
```

## 🔧 Configuration

### API Configuration

Edit `secrets.toml`:

```toml
API_KEY = "your_api_key_here"
```

Get API key from: https://data.gov.in/

### Live Nearby Trader Search

The trader feature uses realistic Bangalore mock traders by default. For live
nearby agricultural buyers and markets, enable Google Maps Platform **Places
API (New)** and add this secret in Streamlit Cloud or `.streamlit/secrets.toml`:

```toml
GOOGLE_MAPS_API_KEY = "your-google-maps-key"
```

Enable billing and restrict the key to the Places API and your deployed domain.
The app searches around the farmer's GPS location using the already predicted
crop, then sorts results by distance. **Get Directions** opens Google Maps with
the farmer's location as the origin and the selected trader as the destination.
Never commit the key to GitHub.

### Model Configuration

The app uses pre-trained models:
- **Crop Recommendation**: Random Forest classifier with top-three ranking
- **Soil Classification**: EfficientNet B0
- **Disease Detection**: YOLOv8

## 📊 Usage Guide

### 🌱 Crop Recommendation

1. Go to **Dashboard**
2. Upload a soil image to detect the soil type and estimate NPK values
3. Enter environmental conditions
4. View the top three crop recommendations with model scores
5. Check market prices and profit analysis for the highest-ranked crop

### 🦠 Disease Detection

1. Go to **Disease Detection**
2. Upload crop/leaf image
3. Select model size (Nano/Small/Medium)
4. View detection results
5. Get treatment recommendations

### 📈 Data Exploration

1. Go to **Data Exploration**
2. View nutrient requirements by crop
3. Analyze climate patterns
4. Check feature correlations
5. Export insights

### 👤 User Profile

1. View account information
2. Access recommendation history
3. Export user data
4. Manage preferences

## 🎓 Training Your Own Models

### Crop Recommendation Model

```bash
python train_model.py
```

This uses: `Crop_recommendation_cleaned.csv`

The training script prints stratified top-1 and top-3 validation scores. The
app requires a soil image before showing recommendations and never falls back
to a default-NPK crop prediction.

### Soil Classification Model

```bash
python train_soil_model.py
```

This uses images in: `soil data pics/`

### Disease Detection Model

See: [DISEASE_DETECTION_GUIDE.md](DISEASE_DETECTION_GUIDE.md)

## 🌍 Supported Crops

Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

## 🦠 Detectable Diseases

- Powdery Mildew
- Early Blight
- Late Blight
- Leaf Spot
- Rust
- Bacterial Wilt
- Anthracnose
- (Custom diseases with trained models)

## 🎨 Theme & Customization

The app uses an agriculture-themed color scheme:
- **Primary Green**: `#2d5016`
- **Secondary Green**: `#6ba547`
- **Accent Brown**: `#f4a460`

Customize in `app.py` → `COLORS` dictionary

## 📱 Performance Tips

| Feature | Speed | RAM | GPU |
|---------|-------|-----|-----|
| Crop Recommendation | ⚡ Fast | 500MB | No |
| Soil Detection | 🚀 Fast | 1GB | Optional |
| Disease Detection (Nano) | 🚀 Fast | 2GB | Optional |
| Disease Detection (Medium) | 🎯 Medium | 3GB | Recommended |

## 🔐 Security

- Passwords hashed with SHA-256
- User data stored locally
- No data sent to external servers
- Secure session management

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8503
```

### YOLO Model Not Loading
```bash
pip install --upgrade ultralytics
```

### Out of Memory (YOLO)
- Use Nano model instead of Medium
- Reduce image size
- Run on CPU: `model(image, device='cpu')`

### Import Errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt
```

## 📚 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **YOLOv8 Docs**: https://docs.ultralytics.com
- **PyTorch Docs**: https://pytorch.org/docs
- **Scikit-learn Docs**: https://scikit-learn.org

## 🤝 Contributing

To improve the system:

1. Add more crop diseases to `disease_detection.py`
2. Train better models with your local data
3. Add region-specific recommendations
4. Integrate weather APIs for better predictions

## 📄 License

This project is for educational and agricultural development purposes.

## 👨‍🌾 Support

For issues or suggestions:
1. Check the troubleshooting section
2. Review the guide documents
3. Check the documentation in code comments

## 🌟 Future Roadmap

- [ ] Mobile app (React Native)
- [ ] Real-time weather integration
- [ ] Pest detection alongside disease
- [ ] Irrigation schedule recommendations
- [ ] Fertilizer optimization
- [ ] Multi-language support
- [ ] Offline mode
- [ ] Cloud sync for user data
- [ ] Drone image integration
- [ ] Yield prediction

## 🎉 Thank You!

Thank you for using KisanYatra. Happy farming! 🌾🚜

---

**Made with ❤️ for Indian Farmers**

Last Updated: August 2026
Version: 1.0.0
