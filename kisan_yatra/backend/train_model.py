# This is a separate script to run once. Let's call it train_model.py
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# Load data
df = pd.read_csv("Crop_recommendation_cleaned.csv")
X = df.drop('label', axis=1)
y = df['label']

# Keep a stratified holdout report so a bad retrain is visible before deployment.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print(f"Validation top-1 accuracy: {accuracy_score(y_test, predictions):.3f}")
print(
    "Validation top-3 accuracy: "
    f"{top_k_accuracy_score(y_test, probabilities, k=3, labels=model.classes_):.3f}"
)

# Retrain on all verified rows before saving the production artifact.
model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True) # Ensure 'models' directory exists
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model 'crop_model.pkl' has been trained and saved successfully in the 'models' folder.")