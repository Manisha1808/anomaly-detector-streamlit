# train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === 1️⃣ Load your training dataset ===
df = pd.read_csv("cleaned_kdd.csv")  # replace with your CLEAN NUMERIC dataset

# === 2️⃣ Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 3️⃣ Train Isolation Forest ===
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# === 4️⃣ Create models directory if not exists ===
os.makedirs("models", exist_ok=True)

# === 5️⃣ Save the model and scaler ===
joblib.dump(model, "models/isolation_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully.")
