import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('models/isolation_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(layout="wide", page_title="Anomaly Detector")
st.title("🚨 Anomaly Detector with Isolation Forest")

uploaded_file = st.file_uploader("📤 Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    try:
        X_scaled = scaler.transform(df)
        preds = model.predict(X_scaled)
        df['Anomaly'] = preds == -1

        st.success(f"✅ Detected {df['Anomaly'].sum()} anomalies out of {len(df)} rows")

        st.subheader("📉 Anomaly Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        clean_df = df.replace({True: 'Anomaly', False: 'Normal'})
        sns.countplot(
    x='Anomaly',
    hue='Anomaly',
    data=clean_df,
    palette='viridis',
    legend=False,
    ax=ax
)
        ax.set_title("Anomaly vs Normal Count")
        st.pyplot(fig)

        st.subheader("📌 Detection Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results CSV", data=csv, file_name="anomaly_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
