import streamlit as st
import pandas as pd
import joblib

st.title("🚌 Transport Demand Prediction Dashboard")

model = joblib.load("best_model.pkl")
st.success("✅ Model loaded successfully!")

uploaded = st.file_uploader("📤 Upload a CSV file for prediction", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("📄 Uploaded Data (first 5 rows):", df.head())
    preds = model.predict(df)
    df["Predicted_Seats"] = preds
    st.write("✅ Predictions (sample):", df.head())
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv", "text/csv")