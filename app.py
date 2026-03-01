import streamlit as st
import pandas as pd
import pickle
import os


# Load model and preprocessor
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)


st.set_page_config(page_title="Sales Demand Forecast")

st.title("📊 Sales Demand Forecast App")

st.write("Enter feature values to predict future Sales")


lag_1 = st.number_input("Lag 1 (Previous Day Sales)", value=0.0)
lag_7 = st.number_input("Lag 7 (Last Week Same Day Sales)", value=0.0)
rolling_mean = st.number_input("Rolling Mean (7 Day Avg)", value=0.0)

year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
dayofweek = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)


if st.button("Predict Sales"):

    input_df = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean": rolling_mean,
        "year": year,
        "month": month,
        "day": day,
        "dayofweek": dayofweek
    }])

    scaled_data = preprocessor.transform(input_df)
    prediction = model.predict(scaled_data)

    st.success(f"Predicted Sales: {round(prediction[0], 2)}")