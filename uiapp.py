import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model (save it first if you haven't)
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # If you scaled your data

st.title("🏠 House Price Predictor")

# Sidebar input fields
st.sidebar.header("Enter House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (GrLivArea)", value=1500)
garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 2)
total_bsmt_sf = st.sidebar.number_input("Total Basement SF", value=800)
first_flr_sf = st.sidebar.number_input("1st Floor SF", value=1000)
year_built = st.sidebar.number_input("Year Built", value=2000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 3, 2)
tot_rooms = st.sidebar.slider("Total Rooms Above Ground", 1, 12, 6)

# Create input dataframe
input_df = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars],
    'TotalBsmtSF': [total_bsmt_sf],
    '1stFlrSF': [first_flr_sf],
    'YearBuilt': [year_built],
    'FullBath': [full_bath],
    'TotRmsAbvGrd': [tot_rooms]
})

# Make prediction
if st.button("Predict House Price"):
    # scale if necessary
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]  # or input_scaled if used
    st.success(f"🏡 Predicted House Price: **${prediction:,.2f}**")

