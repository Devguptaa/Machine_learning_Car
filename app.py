import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

# Load the trained model
model = joblib.load('car_price_predictor')

# Function to preprocess input data and predict car price
def predict_price(p1, p2, p3, p4, p5, p6, p7):
    # One-hot encode categorical variables
    fuel_type = 1 if p3 == 'Petrol' else (2 if p3 == 'Diesel' else 3)
    seller_type = 1 if p4 == 'Dealer' else 2
    transmission = 1 if p5 == 'Manual' else 2
    
    # Create a DataFrame with the input values
    data_new = pd.DataFrame({
        'Present_Price': [p1],
        'Kms_Driven': [p2],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [p6],
        'Age': [p7]
        
    })

    # Predict the price
    result = model.predict(data_new)

    return result[0]

# Streamlit app
def main():
    st.title("Car Price Prediction Using Machine Learning")

    # Input fields for user
    st.sidebar.title("Input Parameters")
    p1 = st.sidebar.number_input("Present_Price")
    p2 = st.sidebar.number_input("Kms_Driven", value=0, step=1, format="%d")
    p3 = st.sidebar.selectbox("Fuel_Type", ['Petrol', 'Diesel', 'CNG'])
    p4 = st.sidebar.selectbox("Seller_Type", ['Dealer', 'Individual'])
    p5 = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
    p6 = st.sidebar.number_input("Owner", value=0, step=1, format="%d")
    p7 = st.sidebar.number_input("Age")

    # Button to predict car price
    if st.sidebar.button("Predict"):
        result = predict_price(p1, p2, p3, p4, p5, p6, p7)
        st.success(f"Car Purchase amount: {result}")

if __name__ == "__main__":
    main()
