import sys
import os
import streamlit as st


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict_price

st.title("🚗 Used Car Price Prediction")

st.write("Enter car details below to estimate resale price.")

year = st.number_input("Manufacturing Year", 2000, 2024)
km_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):
    input_data = {
        "year": year,
        "km_driven": km_driven,
        "owner": owner,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission
    }

    price = predict_price(input_data)

    st.success(f"Estimated Selling Price: ₹ {price:,.2f}")