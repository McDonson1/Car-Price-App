import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and scaler
with open("Car_Modelling_randomforestregressor_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Define options for user input
Brand = [
    'volkswagen', 'audi', 'jeep', 'skoda', 'bmw', 'peugeot', 'ford', 'mazda', 'nissan', 'renault', 
    'mercedes_benz', 'opel', 'seat', 'citroen', 'honda', 'fiat', 'mini', 'smart', 'hyundai', 'sonstige_autos', 
    'alfa_romeo', 'subaru', 'volvo', 'mitsubishi', 'kia', 'suzuki', 'lancia', 'porsche', 'toyota', 
    'chevrolet', 'dacia', 'daihatsu', 'trabant', 'saab', 'chrysler', 'jaguar', 'daewoo', 'rover', 'land_rover', 'lada'
]

Abtest = ["test", "control"]
Gearbox = ["manuell", "automatik"]
FuelType = ["benzin", "diesel", "lpg", "cng", "hybrid", "elektro", "andere"]
VehicleType = ["limousine", "kleinwagen", "kombi", "bus", "cabrio", "coupe", "suv", "andere"]
Model = ['golf', 'grand', 'fabia', '3er', '2_reihe', 'andere', 'c_max', '3_reihe', 'passat', 'navara', 
         'ka', 'polo', 'twingo', 'a_klasse', 'scirocco', '5er', 'meriva', 'arosa', 'c4', 'civic']

# Streamlit UI
st.title("Car Price Predictor App")

# User input
Brand = st.selectbox("Select Brand", Brand)
Abtest = st.selectbox("Select Abtest", Abtest) 
Gearbox = st.selectbox("Select Gearbox", Gearbox) 
FuelType = st.selectbox("Select FuelType", FuelType) 
VehicleType = st.selectbox("Select VehicleType", VehicleType) 
Model = st.selectbox("Select Model", Model) 
DateCreated = st.slider("DateCreated", 2000, 2025)
Kilometer = st.number_input("Kilometer", min_value=80)
PowerPS = st.number_input("PowerPS", min_value=1)

# Input data for prediction
input_data = pd.DataFrame({
    'Brand': [Brand],
    'Abtest': [Abtest],
    'VehicleType': [VehicleType],
    'Gearbox': [Gearbox],
    'FuelType': [FuelType],
    'Model': [Model],
    'DateCreated': [DateCreated],
    'Kilometer': [Kilometer],
    'PowerPS': [PowerPS],
})

# Prediction
if st.button("Predict Car Price"):
    prediction = rf_model.predict(input_data)
    st.success(f"Predicted Car Price: ${prediction[0]:.2f}")
