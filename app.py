import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("knn_model.pkl")

# Title
st.title("Iris Flower Prediction")

# Inputs
sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

# Prediction
if st.button("Predict"):
    data = np.array([[sl, sw, pl, pw]])
    
    # Get prediction
    prediction = model.predict(data)[0]
    
    # Show result
    st.success(f"Prediction: {prediction}")
