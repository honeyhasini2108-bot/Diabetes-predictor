import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# App title
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below:")

# Inputs
preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Prediction
if st.button("Predict"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)

    risk_score = glucose + bmi + age + bp

    st.subheader("Result")

    if result[0] == 1:
        st.error("🛑 HIGH DIABETES RISK")
        st.write("👉 Please consult a doctor.")

    else:
        if risk_score > 250:
            st.warning("⚠️ Borderline risk detected. Maintain healthy lifestyle.")
        else:
            st.success("🟢 LOW RISK - No diabetes detected.")
