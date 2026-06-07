import streamlit as st
import pandas as pd
import pickle

# 1. Saved Files ko load karna
model = pickle.load(open('placement_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 2. Web UI design karna
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Placement Predictor App")
st.write("---")
st.write("Apna CGPA aur IQ enter karke check karein ki aapka placement hoga ya nahi.")

# Input fields
cgpa = st.number_input("Student ka CGPA dalein (5.0 - 10.0):", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
iq = st.number_input("Student ka IQ dalein (80 - 140):", min_value=80, max_value=140, value=100, step=1)

st.write("---")

# Prediction logic
if st.button("Predict Placement", use_container_width=True):
    input_data = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.success("🎉 **Prediction: HAAN!** Placement hone ki poori umeed hai. Badhai ho!")
    else:
        st.error("😞 **Prediction: NAHI!** Placement ke chances thode kam hain. Aur mehnat ki zaroorat hai.")
