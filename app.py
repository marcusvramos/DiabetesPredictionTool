import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load
import pandas as pd

model = load_model('diabetes_mlp.h5')
imputer = load('imputer.joblib')
scaler = load('scaler.joblib')

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for col in columns_to_impute:
        input_df[col] = input_df[col].replace(0, np.nan)
    
    input_imp = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imp)
    prediction = model.predict(input_scaled)
    return int(prediction > 0.5)

st.title('Previsão de Diabetes')
pregnancies = st.number_input('Número de Gravidezes', min_value=0, step=1)
glucose = st.number_input('Nível de Glucose', min_value=0.0)
blood_pressure = st.number_input('Pressão Arterial', min_value=0.0)
skin_thickness = st.number_input('Espessura da Pele', min_value=0.0)
insulin = st.number_input('Insulina', min_value=0.0)
bmi = st.number_input('IMC', min_value=0.0)
dpf = st.number_input('Função de Pedigree de Diabetes', min_value=0.0)
age = st.number_input('Idade', min_value=0)

if st.button('Prever'):
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    if result == 1:
        st.write('O modelo prevê que o paciente tem diabetes.')
    else:
        st.write('O modelo prevê que o paciente não tem diabetes.')