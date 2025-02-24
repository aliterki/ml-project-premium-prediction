import streamlit as st
from prediction_helper import predict

st.title('Health Insurance Prediction App')

# Creating 3 columns for structured input
col1, col2, col3 = st.columns(3)

# Numerical Inputs
with col1:
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, max_value=10, step=1)
    income_lakhs = st.number_input('Income (Lakhs)', min_value=1, max_value=100, step=1)
    genetical_risk = st.number_input('Genetical Risk (%)', min_value=0, max_value=100, step=1)

# Categorical Inputs
with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    region = st.selectbox('Region', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
    marital_status = st.selectbox('Marital Status', ['Unmarried', 'Married'])
    bmi_category = st.selectbox('BMI Category', ['Overweight', 'Underweight', 'Normal', 'Obesity'])


with col3:
    employment_status = st.selectbox('Employment Status', ['Self-Employed', 'Freelancer', 'Salaried'])
    smoking_status = st.selectbox('Smoking Status', ['Regular', 'No Smoking', 'Occasional'])

    medical_history = st.selectbox('Medical History', [
        'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
        'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
        'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
    ])
    insurance_plan = st.selectbox('Insurance Plan', ['Silver', 'Bronze', 'Gold'])

input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# Predict Button
if st.button('Predict'):

    prediction = predict(input_dict)
    st.success(f"Predicted premium: {prediction}")
