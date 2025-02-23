import pandas as pd
from joblib import load



model_rest = load("artifacts\model_rest.joblib")
model_young = load("artifacts\model_young.joblib")
scaler_rest = load("artifacts\scaler_rest.joblib")
scaler_young = load("artifacts\scaler_young.joblib")



def calculate_normalized_risk(medical_history):
    # Risk scores for individual diseases
    risk_score = {
        'Diabetes': 6,
        'High blood pressure': 6,
        'Heart disease': 8,
        'Thyroid': 5,
        'No Disease': 0,
        'none': 0
    }

    # Split diseases if combined
    diseases = medical_history.lower().split(" & ")  # Splitting combined diseases
    # Calculate total risk score
    total_risk = sum(risk_score[disease] for disease in diseases if disease in risk_score)

    # Find max possible risk score (worst-case scenario)
    max_possible_risk = 14

    # Normalize the risk score (between 0 and 1)
    normalized_risk = total_risk / max_possible_risk if max_possible_risk > 0 else 0

    return normalized_risk

def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    scaler = scaler_object['scaler']
    cols_to_scale = scaler_object['cols_to_scale']
    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)
    return df

def preprocessing_input(input_dict):
    expected_columns= ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'genetical_risk', 'normalized_risk_score', 'gender_Male',
       'region_Northwest', 'region_Southeast', 'region_Southwest',
       'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed']

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Initialize dataframe with zeros
    df = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Assign numerical values
    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['genetical_risk'] = input_dict['Genetical Risk']

    # Encoding categorical variables
    if input_dict['Gender'] == 'Male':
        df['gender_Male'] = 1

    if input_dict['Region'] == 'Northwest':
        df['region_Northwest'] = 1
    elif input_dict['Region'] == 'Southeast':
        df['region_Southeast'] = 1
    elif input_dict['Region'] == 'Southwest':
        df['region_Southwest'] = 1

    if input_dict['Marital Status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1

    if input_dict['BMI Category'] == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif input_dict['BMI Category'] == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif input_dict['BMI Category'] == 'Underweight':
        df['bmi_category_Underweight'] = 1

    if input_dict['Smoking Status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif input_dict['Smoking Status'] == 'Regular':
        df['smoking_status_Regular'] = 1

    if input_dict['Employment Status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif input_dict['Employment Status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    df = handle_scaling(input_dict['Age'], df)
    return df



def predict(input_dict):
    input_df = preprocessing_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction)