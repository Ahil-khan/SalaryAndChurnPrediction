import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

def main():
    page = st.sidebar.selectbox("Choose a page", ["Home", "Predict Salary", "Predict Churn"])

    if page == "Home":
        st.title("Welcome")
        st.write("This app is designed to predict salary and customer churn. The salary prediction model uses a neural network to predict salary based on experience, education, position, industry, and location. The customer churn prediction model uses a neural network to predict the likelihood of a customer churning based on their demographic and account information.")
    elif page == "Predict Salary":
        salary_prediction_page()
    elif page == "Predict Churn":
        churn_prediction_page()

def salary_prediction_page():
    try:
        # Load the trained model
        model = tf.keras.models.load_model('salary_model.keras')
        # Load the scaler
        with open('scaler_salary_model.pkl', 'rb') as file:
            scaler = pickle.load(file)
        # Load the encoders
        with open('onehot_encoder_education_salary_model.pkl', 'rb') as file:
            onehot_encoder_education = pickle.load(file)
        with open('onehot_encoder_position_salary_model.pkl', 'rb') as file:
            onehot_encoder_position = pickle.load(file)
        with open('onehot_encoder_industry_salary_model.pkl', 'rb') as file:
            onehot_encoder_industry = pickle.load(file)
        with open('onehot_encoder_location_salary_model.pkl', 'rb') as file:
            onehot_encoder_location = pickle.load(file)

        st.title("Salary Prediction")
        experience = st.number_input('Experience (years)')
        education = st.selectbox('Education', onehot_encoder_education.categories_[0])
        position = st.selectbox('Position', onehot_encoder_position.categories_[0])
        industry = st.selectbox('Industry', onehot_encoder_industry.categories_[0])
        location = st.selectbox('Location', onehot_encoder_location.categories_[0])

        # Prepare the input data
        education_encoded = onehot_encoder_education.transform([[education]]).toarray()
        position_encoded = onehot_encoder_position.transform([[position]]).toarray()
        industry_encoded = onehot_encoder_industry.transform([[industry]]).toarray()
        location_encoded = onehot_encoder_location.transform([[location]]).toarray()

        input_data = pd.DataFrame({
            'YearsExperience': [experience]
        })
        education_encoded_df = pd.DataFrame(education_encoded, columns=onehot_encoder_education.get_feature_names_out(['EducationLevel']))
        position_encoded_df = pd.DataFrame(position_encoded, columns=onehot_encoder_position.get_feature_names_out(['Position']))
        industry_encoded_df = pd.DataFrame(industry_encoded, columns=onehot_encoder_industry.get_feature_names_out(['Industry']))
        location_encoded_df = pd.DataFrame(location_encoded, columns=onehot_encoder_location.get_feature_names_out(['Location']))

        input_data = pd.concat([input_data.reset_index(drop=True), education_encoded_df, position_encoded_df, industry_encoded_df, location_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        if st.button('Predict'):
            prediction = model.predict(input_data_scaled)
            predicted_salary = prediction[0][0]
            st.write(f"Predicted Salary: {predicted_salary:.2f}")
    except Exception as e:
        st.write("An error occurred:")
        st.write(e)

def churn_prediction_page():
    try:
        # Load the trained model
        model = tf.keras.models.load_model('churn_model.keras')
        # Load the encoder and scaler
        with open('label_encoder_gender_churn_model.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open('onehot_encoder_geo_churn_model.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        with open('scaler_churn_model.pkl', 'rb') as file:
            scaler = pickle.load(file)

        st.title("Customer Churn Prediction")
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 93)
        balance = st.number_input('Balance')
        credit_score = st.number_input('Credit Score')
        estimated_salary = st.number_input('Estimated Salary')
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider("Number of Products", 1, 4)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Is Active Member', [0, 1])

        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        if st.button('Predict'):
            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]
            st.write(f"Churn Probability: {prediction_proba:.2f}")
            if prediction_proba > 0.5:
                st.write("The Customer is likely to churn")
            else:
                st.write('The customer is not likely to leave the bank')
    except Exception as e:
        st.write("An error occurred:")
        st.write(e)

if __name__ == "__main__": main()