import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Load the trained model and preprocessing objects
model=tf.keras.models.load_model("model.h5")
# load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)
with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")
# Create input fields for user data
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary= st.number_input("Estimated Salary")
tenure=st.number_input("Tenure", min_value=0, max_value=10)
num_of_products=st.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is active Member",[0,1])

# prepare the input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]  
})
# one-hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# concat the encoded geo with the input data
input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

# predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]
st.write(f'Churn Probability: {prediction_prob:.2f}')
if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
