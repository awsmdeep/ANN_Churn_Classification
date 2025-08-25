import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# =========================
# LOAD MODEL & ENCODERS
# =========================

# Load trained ANN model
model = tf.keras.models.load_model('model.h5')

# Load label encoder for Gender
with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Load OneHotEncoder for Geography
with open("One_hot_encoder.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

# Load StandardScaler (used during training)
with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)


# =========================
# STREAMLIT APP DESIGN
# =========================

# Page title
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
        📊 Customer Churn Prediction
    </h1>
    <p style='text-align: center; color: gray; font-size:18px;'>
        Enter customer details below and predict whether they are likely to churn 🔮
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)


# =========================
# USER INPUT FORM
# =========================
with st.form("input_form"):

    st.subheader("📝 Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, step=1)

    with col2:
        tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10)
        num_of_products = st.slider('🛒 Number of Products', 1, 4)
        has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
        is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

    st.subheader("💰 Financial Information")
    balance = st.number_input('🏦 Balance', min_value=0.0, step=100.0)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, step=100.0)

    # Submit button
    submitted = st.form_submit_button("🔮 Predict Churn")


# =========================
# DATA PREPROCESSING
# =========================
if submitted:
    # Encode Gender into numerical (0/1)
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # One-hot encode Geography (convert categorical into multiple binary columns)
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Create input DataFrame with user values
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Merge geography encoding
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale numeric data (same scaling as training)
    input_data_scaled = scalar.transform(input_data)


    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # =========================
    # DISPLAY RESULT
    # =========================
    st.markdown("---")
    st.subheader("📌 Prediction Result:")

    if prediction_proba > 0.5:
        st.markdown(
            f"""
            <div style='padding:20px; background-color:#FFCCCC; border-radius:10px; text-align:center'>
                <h2 style='color:#FF0000;'>⚠️ The customer is LIKELY to churn</h2>
                <p style='font-size:18px;'>Probability: <b>{prediction_proba:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='padding:20px; background-color:#CCFFCC; border-radius:10px; text-align:center'>
                <h2 style='color:#008000;'>✅ The customer is NOT likely to churn</h2>
                <p style='font-size:18px;'>Probability: <b>{prediction_proba:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
