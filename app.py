import streamlit as st
import numpy as np
import xgboost as xgb

# Load the XGBoost fraud detection model (classification model)
fraud_model = xgb.Booster()
fraud_model.load_model('xgboost_fraud_detection.model')

# Load the XGBRegressor for feature estimation (regression model)
regression_model = xgb.XGBRegressor()
regression_model.load_model('xgbr_model.json')

# Function to standardize feature values using mean and variance
def standardize_value(value, mean, variance):
    std_deviation = np.sqrt(variance)
    standardized_value = (value - mean) / std_deviation
    return standardized_value

# Custom scaling function (instead of MinMaxScaler)
def scale_feature_custom(value, original_min, original_max, target_min, target_max):
    scaled_value = ((value - original_min) / (original_max - original_min)) * (target_min - target_max) + target_max
    return scaled_value

# Function to convert user inputs (real-world values) to model input range
def convert_to_model_input(phone_no, card_no, cvv, expiry_date, transaction_id, amount):
    # Convert phone number, card number, etc. into numerical values
    phone_value = (int(str(phone_no)[-4:]) % 10000) / 10000 * 60 - 30
    card_value = (sum([int(digit) for digit in str(card_no)]) % 100) / 100 * 60 - 30
    cvv_value = (cvv % 1000) / 1000 * 60 - 30
    year = int(expiry_date[-4:])
    month = int(expiry_date[:2])
    expiry_value = ((year - 2024) * 12 + month) / (12 * 10) * 60 - 30  # Example conversion
    transaction_value = (hash(transaction_id) % 100000) / 100000 * 60 - 30
    
    # Standardize the values using the means and variances you provided
    phone_value_standardized = standardize_value(phone_value, 2.594614757328662e-17, 1.0000035111619878)
    card_value_standardized = standardize_value(card_value, -5.189229514657324e-18, 1.0000035111619712)
    cvv_value_standardized = standardize_value(cvv_value, -3.1933720090198915e-18, 1.000003511161991)
    expiry_value_standardized = standardize_value(expiry_value, 1.676520304735443e-17, 1.000003511161973)
    transaction_value_standardized = standardize_value(transaction_value, -1.2524005847874887e-17, 1.0000035111619898)

    # Now scale the standardized values to the required ranges using custom scaling
    phone_value_scaled = scale_feature_custom(phone_value_standardized, -1, 1, -20, 10)
    card_value_scaled = scale_feature_custom(card_value_standardized, -1, 1, -4, 11)
    cvv_value_scaled = scale_feature_custom(cvv_value_standardized, -1, 1, -61, 16)
    expiry_value_scaled = scale_feature_custom(expiry_value_standardized, -1, 1, -18, 7)
    transaction_value_scaled = scale_feature_custom(transaction_value_standardized, -1, 1, -5, 7)

    # Return the required 6 features
    return phone_value_scaled, card_value_scaled, cvv_value_scaled, expiry_value_scaled, transaction_value_scaled, amount

# Function to estimate the remaining features using the regression model
def estimate_remaining_features(v4, v8, v12, v13, v14, amount):
    # Create an array from known inputs
    input_features = np.array([v4, v8, v12, v13, v14, amount]).reshape(1, -1)
    
    # Use the regression model to predict the remaining features
    predicted_features = regression_model.predict(input_features)
    
    return predicted_features.flatten()  # Flatten to get 1D array of predicted features

# Function to prepare input for fraud prediction model
def prepare_input(phone_no, card_no, cvv, expiry_date, transaction_id, amount):
    # Convert user inputs into model-compatible format (6 features now)
    v14, v4, v8, v12, v13, amount = convert_to_model_input(phone_no, card_no, cvv, expiry_date, transaction_id, amount)
    
    # Estimate the remaining features using the regression model
    estimated_features = estimate_remaining_features(v4, v8, v12, v13, v14, amount)
    
    # Combine known inputs with the estimated features
    full_input = np.hstack((v4, v8, v12, v13, v14, amount, estimated_features))
    
    # Drop extra features to keep the total number of features as 25
    final_input = full_input[:25]
    
    # Ensure the final input array has 25 features
    if final_input.shape[0] != 25:
        raise ValueError(f"Expected 25 features but got {final_input.shape[0]}. Please check your input transformation.")
    
    return final_input

# Streamlit app UI
st.title("Credit Card Fraud Detection")

st.write("Please input the following transaction details:")

# User inputs
phone_no = st.text_input("Phone Number (e.g., 9876543210):", max_chars=10)
card_no = st.text_input("Card Number (e.g., 1234567812345678):", max_chars=16)
cvv = st.number_input("CVV (3-digit code):", min_value=100, max_value=999)
expiry_date = st.text_input("Expiry Date (mmyyyy):", max_chars=6)
transaction_id = st.text_input("Transaction ID (alphanumeric):", max_chars=16)
amount = st.number_input("Transaction Amount:", min_value=0.0, max_value=10000.0, step=0.1)

# Button to predict fraud
if st.button("Predict Fraud"):
    # Check if all inputs are provided
    if phone_no and card_no and cvv and expiry_date and transaction_id:
        # Prepare the input data for the model
        model_input = prepare_input(phone_no, card_no, cvv, expiry_date, transaction_id, amount)
        
        # Make prediction using the fraud detection model
        dmatrix = xgb.DMatrix(model_input.reshape(1, -1))  # Ensure input is 2D
        prediction = fraud_model.predict(dmatrix)
        prediction = 10000 * prediction

        # Output the result
        if prediction >= 0.5:  # Assuming a threshold for binary classification
            st.error("This transaction is predicted to be fraudulent.")
            
        else:
            st.success("This transaction is predicted to be legitimate.")
            
    else:
        st.warning("Please fill in all the required fields.")
