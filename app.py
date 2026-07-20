import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model/credit_default_prediction_model.pkl")

model = load_model()

st.title("💳 Credit Card Default Prediction")

st.markdown(
    """
Predict whether a customer is likely to default on their
credit card payment next month using Machine Learning.
"""
)

st.header("Customer Information")
limit_bal = st.number_input(
    "Credit Limit",
    min_value=10000,
    max_value=1000000,
    value=200000,
    step=10000
)
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

sex = 1 if gender == "Male" else 2
education_options = {
    "Graduate School":1,
    "University":2,
    "High School":3,
    "Others":4,
    "Unknown":0
}

education = st.selectbox(
    "Education",
    list(education_options.keys())
)

education_value = education_options[education]

marriage_options = {
    "Married":1,
    "Single":2,
    "Others":3
}

marriage = st.selectbox(
    "Marital Status",
    list(marriage_options.keys())
)

marriage_value = marriage_options[marriage]

age = st.slider(
    "Age",
    21,
    80,
    35
)

payment_status = {
    "Paid Duly":-1,
    "No Consumption":-2,
    "Revolving Credit":0,
    "1 Month Delay":1,
    "2 Months Delay":2,
    "3 Months Delay":3,
    "4 Months Delay":4,
    "5 Months Delay":5,
    "6 Months Delay":6,
    "7 Months Delay":7,
    "8 Months Delay":8
}

st.header("Bill Amounts")
col1,col2,col3 = st.columns(3)

st.header("Previous Payments")
col1,col2,col3 = st.columns(3)

if st.button("Predict Default Risk"):

    input_data = pd.DataFrame({
    ...
})
    
    prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error("High Risk of Default")
else:
    st.success("Low Risk of Default")

    st.metric(
    "Probability of Default",
    f"{probability:.2%}"
)
    
if probability < 0.30:
    risk = "Low"

elif probability < 0.60:
    risk = "Medium"

else:
    risk = "High"

st.info(f"Risk Level : {risk}")

st.markdown("---")

st.caption(
    "Developed by Abdul Munim | Machine Learning Project"
)