import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/credit_default_prediction_model.pkl")

model = load_model()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("About")

st.sidebar.info(
    """
    **Credit Card Default Prediction**

    **Model:** Random Forest (GridSearchCV)

    **Dataset:** UCI Credit Card Default Dataset

    **Developer:** Abdul Munim
    """
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title("💳 Credit Card Default Prediction")

st.write(
    "Predict whether a customer is likely to default on their credit card payment next month."
)

# ====================================================
# Customer Information
# ====================================================
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
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

    age = st.slider(
        "Age",
        min_value=21,
        max_value=80,
        value=35
    )

with col2:

    education_options = {
        "Graduate School": 1,
        "University": 2,
        "High School": 3,
        "Others": 4,
        "Unknown": 0
    }

    education = st.selectbox(
        "Education",
        list(education_options.keys())
    )

    education_value = education_options[education]

    marriage_options = {
        "Married": 1,
        "Single": 2,
        "Others": 3
    }

    marriage = st.selectbox(
        "Marital Status",
        list(marriage_options.keys())
    )

    marriage_value = marriage_options[marriage]

# ====================================================
# Repayment History
# ====================================================
st.header("Repayment History")

payment_status = {
    "No Consumption (-2)": -2,
    "Paid Duly (-1)": -1,
    "Revolving Credit (0)": 0,
    "1 Month Delay": 1,
    "2 Months Delay": 2,
    "3 Months Delay": 3,
    "4 Months Delay": 4,
    "5 Months Delay": 5,
    "6 Months Delay": 6,
    "7 Months Delay": 7,
    "8 Months Delay": 8
}

col1, col2, col3 = st.columns(3)

with col1:
    pay0 = payment_status[
        st.selectbox("September", payment_status.keys())
    ]

    pay2 = payment_status[
        st.selectbox("August", payment_status.keys())
    ]

with col2:
    pay3 = payment_status[
        st.selectbox("July", payment_status.keys())
    ]

    pay4 = payment_status[
        st.selectbox("June", payment_status.keys())
    ]

with col3:
    pay5 = payment_status[
        st.selectbox("May", payment_status.keys())
    ]

    pay6 = payment_status[
        st.selectbox("April", payment_status.keys())
    ]

# ====================================================
# Bill Amounts
# ====================================================
st.header("Bill Amounts")

col1, col2, col3 = st.columns(3)

with col1:
    bill1 = st.number_input("Bill Amount September", value=0)
    bill2 = st.number_input("Bill Amount August", value=0)

with col2:
    bill3 = st.number_input("Bill Amount July", value=0)
    bill4 = st.number_input("Bill Amount June", value=0)

with col3:
    bill5 = st.number_input("Bill Amount May", value=0)
    bill6 = st.number_input("Bill Amount April", value=0)

# ====================================================
# Previous Payments
# ====================================================
st.header("Previous Payments")

col1, col2, col3 = st.columns(3)

with col1:
    pay_amt1 = st.number_input("Payment September", value=0)
    pay_amt2 = st.number_input("Payment August", value=0)

with col2:
    pay_amt3 = st.number_input("Payment July", value=0)
    pay_amt4 = st.number_input("Payment June", value=0)

with col3:
    pay_amt5 = st.number_input("Payment May", value=0)
    pay_amt6 = st.number_input("Payment April", value=0)

# ====================================================
# Prediction
# ====================================================
st.markdown("---")

if st.button("Predict Default Risk", use_container_width=True):

    input_data = pd.DataFrame({
        "LIMIT_BAL": [limit_bal],
        "SEX": [sex],
        "EDUCATION": [education_value],
        "MARRIAGE": [marriage_value],
        "AGE": [age],
        "PAY_0": [pay0],
        "PAY_2": [pay2],
        "PAY_3": [pay3],
        "PAY_4": [pay4],
        "PAY_5": [pay5],
        "PAY_6": [pay6],
        "BILL_AMT1": [bill1],
        "BILL_AMT2": [bill2],
        "BILL_AMT3": [bill3],
        "BILL_AMT4": [bill4],
        "BILL_AMT5": [bill5],
        "BILL_AMT6": [bill6],
        "PAY_AMT1": [pay_amt1],
        "PAY_AMT2": [pay_amt2],
        "PAY_AMT3": [pay_amt3],
        "PAY_AMT4": [pay_amt4],
        "PAY_AMT5": [pay_amt5],
        "PAY_AMT6": [pay_amt6]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    st.metric(
        "Probability of Default",
        f"{probability:.2%}"
    )

    if probability < 0.30:
        risk = "🟢 Low"
    elif probability < 0.60:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"

    st.info(f"Risk Level: **{risk}**")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Abdul Munim | Machine Learning Project"
)