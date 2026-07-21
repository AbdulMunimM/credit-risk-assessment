import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

div[data-testid="stMetric"]{
    background-color:#f8f9fa;
    border-radius:10px;
    padding:15px;
    border:1px solid #e6e6e6;
}

div.stButton > button{
    height:55px;
    font-size:18px;
    font-weight:bold;
    border-radius:12px;
}

div[data-testid="stExpander"]{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.title("💳 Credit Risk Predictor")

    st.markdown("---")

    st.markdown("""
### 📌 Project

This application predicts whether a customer is likely to **default on their credit card payment next month**.

### 🤖 Model

- Random Forest Classifier
- Optimized using GridSearchCV

### 📂 Dataset

UCI Credit Card Default Dataset

### 👨‍💻 Developer

Abdul Munim

---
""")

    st.info(
        "The prediction is based on customer demographics, repayment history, bill statements, and previous payments."
    )

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.title("💳 Credit Card Default Prediction")

st.markdown("""
Predict the likelihood that a customer will default on their **next month's credit card payment** using a machine learning model.

The model has been trained on the **UCI Credit Card Default Dataset** and optimized using **GridSearchCV**.

---
""")

# ====================================================
# CUSTOMER INFORMATION
# ====================================================

with st.expander("👤 Customer Information", expanded=True):

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
# REPAYMENT HISTORY
# ====================================================

with st.expander("📈 Repayment History", expanded=True):

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
            st.selectbox(
                "September Repayment Status",
                payment_status.keys()
            )
        ]

        pay2 = payment_status[
            st.selectbox(
                "August Repayment Status",
                payment_status.keys()
            )
        ]

    with col2:
        pay3 = payment_status[
            st.selectbox(
                "July Repayment Status",
                payment_status.keys()
            )
        ]

        pay4 = payment_status[
            st.selectbox(
                "June Repayment Status",
                payment_status.keys()
            )
        ]

    with col3:
        pay5 = payment_status[
            st.selectbox(
                "May Repayment Status",
                payment_status.keys()
            )
        ]

        pay6 = payment_status[
            st.selectbox(
                "April Repayment Status",
                payment_status.keys()
            )
        ]


# ====================================================
# BILL AMOUNTS
# ====================================================

with st.expander("💰 Bill Amounts", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        bill1 = st.number_input(
            "Bill Amount - September",
            min_value=0,
            value=0
        )

        bill2 = st.number_input(
            "Bill Amount - August",
            min_value=0,
            value=0
        )

    with col2:
        bill3 = st.number_input(
            "Bill Amount - July",
            min_value=0,
            value=0
        )

        bill4 = st.number_input(
            "Bill Amount - June",
            min_value=0,
            value=0
        )

    with col3:
        bill5 = st.number_input(
            "Bill Amount - May",
            min_value=0,
            value=0
        )

        bill6 = st.number_input(
            "Bill Amount - April",
            min_value=0,
            value=0
        )


# ====================================================
# PREVIOUS PAYMENTS
# ====================================================

with st.expander("💵 Previous Payments", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        pay_amt1 = st.number_input(
            "Payment - September",
            min_value=0,
            value=0
        )

        pay_amt2 = st.number_input(
            "Payment - August",
            min_value=0,
            value=0
        )

    with col2:
        pay_amt3 = st.number_input(
            "Payment - July",
            min_value=0,
            value=0
        )

        pay_amt4 = st.number_input(
            "Payment - June",
            min_value=0,
            value=0
        )

    with col3:
        pay_amt5 = st.number_input(
            "Payment - May",
            min_value=0,
            value=0
        )

        pay_amt6 = st.number_input(
            "Payment - April",
            min_value=0,
            value=0
        )


# ====================================================
# CUSTOMER SUMMARY
# ====================================================

st.markdown("---")

st.subheader("📋 Customer Summary")

summary = pd.DataFrame(
    {
        "Feature": [
            "Credit Limit",
            "Gender",
            "Age",
            "Education",
            "Marital Status"
        ],
        "Value": [
            f"${limit_bal:,.0f}",
            gender,
            age,
            education,
            marriage
        ]
    }
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Review the customer information above before generating the prediction."
)

st.markdown("")


# ====================================================
# PREDICT BUTTON
# ====================================================

predict = st.button(
    "🚀 Predict Default Risk",
    type="primary",
    use_container_width=True
)

if predict:

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

    with st.spinner("Analyzing customer financial profile..."):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    # ====================================================
    # Prediction Results
    # ====================================================

    st.markdown("---")
    st.header("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Credit Card Default")
    else:
        st.success("✅ Low Risk of Credit Card Default")

    # -------------------------------
    # Metrics
    # -------------------------------

    if probability < 0.30:
        risk = "🟢 Low"
    elif probability < 0.60:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probability of Default",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    # -------------------------------
    # Probability Progress
    # -------------------------------

    st.write("### Default Probability")

    st.progress(float(probability))

    

    # -------------------------------
    # Interpretation
    # -------------------------------

    st.write("### Interpretation")

    if probability < 0.30:

        st.success("""
This customer has a **low probability** of defaulting next month.

The repayment history and financial information indicate relatively healthy credit behavior.
""")

    elif probability < 0.60:

        st.warning("""
This customer has a **moderate probability** of default.

Additional financial review or monitoring may be appropriate before making lending decisions.
""")

    else:

        st.error("""
This customer has a **high probability** of defaulting next month.

Financial institutions should carefully evaluate the customer's repayment history before extending additional credit.
""")

    # -------------------------------
    # Recommendation
    # -------------------------------

    st.write("### Recommendation")

    if probability < 0.30:

        st.info("""
✔ Low credit risk

• Eligible for standard review

• Continue regular monitoring
""")

    elif probability < 0.60:

        st.info("""
✔ Medium credit risk

• Perform additional verification

• Consider lower credit exposure

• Review recent payment behavior
""")

    else:

        st.info("""
✔ High credit risk

• Manual review recommended

• Consider reducing credit exposure

• Verify repayment capability

• Request additional financial information
""")

# ====================================================
# Footer
# ====================================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.caption(
        "Developed by Abdul Munim"
    )

with col2:

    st.caption(
        "Machine Learning • Streamlit • Scikit-learn"
    )

st.caption(
    """
This application is intended for educational and demonstration purposes.
Predictions should support—not replace—professional financial decision-making.
"""
)