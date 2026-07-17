import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = joblib.load(
    "model/credit_risk_model.pkl"
)


# ---------------------------------------------------------
# PDF REPORT GENERATOR
# ---------------------------------------------------------

def generate_pdf(
    customer,
    risk_level,
    probability,
    confidence,
    factors
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "CreditGuard AI",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            "AI Credit Risk Assessment Report",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            "<b>Customer Profile</b>",
            styles["Heading3"]
        )
    )

    for key,value in customer.items():

        content.append(
            Paragraph(
                f"{key}: {value}",
                styles["Normal"]
            )
        )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            "<b>Prediction Results</b>",
            styles["Heading3"]
        )
    )

    content.append(
        Paragraph(
            f"Risk Level: {risk_level}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Default Probability: {probability:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Model Confidence: {confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            "<b>Key Risk Indicators</b>",
            styles["Heading3"]
        )
    )

    for item in factors:

        content.append(
            Paragraph(
                item,
                styles["Normal"]
            )
        )

    doc.build(content)

    buffer.seek(0)

    return buffer


# ---------------------------------------------------------
# RISK GAUGE
# ---------------------------------------------------------

def risk_gauge(score):

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=score,

            title={
                "text":"Credit Risk Score"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"darkblue"
                },

                "steps":[

                    {
                        "range":[0,30],
                        "color":"#22C55E"
                    },

                    {
                        "range":[30,60],
                        "color":"#FACC15"
                    },

                    {
                        "range":[60,100],
                        "color":"#EF4444"
                    }

                ]

            }

        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return fig


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.main-header{

    background:linear-gradient(
        90deg,
        #1E3A8A,
        #2563EB
    );

    padding:30px;

    border-radius:15px;

    color:white;

    text-align:center;

    margin-bottom:25px;

}

.metric-card{

    background:#F8FAFC;

    padding:25px;

    border-radius:15px;

    border:1px solid #E2E8F0;

    box-shadow:0px 4px 10px rgba(0,0,0,0.05);

}

.footer{

    text-align:center;

    color:gray;

    padding-top:30px;

    font-size:14px;

}

.sidebar-title{

    font-size:22px;

    font-weight:bold;

    color:#2563EB;

}

</style>
""",
unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("""
<div class="main-header">

<h1>🛡️ CreditGuard AI</h1>

<h4>
AI-Powered Credit Risk Assessment System
</h4>

<p>
Predict customer credit default risk using Machine Learning
</p>

</div>
""",
unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR TITLE
# ---------------------------------------------------------

st.sidebar.markdown(
    '<p class="sidebar-title">Customer Information</p>',
    unsafe_allow_html=True
)
# ---------------------------------------------------------
# PERSONAL INFORMATION
# ---------------------------------------------------------

with st.sidebar.expander(
    "👤 Personal Information",
    expanded=True
):

    gender_options = {
        "Male": 1,
        "Female": 2
    }

    gender = st.selectbox(
        "Gender",
        list(gender_options.keys())
    )

    sex = gender_options[gender]


    education_options = {

        "Unknown": 0,
        "High School": 1,
        "College": 2,
        "University": 3,
        "Diploma": 4,
        "PhD": 5,
        "Other": 6

    }

    education = st.selectbox(
        "Education Level",
        list(education_options.keys())
    )

    education_value = education_options[education]


    marriage_options = {

        "Unknown": 0,
        "Married": 1,
        "Single": 2,
        "Other": 3

    }

    marriage = st.selectbox(
        "Marital Status",
        list(marriage_options.keys())
    )

    marriage_value = marriage_options[marriage]


    age = st.slider(
        "Age",
        min_value=21,
        max_value=80,
        value=30
    )


# ---------------------------------------------------------
# FINANCIAL INFORMATION
# ---------------------------------------------------------

with st.sidebar.expander(
    "💰 Financial Information",
    expanded=True
):

    st.caption(
        "Total approved credit limit for the customer."
    )

    limit_bal = st.number_input(
        "Credit Limit",
        min_value=10000,
        value=50000,
        step=10000
    )


# ---------------------------------------------------------
# PAYMENT HISTORY
# ---------------------------------------------------------

with st.sidebar.expander(
    "📅 Payment History (Last 6 Months)"
):

    st.info(
        """
Payment status for the last six months.

• -1 = Paid on time

• 0 = No delay

• 1–6 = Months overdue

Higher delays generally increase default risk.
"""
    )

    payment_status = {

        "Paid on time (-1)": -1,
        "No delay (0)": 0,
        "1 Month Late": 1,
        "2 Months Late": 2,
        "3 Months Late": 3,
        "4 Months Late": 4,
        "5 Months Late": 5,
        "6+ Months Late": 6

    }

    pay_0 = payment_status[
        st.selectbox(
            "Latest Payment Status",
            payment_status.keys()
        )
    ]

    pay_2 = payment_status[
        st.selectbox(
            "2 Months Ago",
            payment_status.keys()
        )
    ]

    pay_3 = payment_status[
        st.selectbox(
            "3 Months Ago",
            payment_status.keys()
        )
    ]

    pay_4 = payment_status[
        st.selectbox(
            "4 Months Ago",
            payment_status.keys()
        )
    ]

    pay_5 = payment_status[
        st.selectbox(
            "5 Months Ago",
            payment_status.keys()
        )
    ]

    pay_6 = payment_status[
        st.selectbox(
            "6 Months Ago",
            payment_status.keys()
        )
    ]


# ---------------------------------------------------------
# CREDIT CARD BILLS
# ---------------------------------------------------------

with st.sidebar.expander(
    "🧾 Credit Card Statements"
):

    st.info(
        """
Outstanding balance shown on the customer's
monthly credit card statements.

Higher balances relative to the credit limit
may indicate increased financial risk.
"""
    )

    bill_amt1 = st.number_input(
        "Latest Statement Balance",
        min_value=0
    )

    bill_amt2 = st.number_input(
        "Previous Month Balance",
        min_value=0
    )

    bill_amt3 = st.number_input(
        "3 Months Ago",
        min_value=0
    )

    bill_amt4 = st.number_input(
        "4 Months Ago",
        min_value=0
    )

    bill_amt5 = st.number_input(
        "5 Months Ago",
        min_value=0
    )

    bill_amt6 = st.number_input(
        "6 Months Ago",
        min_value=0
    )


# ---------------------------------------------------------
# MONTHLY PAYMENTS
# ---------------------------------------------------------

with st.sidebar.expander(
    "💳 Monthly Payments"
):

    st.info(
        """
Actual payments made toward the credit card.

Customers consistently paying a large portion
of their balance generally present lower risk.
"""
    )

    pay_amt1 = st.number_input(
        "Latest Payment",
        min_value=0
    )

    pay_amt2 = st.number_input(
        "2 Months Ago",
        min_value=0
    )

    pay_amt3 = st.number_input(
        "3 Months Ago",
        min_value=0
    )

    pay_amt4 = st.number_input(
        "4 Months Ago",
        min_value=0
    )

    pay_amt5 = st.number_input(
        "5 Months Ago",
        min_value=0
    )

    pay_amt6 = st.number_input(
        "6 Months Ago",
        min_value=0
    )


# ---------------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------------

predict = st.sidebar.button(
    "🔍 Predict Credit Risk",
    use_container_width=True,
    type="primary"
)


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

left, right = st.columns([1, 1])


# ---------------------------------------------------------
# CUSTOMER PROFILE
# ---------------------------------------------------------

with left:

    st.subheader("📋 Customer Profile")

    st.markdown(
        f"""
<div class="metric-card">

<h3>Customer Summary</h3>

<hr>

<b>Gender:</b> {gender}

<br><br>

<b>Education:</b> {education}

<br><br>

<b>Marital Status:</b> {marriage}

<br><br>

<b>Age:</b> {age}

<br><br>

<b>Credit Limit:</b> {limit_bal:,}

</div>
""",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# RISK ASSESSMENT
# ---------------------------------------------------------

with right:

    st.subheader("🛡️ Risk Assessment")

    if predict:

        input_data = pd.DataFrame({

            "LIMIT_BAL":[limit_bal],
            "SEX":[sex],
            "EDUCATION":[education_value],
            "MARRIAGE":[marriage_value],
            "AGE":[age],

            "PAY_0":[pay_0],
            "PAY_2":[pay_2],
            "PAY_3":[pay_3],
            "PAY_4":[pay_4],
            "PAY_5":[pay_5],
            "PAY_6":[pay_6],

            "BILL_AMT1":[bill_amt1],
            "BILL_AMT2":[bill_amt2],
            "BILL_AMT3":[bill_amt3],
            "BILL_AMT4":[bill_amt4],
            "BILL_AMT5":[bill_amt5],
            "BILL_AMT6":[bill_amt6],

            "PAY_AMT1":[pay_amt1],
            "PAY_AMT2":[pay_amt2],
            "PAY_AMT3":[pay_amt3],
            "PAY_AMT4":[pay_amt4],
            "PAY_AMT5":[pay_amt5],
            "PAY_AMT6":[pay_amt6]

        })


        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        risk_probability = probability[0][1] * 100

        confidence = max(probability[0]) * 100


        if risk_probability < 30:

            risk_level = "🟢 Low Risk"

            recommendation = (
                "Approve with standard verification."
            )

        elif risk_probability < 60:

            risk_level = "🟡 Medium Risk"

            recommendation = (
                "Manual review is recommended."
            )

        else:

            risk_level = "🔴 High Risk"

            recommendation = (
                "Further verification is strongly recommended."
            )


        st.markdown(f"""

<div class="metric-card">

<h2 style="text-align:center;">
{risk_level}
</h2>

<hr>

<h1 style="text-align:center;">
{risk_probability:.2f}%
</h1>

<h4 style="text-align:center;">
Default Probability
</h4>

<hr>

<b>Model Confidence</b>

<br>

{confidence:.2f}%

<br><br>

<b>Recommendation</b>

<br>

{recommendation}

</div>

""", unsafe_allow_html=True)


        st.plotly_chart(
            risk_gauge(
                risk_probability
            ),
            use_container_width=True
        )


        st.progress(
            int(risk_probability)
        )


        st.subheader(
            "🔍 Key Risk Indicators"
        )


        risk_factors = []


        if pay_0 > 0:

            risk_factors.append(
                f"🔴 Latest payment is {pay_0} month(s) overdue."
            )


        if pay_2 > 0:

            risk_factors.append(
                "🔴 Previous payment history shows delays."
            )


        if limit_bal > 0:

            utilization = (
                bill_amt1 / limit_bal
            )


            if utilization >= 0.80:

                risk_factors.append(
                    "🔴 Credit utilization exceeds 80%."
                )

            elif utilization <= 0.30:

                risk_factors.append(
                    "🟢 Credit utilization is below 30%."
                )


        if bill_amt1 > 0:

            payment_ratio = (
                pay_amt1 / bill_amt1
            )


            if payment_ratio < 0.20:

                risk_factors.append(
                    "🔴 Latest payment covered less than 20% of the outstanding balance."
                )

            elif payment_ratio >= 1:

                risk_factors.append(
                    "🟢 Latest statement was fully paid."
                )


        if age < 25:

            risk_factors.append(
                "🟡 Younger applicants generally have limited credit history."
            )


        if age > 55:

            risk_factors.append(
                "🟢 Mature applicants often demonstrate more stable repayment behaviour."
            )


        if pay_0 <= 0 and pay_2 <= 0:

            risk_factors.append(
                "🟢 No recent payment delays detected."
            )


        if len(risk_factors) == 0:

            risk_factors.append(
                "No significant positive or negative indicators detected."
            )


        positive = 0
        negative = 0
        neutral = 0


        for factor in risk_factors:

            if factor.startswith("🔴"):

                negative += 1
                st.error(factor)

            elif factor.startswith("🟢"):

                positive += 1
                st.success(factor)

            else:

                neutral += 1
                st.warning(factor)


        st.divider()


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Positive Indicators",
            positive
        )

        c2.metric(
            "Risk Indicators",
            negative
        )

        c3.metric(
            "Neutral Indicators",
            neutral
        )


        customer_data = {

            "Gender": gender,
            "Education": education,
            "Marital Status": marriage,
            "Age": age,
            "Credit Limit": f"{limit_bal:,}"

        }


        pdf = generate_pdf(

            customer_data,

            risk_level,

            risk_probability,

            confidence,

            risk_factors

        )


        st.download_button(

            "📄 Download Credit Report",

            pdf,

            file_name="CreditGuard_Report.pdf",

            mime="application/pdf",

            use_container_width=True

        )

    else:

        st.info(
            "Complete the customer information and click **Predict Credit Risk**."
        )
# ---------------------------------------------------------
# MODEL MONITORING
# ---------------------------------------------------------

st.divider()

st.subheader("📊 Model Monitoring Dashboard")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Algorithm",
    "Random Forest"
)

m2.metric(
    "Accuracy",
    "81.78%"
)

m3.metric(
    "ROC-AUC",
    "77.05%"
)

m4.metric(
    "Features",
    "23"
)


st.markdown("---")


left, right = st.columns([1,1])


with left:

    st.markdown("""
### 🤖 Model Information

- **Algorithm:** Random Forest Classifier
- **Trees:** 200
- **Maximum Depth:** 10
- **Minimum Samples Split:** 10
- **Minimum Samples Leaf:** 5
- **Categorical Encoding:** One-Hot Encoding
- **Preprocessing:** Scikit-learn Pipeline
- **Prediction:** Probability-based Binary Classification
""")


with right:

    st.markdown("""
### 📈 Dataset Information

- **Dataset:** UCI Default of Credit Card Clients
- **Records:** 30,000
- **Input Features:** 23
- **Target:** Default Payment (Next Month)
- **Missing Values:** None
- **Framework:** Scikit-learn
""")


st.markdown("---")


st.subheader("💡 Model Insights")


i1, i2, i3 = st.columns(3)


with i1:

    st.info(
"""
### Strengths

✅ Pipeline-based preprocessing

✅ Random Forest classifier

✅ Probability prediction

✅ Professional dashboard

✅ Downloadable PDF report
"""
    )


with i2:

    st.warning(
"""
### Important Notes

• This is a portfolio project.

• Predictions should not be used for real lending decisions.

• Educational labels follow the original dataset encoding.

• Payment history strongly influences predictions.
"""
    )


with i3:

    st.success(
"""
### Technologies

• Python

• Streamlit

• Scikit-learn

• Plotly

• Pandas

• Joblib

• ReportLab
"""
    )


# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------

with st.expander("ℹ️ About CreditGuard AI"):

    st.markdown("""
CreditGuard AI is an end-to-end Machine Learning portfolio project that predicts
the probability of credit default using customer demographic information,
credit history, billing records and payment behaviour.

### Key Features

- AI-powered credit risk prediction
- Probability-based risk scoring
- Interactive risk gauge
- Business-friendly risk indicators
- PDF credit assessment report
- Responsive Streamlit dashboard
- End-to-end Scikit-learn pipeline

This application demonstrates a complete Machine Learning workflow including:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Hyperparameter tuning
5. Pipeline serialization
6. Interactive deployment
""")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
"""
<hr>

<div style="text-align:center;color:#6B7280;font-size:14px;">

<h4>🛡️ CreditGuard AI</h4>

AI-Powered Credit Risk Assessment System

<br><br>

Built using
<strong>Python</strong> •
<strong>Scikit-learn</strong> •
<strong>Streamlit</strong> •
<strong>Plotly</strong>

<br><br>

Machine Learning Portfolio Project © 2026

</div>
""",
unsafe_allow_html=True
)