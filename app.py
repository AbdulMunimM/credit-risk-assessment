import streamlit as st
import pandas as pd
import joblib
import shap
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
# FEATURE EXPLANATION MAP
# ---------------------------------------------------------

feature_labels = {

    "remainder__LIMIT_BAL":
        "Credit Limit",

    "remainder__AGE":
        "Customer Age",

    "remainder__PAY_0":
        "Latest Payment Status",

    "remainder__PAY_2":
        "Payment Status (2 Months Ago)",

    "remainder__PAY_3":
        "Payment Status (3 Months Ago)",

    "remainder__PAY_4":
        "Payment Status (4 Months Ago)",

    "remainder__PAY_5":
        "Payment Status (5 Months Ago)",

    "remainder__PAY_6":
        "Payment Status (6 Months Ago)",


    "remainder__BILL_AMT1":
        "Latest Statement Balance",

    "remainder__BILL_AMT2":
        "Previous Statement Balance",

    "remainder__BILL_AMT3":
        "Statement Balance (3 Months Ago)",

    "remainder__BILL_AMT4":
        "Statement Balance (4 Months Ago)",

    "remainder__BILL_AMT5":
        "Statement Balance (5 Months Ago)",

    "remainder__BILL_AMT6":
        "Statement Balance (6 Months Ago)",


    "remainder__PAY_AMT1":
        "Latest Payment Made",

    "remainder__PAY_AMT2":
        "Payment Made (2 Months Ago)",

    "remainder__PAY_AMT3":
        "Payment Made (3 Months Ago)",

    "remainder__PAY_AMT4":
        "Payment Made (4 Months Ago)",

    "remainder__PAY_AMT5":
        "Payment Made (5 Months Ago)",

    "remainder__PAY_AMT6":
        "Payment Made (6 Months Ago)",

}



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
            "CreditGuard AI - Credit Risk Report",
            styles["Title"]
        )
    )


    content.append(
        Spacer(1,20)
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
            "Important Risk Factors:",
            styles["Heading3"]
        )
    )


    for factor in factors:

        content.append(
            Paragraph(
                factor,
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
                "text":
                "Credit Risk Score"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "steps":[

                    {
                        "range":[0,30],
                        "color":"green"
                    },

                    {
                        "range":[30,60],
                        "color":"yellow"
                    },

                    {
                        "range":[60,100],
                        "color":"red"
                    }

                ]

            }
        )
    )


    fig.update_layout(
        height=300
    )


    return fig




# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top:2rem;
}


.main-header {

    background:linear-gradient(90deg,#1E3A8A,#2563EB);
    padding:30px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:20px;

}


.metric-card {

    background:#F8FAFC;
    padding:25px;
    border-radius:15px;
    border:1px solid #E2E8F0;
    box-shadow:0px 4px 10px rgba(0,0,0,0.05);

}


.footer {

    text-align:center;
    color:gray;
    padding-top:30px;

}


.sidebar-title {

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
        "Male":1,
        "Female":2
    }


    gender = st.selectbox(
        "Gender",
        list(gender_options.keys())
    )

    sex = gender_options[gender]



    education_options = {

        "Unknown":0,
        "High School":1,
        "College":2,
        "University":3,
        "Diploma":4,
        "PhD":5,
        "Other":6

    }


    education = st.selectbox(
        "Education Level",
        list(education_options.keys())
    )


    education_value = education_options[education]



    marriage_options = {

        "Unknown":0,
        "Married":1,
        "Single":2,
        "Other":3

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
        30
    )



# ---------------------------------------------------------
# FINANCIAL INFORMATION
# ---------------------------------------------------------

with st.sidebar.expander(
    "💰 Financial Information",
    expanded=True
):

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

    st.caption(
"""
Payment status:

-1 = Paid on time
0 = No delay
1+ = Months delayed

Higher delays increase default risk.
"""
    )


    pay_options = {

        "Paid on time (-1)": -1,
        "No delay (0)":0,
        "1 month delay":1,
        "2 months delay":2,
        "3 months delay":3,
        "4 months delay":4,
        "5 months delay":5,
        "6+ months delay":6

    }


    pay_0 = pay_options[
        st.selectbox(
            "Latest Payment Status",
            list(pay_options.keys())
        )
    ]


    pay_2 = pay_options[
        st.selectbox(
            "Payment Status (2 Months Ago)",
            list(pay_options.keys())
        )
    ]


    pay_3 = pay_options[
        st.selectbox(
            "Payment Status (3 Months Ago)",
            list(pay_options.keys())
        )
    ]


    pay_4 = pay_options[
        st.selectbox(
            "Payment Status (4 Months Ago)",
            list(pay_options.keys())
        )
    ]


    pay_5 = pay_options[
        st.selectbox(
            "Payment Status (5 Months Ago)",
            list(pay_options.keys())
        )
    ]


    pay_6 = pay_options[
        st.selectbox(
            "Payment Status (6 Months Ago)",
            list(pay_options.keys())
        )
    ]



# ---------------------------------------------------------
# BILL AMOUNTS
# ---------------------------------------------------------

with st.sidebar.expander(
    "🧾 Credit Card Bills"
):

    bill_amt1 = st.number_input(
        "Latest Statement Balance",
        min_value=0
    )

    bill_amt2 = st.number_input(
        "Previous Statement Balance",
        min_value=0
    )

    bill_amt3 = st.number_input(
        "Statement Balance (3 Months Ago)",
        min_value=0
    )

    bill_amt4 = st.number_input(
        "Statement Balance (4 Months Ago)",
        min_value=0
    )

    bill_amt5 = st.number_input(
        "Statement Balance (5 Months Ago)",
        min_value=0
    )

    bill_amt6 = st.number_input(
        "Statement Balance (6 Months Ago)",
        min_value=0
    )



# ---------------------------------------------------------
# PAYMENT AMOUNTS
# ---------------------------------------------------------

with st.sidebar.expander(
    "💳 Monthly Payments"
):

    pay_amt1 = st.number_input(
        "Latest Payment Made",
        min_value=0
    )

    pay_amt2 = st.number_input(
        "Payment Made (2 Months Ago)",
        min_value=0
    )

    pay_amt3 = st.number_input(
        "Payment Made (3 Months Ago)",
        min_value=0
    )

    pay_amt4 = st.number_input(
        "Payment Made (4 Months Ago)",
        min_value=0
    )

    pay_amt5 = st.number_input(
        "Payment Made (5 Months Ago)",
        min_value=0
    )

    pay_amt6 = st.number_input(
        "Payment Made (6 Months Ago)",
        min_value=0)



predict = st.sidebar.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
)



# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

left,right = st.columns([1,1])



with left:

    st.subheader(
        "📋 Customer Profile"
    )


    st.markdown(
f"""

<div class="metric-card">

<b>Gender:</b> {gender}

<br><br>

<b>Education:</b> {education}

<br><br>

<b>Marriage:</b> {marriage}

<br><br>

<b>Age:</b> {age}

<br><br>

<b>Credit Limit:</b> {limit_bal:,}

</div>

""",
unsafe_allow_html=True
)



with right:


    st.subheader(
        "🛡️ Risk Assessment"
    )


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



        probability = model.predict_proba(
            input_data
        )


        risk_probability = probability[0][1]*100

        confidence = max(probability[0])*100



        if risk_probability < 30:

            risk_level="Low Risk 🟢"
            recommendation="Customer has low default probability."


        elif risk_probability < 60:

            risk_level="Medium Risk 🟡"
            recommendation="Customer requires additional review."


        else:

            risk_level="High Risk 🔴"
            recommendation="Customer requires verification."



        st.markdown(
f"""

<div class="metric-card">

<h2>
{risk_level}
</h2>

<h1>
{risk_probability:.2f}%
</h1>


<b>Default Probability</b>

<br><br>

<b>Confidence:</b>
{confidence:.2f}%


<br><br>

<b>Recommendation:</b>

<br>

{recommendation}

</div>

""",
unsafe_allow_html=True
)



        # Risk Gauge

        st.plotly_chart(
            risk_gauge(risk_probability),
            use_container_width=True
        )



        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        st.subheader(
            "🔍 Why This Prediction?"
        )


        shap_factors=[]


        try:

            preprocessor=model.named_steps["preprocessor"]

            rf_model=model.named_steps["model"]


            transformed=preprocessor.transform(
                input_data
            )


            features=preprocessor.get_feature_names_out()


            explainer=shap.TreeExplainer(
                rf_model
            )


            values=explainer.shap_values(
                transformed
            )


            if isinstance(values,list):

                values=values[1][0]

            elif len(values.shape)==3:

                values=values[0,:,1]

            else:

                values=values[0]



            shap_df=pd.DataFrame({

                "Feature":features,
                "Impact":values

            })


            shap_df["Display"]=shap_df["Feature"].map(
                feature_labels
            ).fillna(
                shap_df["Feature"]
            )


            shap_df["Abs"]=shap_df["Impact"].abs()


            shap_df=shap_df.sort_values(
                "Abs",
                ascending=False
            ).head(8)



            chart_df=shap_df[
                ["Display","Impact"]
            ]


            st.bar_chart(
                chart_df.set_index("Display")
            )



            for _,row in shap_df.iterrows():

                if row["Impact"]>0:

                    text=f"🔴 {row['Display']} increased risk"

                    st.error(text)

                else:

                    text=f"🟢 {row['Display']} reduced risk"

                    st.success(text)



                shap_factors.append(text)



        except Exception as e:

            st.warning(
                f"SHAP unavailable: {e}"
            )



        # -------------------------------------------------
        # PDF REPORT
        # -------------------------------------------------

        customer_data={

            "Gender":gender,
            "Education":education,
            "Age":age,
            "Credit Limit":limit_bal

        }


        pdf=generate_pdf(

            customer_data,
            risk_level,
            risk_probability,
            confidence,
            shap_factors

        )


        st.download_button(

            "📄 Download Credit Report",

            pdf,

            file_name="credit_report.pdf",

            mime="application/pdf"

        )


    else:

        st.info(
            "Enter customer information and click Predict"
        )



# ---------------------------------------------------------
# MODEL MONITORING
# ---------------------------------------------------------

st.divider()

st.subheader(
    "📊 Model Monitoring"
)


m1,m2,m3,m4=st.columns(4)


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



# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
"""
<div class="footer">

CreditGuard AI

<br>

Machine Learning Portfolio Project

<br>

Built with ❤️ using Streamlit & Scikit-Learn

</div>
""",
unsafe_allow_html=True
)