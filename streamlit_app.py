import streamlit as st
import requests

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Credit Risk ML Platform",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("🔍 Loan Default Risk Prediction")

st.markdown(
    "Assess Probability of Default (PD)"
)

st.markdown("---")

# =========================
# USER INPUTS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

with col2:
    income = st.number_input(
        "Annual Income (₹)",
        100000,
        value=1200000
    )

with col3:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        50000,
        value=2500000
    )

# =========================
# DERIVED FEATURES
# =========================

loan_to_income = (
    loan_amount / income
    if income > 0 else 0
)

processing_fee = loan_amount * 0.02
gst = processing_fee * 0.18

net_disbursement = (
    loan_amount
    - processing_fee
    - gst
)

# =========================
# RISK INPUTS
# =========================

col4, col5, col6 = st.columns(3)

with col4:
    loan_tenure_months = st.slider(
        "Loan Tenure (Months)",
        6,
        240,
        36
    )

with col5:
    delinquency_ratio = st.slider(
        "Delinquency Ratio (%)",
        0,
        100,
        20
    )

with col6:
    credit_utilization_ratio = st.slider(
        "Credit Utilization (%)",
        0,
        100,
        40
    )

# =========================
# ADDITIONAL FEATURES
# =========================

col7, col8, col9 = st.columns(3)

with col7:
    enquiry_count = st.slider(
        "Enquiry Count",
        0,
        10,
        3
    )

with col8:
    years_at_address = st.slider(
        "Years at Current Address",
        0,
        30,
        5
    )

with col9:
    number_of_open_accounts = st.selectbox(
        "Open Accounts",
        [1, 2, 3, 4]
    )

# =========================
# CATEGORICALS
# =========================

col10, col11, col12 = st.columns(3)

with col10:
    residence_type = st.selectbox(
        "Residence Type",
        ["Owned", "Rented", "Mortgage"]
    )

with col11:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Education", "Home", "Auto", "Personal"]
    )

with col12:
    loan_type = st.selectbox(
        "Loan Type",
        ["Secured", "Unsecured"]
    )

# =========================
# ENGINEERED FEATURES
# =========================

# IMPORTANT:
# training used 0-100 scale
# NOT 0-1 scale

util_enquiry_interaction = (
    credit_utilization_ratio
    * enquiry_count
)

account_stability = (
    years_at_address / age
    if age > 0 else 0
)

loan_to_balance = (
    loan_amount / income
    if income > 0 else 0
)

# =========================
# PREDICT
# =========================

if st.button("🔎 Predict Risk"):

    payload = {

        "credit_utilization_ratio":
            credit_utilization_ratio,

        "delinquency_ratio":
            delinquency_ratio,

        "loan_to_income":
            loan_to_income,

        "loan_tenure_months":
            loan_tenure_months,

        "net_disbursement":
            net_disbursement,

        "age":
            age,

        "number_of_open_accounts":
            number_of_open_accounts,

        "loan_purpose":
            loan_purpose,

        "residence_type":
            residence_type,

        "loan_type":
            loan_type,

        "util_enquiry_interaction":
            util_enquiry_interaction,

        "account_stability":
            account_stability,

        "loan_to_balance":
            loan_to_balance,

        "source":
            "streamlit"
    }

    url = "https://loan-default-api-4snt.onrender.com/predict"

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            prob = result["default_probability"]

            st.success(
                f"Probability of Default: {prob:.2%}"
            )

            st.write(
                f"Threshold Used: {result['threshold_used']:.2f}"
            )

            # =========================
            # RISK LABELS
            # =========================

            if prob >= 0.75:

                st.error("⚠️ Very High Risk")

            elif prob >= 0.40:

                st.warning("⚠️ Medium Risk")

            else:

                st.success("✅ Low Risk")

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    except Exception as e:

        st.error(str(e))