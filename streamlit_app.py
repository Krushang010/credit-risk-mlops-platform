import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Credit Risk ML Platform",
    layout="wide"
)

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🔍 Loan Prediction",
        "📊 Monitoring Dashboard"
    ]
)

# =====================================================
# PAGE 1 — LOAN PREDICTION
# =====================================================

if page == "🔍 Loan Prediction":

    st.markdown(
        "<h1 style='text-align:center;color:#0e76a8;'>🔍 Loan Default Risk Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='text-align:center;color:gray;'>Assess Probability of Default (PD)</h4>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================
    # INPUTS
    # =========================

    st.markdown("### 👤 Personal & Financial Information")

    row1 = st.columns(3)

    with row1[0]:
        age = st.number_input('📅 Age', 18, 100, 28)

    with row1[1]:
        income = st.number_input('💼 Annual Income (₹)', 0, value=1200000)

    with row1[2]:
        loan_amount = st.number_input('🏦 Loan Amount (₹)', 0, value=2560000)

    # =========================
    # DERIVED FEATURES
    # =========================

    loan_to_income_ratio = (
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

    st.markdown("### 💰 Loan Information")

    row2 = st.columns(3)

    with row2[0]:
        st.markdown(
            f"**Loan-to-Income Ratio:** `{loan_to_income_ratio:.2f}`"
        )

    with row2[1]:
        loan_tenure_months = st.number_input(
            '📆 Tenure (months)',
            0,
            value=36
        )

    with row2[2]:
        avg_dpd_per_delinquency = st.number_input(
            '📊 Avg DPD',
            0,
            value=20
        )

    st.markdown("### 📉 Credit Behavior")

    row3 = st.columns(3)

    with row3[0]:
        delinquency_ratio = st.slider(
            'Delinquency (%)',
            0,
            100,
            30
        )

    with row3[1]:
        credit_utilization_ratio = st.slider(
            'Utilization (%)',
            0,
            100,
            30
        )

    with row3[2]:
        num_open_accounts = st.selectbox(
            'Open Accounts',
            [0,1,2,3,4],
            index=1
        )

    st.markdown("### 🏠 Loan Details")

    row4 = st.columns(3)

    with row4[0]:
        residence_type = st.selectbox(
            'Residence',
            ['Owned','Rented','Mortgage']
        )

    with row4[1]:
        loan_purpose = st.selectbox(
            'Purpose',
            ['Education','Home','Auto','Personal']
        )

    with row4[2]:
        loan_type = st.selectbox(
            'Type',
            ['Unsecured','Secured']
        )

    st.markdown("---")

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button('🔎 Predict', use_container_width=True):

        # 🔥 Render API URL
        url = "https://loan-default-api-4snt.onrender.com/predict"

        # =========================
        # ENGINEERED FEATURES
        # =========================

        util_enquiry_interaction = (
            credit_utilization_ratio
            * avg_dpd_per_delinquency
        )

        account_stability = (
            loan_tenure_months / age
            if age > 0 else 0
        )

        loan_to_balance = (
            loan_amount / income
            if income > 0 else 0
        )

        # =========================
        # API PAYLOAD
        # =========================

        payload = {

            "credit_utilization_ratio":
                credit_utilization_ratio,

            "delinquency_ratio":
                delinquency_ratio,

            "loan_to_income":
                loan_to_income_ratio,

            "loan_tenure_months":
                loan_tenure_months,

            "net_disbursement":
                net_disbursement,

            "age":
                age,

            "number_of_open_accounts":
                num_open_accounts,

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

        with st.spinner("⏳ Predicting..."):

            try:

                response = requests.post(
                    url,
                    json=payload,
                    timeout=20
                )

            except requests.exceptions.RequestException as e:

                st.error("🚨 API request failed")

                st.write(str(e))

                st.stop()

        if response.status_code == 200:

            result = response.json()

            prob = result["default_probability"]

            threshold = result["threshold_used"]

            st.success(
                f"📊 Probability of Default: {prob:.2%}"
            )

            if prob >= threshold:

                st.error("⚠️ High Risk")

            elif prob >= (threshold * 0.5):

                st.warning("⚠️ Medium Risk")

            else:

                st.success("✅ Low Risk")

        else:

            st.error(
                f"🚨 API Error: {response.status_code}"
            )

            st.write(response.text)

    st.markdown("---")

    st.caption(
        "⚠️ First request may take time due to server sleep"
    )

# =====================================================
# PAGE 2 — MONITORING DASHBOARD
# =====================================================

elif page == "📊 Monitoring Dashboard":

    st.title("📊 ML Monitoring Dashboard")

    st.markdown("""
    Monitor data drift, alerts, and model stability over time.
    """)

    # =========================
    # LOAD DATA
    # =========================

    import os

    drift_path = "metrics/drift_history.csv"

    if os.path.exists(drift_path):

        df = pd.read_csv(drift_path)

        # =========================
        # RAW TABLE
        # =========================

        st.subheader("📋 Monitoring History")

        st.dataframe(
            df,
            use_container_width=True
        )

        # =========================
        # DRIFT RATIO TREND
        # =========================

        st.subheader("📈 Drift Ratio Over Time")

        fig1 = px.line(
            df,
            x="date",
            y="drift_ratio",
            markers=True,
            title="Drift Ratio Trend"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # =========================
        # DRIFTED COLUMNS
        # =========================

        st.subheader("📉 Drifted Columns")

        fig2 = px.bar(
            df,
            x="date",
            y="drifted_columns",
            title="Drifted Columns Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =========================
        # ALERT SUMMARY
        # =========================

        st.subheader("🚨 Alert Summary")

        alert_count = (
            df["alert"] == "YES"
        ).sum()

        st.metric(
            label="Total Alerts Triggered",
            value=alert_count
        )

        # =========================
        # ALERT EVENTS
        # =========================

        st.subheader("⚠️ High Drift Events")

        alerts_df = df[
            df["alert"] == "YES"
        ]

        st.dataframe(
            alerts_df,
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ No monitoring history available yet."
        )