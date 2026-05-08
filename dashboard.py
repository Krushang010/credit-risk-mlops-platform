import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("📊 ML Monitoring Dashboard")

st.markdown("""
Monitor data drift, alerts, and model stability over time.
""")

# =========================
# LOAD METRICS
# =========================

df = pd.read_csv("metrics/drift_history.csv")

# =========================
# DISPLAY RAW DATA
# =========================

st.subheader("📋 Monitoring History")

st.dataframe(df, use_container_width=True)

# =========================
# DRIFT RATIO CHART
# =========================

st.subheader("📈 Drift Ratio Over Time")

fig1 = px.line(
    df,
    x="date",
    y="drift_ratio",
    markers=True,
    title="Drift Ratio Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# DRIFTED COLUMNS CHART
# =========================

st.subheader("📉 Drifted Columns")

fig2 = px.bar(
    df,
    x="date",
    y="drifted_columns",
    title="Drifted Columns Count"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# ALERT SUMMARY
# =========================

st.subheader("🚨 Alert Summary")

alert_count = (df["alert"] == "YES").sum()

st.metric(
    label="Total Alerts Triggered",
    value=alert_count
)

# =========================
# FILTER ALERT ROWS
# =========================

st.subheader("⚠️ High Drift Events")

alerts_df = df[df["alert"] == "YES"]

st.dataframe(alerts_df, use_container_width=True)