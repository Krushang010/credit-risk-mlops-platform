from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import json
from datetime import datetime
import os
from typing import Optional
# =========================
# LOGGING SETUP (JSON logs)
# =========================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# =========================
# STRUCTURED DATA LOGGING (for drift)
# =========================
LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_input(data_dict):
    df_log = pd.DataFrame([data_dict])

    file_name = datetime.now().strftime("%Y-%m-%d") + ".csv"
    file_path = os.path.join(LOG_DIR, file_name)

    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        df_log.to_csv(file_path, index=False)
    else:
        df_log.to_csv(file_path, mode='a', header=False, index=False)

# =========================
# LOAD MODEL ARTIFACT
# =========================
artifact = joblib.load("artifacts/model_data.joblib")

model = artifact["model"]
features = artifact["features"]
woe_cont = artifact["woe_cont_dict"]
bin_edges = artifact["bin_edges_dict"]
woe_cat = artifact["woe_cat_dict"]

# =========================
# FASTAPI INIT
# =========================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

# =========================
# INPUT SCHEMA
# =========================
class LoanInput(BaseModel):
    credit_utilization_ratio: float
    delinquency_ratio: float
    loan_to_income: float
    loan_tenure_months: int
    net_disbursement: float
    age: int
    number_of_open_accounts: int
    loan_purpose: str
    residence_type: str
    loan_type: str
    util_enquiry_interaction: float
    account_stability: float
    loan_to_balance: float
    source: Optional[str] = "unknown"

# =========================
# WOE TRANSFORMATION
# =========================
def apply_woe(df):

    # Continuous
    for col in woe_cont:
        bins = bin_edges[col]

        df["_bin"] = pd.cut(df[col], bins=bins, include_lowest=True)

        df[col + "_woe"] = (
            df["_bin"]
            .map(woe_cont[col])
            .astype(float)
            .fillna(0)
        )

        df.drop(columns=["_bin"], inplace=True)

    # Categorical
    for col in woe_cat:
        if col in df.columns:
            df[col + "_woe"] = (
                df[col]
                .map(woe_cat[col])
                .astype(float)
                .fillna(0)
            )

    return df

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: LoanInput):

    input_data = data.dict()

    # Extract source (default = unknown)
    source = input_data.get("source", "unknown")

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Apply WOE
    df = apply_woe(df)

    # Select features
    df = df[features]

    # 🔥 Add metadata for logging
    log_record = df.to_dict(orient="records")[0]
    log_record["source"] = source
    log_record["timestamp"] = str(datetime.now())

    # 🔥 Structured logging (CSV)
    log_input(log_record)

    # Model prediction
    prob = model.predict_proba(df)[0][1]

    output = {
        "default_probability": float(prob),
        "risk_flag": int(prob > 0.6)
    }

    # JSON logging (debug + audit)
    log_entry = {
        "timestamp": str(datetime.now()),
        "source": source,
        "input": input_data,
        "output": output
    }

    logging.info(json.dumps(log_entry))

    return output