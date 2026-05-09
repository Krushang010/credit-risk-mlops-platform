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
# LOGGING SETUP
# =========================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# =========================
# STRUCTURED DATA LOGGING
# =========================

LOG_DIR = "data/logs"

os.makedirs(LOG_DIR, exist_ok=True)

def log_input(data_dict):

    df_log = pd.DataFrame([data_dict])

    file_name = datetime.now().strftime("%Y-%m-%d") + ".csv"

    file_path = os.path.join(LOG_DIR, file_name)

    if not os.path.exists(file_path):

        df_log.to_csv(file_path, index=False)

    else:

        df_log.to_csv(
            file_path,
            mode='a',
            header=False,
            index=False
        )

# =========================
# LOAD MODEL
# =========================

artifact = joblib.load(
    "artifacts/model_data.joblib"
)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

woe_cont = artifact["woe_cont_dict"]
woe_cat = artifact["woe_cat_dict"]
bin_edges = artifact["bin_edges_dict"]

# =========================
# FASTAPI INIT
# =========================

app = FastAPI()

@app.get("/")

def home():

    return {
        "message": "Credit Risk API Running"
    }

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

    # Continuous Variables
    for col in woe_cont:

        bins = bin_edges[col]

        df["_bin"] = pd.cut(
            df[col],
            bins=bins,
            include_lowest=True
        )

        df[col + "_woe"] = (
            df["_bin"]
            .map(woe_cont[col])
            .astype(float)
            .fillna(0)
        )

        df.drop(columns=["_bin"], inplace=True)

    # Categorical Variables
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

    source = input_data.get(
        "source",
        "unknown"
    )

    # Convert to dataframe
    df = pd.DataFrame([input_data])

    # Apply WoE
    df = apply_woe(df)

    # Feature Alignment
    df = df[features]

    # Predict Probability
    prob = model.predict_proba(df)[0][1]

    # Final Output
    output = {

        "default_probability":
            float(prob),

        "risk_flag":
            int(prob > threshold),

        "threshold_used":
            float(threshold)
    }

    # Structured Logging
    log_record = df.to_dict(
        orient="records"
    )[0]

    log_record["prediction_probability"] = float(prob)

    log_record["risk_flag"] = int(
        prob > threshold
    )

    log_record["source"] = source

    log_record["timestamp"] = str(
        datetime.now()
    )

    log_input(log_record)

    # JSON Logs
    logging.info(json.dumps({

        "timestamp":
            str(datetime.now()),

        "source":
            source,

        "input":
            input_data,

        "output":
            output
    }))

    return output