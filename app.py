from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load artifact
artifact = joblib.load("artifacts/model_data.joblib")

model = artifact["model"]
features = artifact["features"]
woe_cont = artifact["woe_cont_dict"]
bin_edges = artifact["bin_edges_dict"]
woe_cat = artifact["woe_cat_dict"]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

# ===============================
# INPUT SCHEMA (RAW FEATURES)
# ===============================
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


# ===============================
# WOE TRANSFORM
# ===============================
def apply_woe(df):

    # ===============================
    # CONTINUOUS
    # ===============================
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

    # ===============================
    # CATEGORICAL
    # ===============================
    for col in woe_cat:
        if col in df.columns:
            df[col + "_woe"] = (
                df[col]
                .map(woe_cat[col])
                .astype(float)
                .fillna(0)
            )

    return df   # 🔥 THIS WAS MISSING
# ===============================
# PREDICTION
# ===============================
@app.post("/predict")
def predict(data: LoanInput):

    df = pd.DataFrame([data.dict()])

    # Apply WOE
    df = apply_woe(df)

    # Select model features
    df = df[features]

    prob = model.predict_proba(df)[0][1]

    return {
        "default_probability": float(prob),
        "risk_flag": int(prob > 0.6)
    }