````markdown
# 🏦 Credit Risk MLOps Platform

An end-to-end Machine Learning + MLOps system for real-time loan default prediction, monitoring, and production deployment.

This project simulates how modern financial institutions deploy and monitor credit risk models in production environments.

---

# 🚀 Business Problem

Banks and fintech companies must identify whether a customer is likely to default on a loan before approval.

A wrong prediction can lead to:
- Financial loss
- High-risk loan approvals
- Increased NPAs (Non-Performing Assets)

This project solves that problem using Machine Learning, real-time APIs, monitoring systems, and deployment pipelines.

---

# 🎯 Project Goals

✅ Predict probability of loan default  
✅ Build interpretable ML pipeline  
✅ Deploy model as production API  
✅ Create real-time prediction UI  
✅ Monitor production data drift  
✅ Simulate industry-level MLOps workflow  

---

# 🧠 End-to-End System Architecture

```text
                        ┌──────────────────────┐
                        │      User Input      │
                        │  Loan Application    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   Streamlit Frontend UI  │
                    │  (Prediction Dashboard)  │
                    └──────────┬───────────────┘
                               │ API Request
                               ▼
                    ┌──────────────────────────┐
                    │      FastAPI Backend     │
                    │   Real-Time Prediction   │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │ Logistic Regression Model│
                    │  WoE + IV + VIF Features │
                    └──────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌──────────────────┐         ┌────────────────────┐
     │ Prediction Logs  │         │ MLflow + DagsHub   │
     │ Monitoring Data  │         │ Experiment Tracking│
     └────────┬─────────┘         └────────────────────┘
              │
              ▼
     ┌──────────────────────────┐
     │   Evidently AI Monitor   │
     │  Drift Detection System  │
     └──────────┬───────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐   ┌─────────────────┐
│ Email Alerts │   │ Monitoring UI   │
│ Drift Alerts │   │ Drift Dashboard │
└──────────────┘   └─────────────────┘
````

---

# ⚙️ Tech Stack

| Layer               | Technology          |
| ------------------- | ------------------- |
| ML Model            | Logistic Regression |
| Feature Engineering | WoE / IV / VIF      |
| Backend API         | FastAPI             |
| Frontend UI         | Streamlit           |
| Monitoring          | Evidently AI        |
| Experiment Tracking | MLflow + DagsHub    |
| Deployment          | Render              |
| Containerization    | Docker              |
| Visualization       | Plotly              |
| Version Control     | GitHub              |

---

# 📊 Machine Learning Pipeline

## Data Processing

* Missing value handling
* Outlier treatment
* Feature engineering
* Behavioral risk analysis

---

## Feature Engineering

Created domain-driven features such as:

* Loan-to-income ratio
* Credit utilization behavior
* Account stability
* Interaction features

---

## Risk Modeling

Applied:

* WoE (Weight of Evidence)
* IV (Information Value)
* VIF Analysis

to improve:

* interpretability
* feature stability
* multicollinearity reduction

---

## Model Development

Models Tested:

* Logistic Regression
* Random Forest
* XGBoost

Final selection:
✅ Logistic Regression

Reason:

* Highly interpretable
* Industry-friendly
* Stable for risk modeling

---

# 📈 Model Performance

| Metric       | Performance |
| ------------ | ----------- |
| Recall       | ~90%        |
| ROC-AUC      | Strong      |
| Precision    | Balanced    |
| F1 Score     | Optimized   |
| KS Statistic | Strong      |

Primary business focus:

* Detect maximum risky customers
* Minimize false approvals

---

# 🔬 MLflow + DagsHub Tracking

Implemented experiment tracking for:

* Hyperparameters
* Metrics
* Model comparison
* Champion model selection

Tracked:

* Accuracy
* Recall
* Precision
* ROC-AUC
* F1 Score

---

# 🌐 FastAPI Backend

Production-ready API for real-time inference.

### Features

✅ REST API
✅ JSON request handling
✅ Swagger documentation
✅ Cloud deployment support

---

# 🖥️ Streamlit Frontend

Interactive dashboard for:

* Loan risk prediction
* Risk classification
* Monitoring visualization
* Drift analytics

---

# 📡 Monitoring Pipeline

Implemented production-style ML monitoring using Evidently AI.

### Monitoring Features

✅ Data drift detection
✅ Drift history tracking
✅ Monitoring dashboard
✅ Alert system
✅ Drift visualization

---

# 🚨 Drift Alert System

Automated alert triggers when:

* Drift ratio exceeds threshold
* Multiple features drift simultaneously

Notifications include:

* Drift ratio
* Drifted feature count
* Monitoring alerts

---

# 🐳 Dockerized Deployment

Both services containerized separately:

| Service             | Purpose            |
| ------------------- | ------------------ |
| FastAPI Container   | Backend inference  |
| Streamlit Container | Frontend dashboard |

Benefits:

* Environment consistency
* Easy deployment
* Reproducibility

---

# ☁️ Cloud Deployment

Deployed on Render using Docker-based architecture.

### Live Components

✅ Backend API
✅ Frontend Dashboard
✅ Monitoring System

---

# 📂 Project Structure

```text
credit-risk-mlops-platform/
│
├── artifacts/
├── notebooks/
├── reports/
├── metrics/
├── app.py
├── streamlit_app.py
├── monitor.py
├── dashboard.py
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── README.md
```

---

# 🔥 Key Highlights

✅ End-to-End ML System
✅ Real-Time Prediction API
✅ Production Deployment
✅ ML Monitoring Pipeline
✅ Drift Detection System
✅ Dockerized Architecture
✅ MLflow Experiment Tracking
✅ Monitoring Dashboard
✅ Automated Alerting System

---

# 📚 Key Learnings

This project provided practical experience in:

* Production ML deployment
* MLOps workflows
* Monitoring & observability
* API development
* Docker containerization
* Experiment tracking
* Cloud deployment
* Drift monitoring systems

---

# 👨‍💻 Author

Krushang Patel

Data Scientist | Machine Learning | Forecasting | MLOps

```
```
