import pandas as pd
import os
from evidently import Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from datetime import datetime
import csv
import smtplib
from email.mime.text import MIMEText
# =========================
# EMAIL ALERT FUNCTION
# =========================

def send_alert_email(drift_ratio, drift_count, total_columns):

    sender_email = "krushangpatel180@gmail.com"

    sender_password = "hvdt nawx gvxz igjv"

    receiver_email = "kushnjr11@gmail.com"

    subject = "🚨 ML Drift Alert"

    body = f"""
High data drift detected!

Drift Ratio: {round(drift_ratio, 2)}

Drifted Columns: {drift_count}/{total_columns}

Please review monitoring dashboard and drift report.
"""

    msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:

        server = smtplib.SMTP("smtp.gmail.com", 587)

        server.starttls()

        server.login(sender_email, sender_password)

        server.sendmail(
            sender_email,
            receiver_email,
            msg.as_string()
        )

        server.quit()

        print("✅ Alert email sent successfully")

    except Exception as e:

        print("❌ Failed to send email:", e)

# =========================
# LOAD REFERENCE DATA
# =========================
reference = pd.read_csv("reference_data.csv")

# =========================
# LOAD LATEST LOG FILE
# =========================
LOG_DIR = "data/logs"

files = sorted(os.listdir(LOG_DIR))
latest_file = files[-1]

current_path = os.path.join(LOG_DIR, latest_file)

# 🔥 Load current data FIRST
current = pd.read_csv(current_path)

# 🔥 THEN clean metadata
drop_cols = ["source", "timestamp"]

for col in drop_cols:
    if col in current.columns:
        current = current.drop(columns=[col])

print(f"Using log file: {latest_file}")
print("Reference shape:", reference.shape)
print("Current shape:", current.shape)

# =========================
# CONVERT TO DATASET
# =========================
ref_data = Dataset.from_pandas(reference)
cur_data = Dataset.from_pandas(current)

# =========================
# CREATE METRICS
# =========================
metrics = []

for col in reference.columns:
    metrics.append(ValueDrift(column=col))

metrics.append(DriftedColumnsCount())

# =========================
# RUN REPORT
# =========================
report = Report(metrics=metrics)

result = report.run(reference_data=ref_data, current_data=cur_data)

# =========================
# SAVE REPORT
# =========================
os.makedirs("reports", exist_ok=True)

report_path = f"reports/drift_{latest_file.replace('.csv', '.html')}"

result.save_html(report_path)

print("✅ Drift report generated:", report_path)

# =========================
# EXTRACT DRIFT METRICS
# =========================

# =========================
# CALCULATE DRIFT COUNT MANUALLY
# =========================

drift_count = 0

# =========================
# SIMPLE DRIFT CALCULATION
# =========================

drift_count = 0

threshold = 0.30  # 30% difference threshold

for col in reference.columns:

    ref_mean = reference[col].mean()
    cur_mean = current[col].mean()

    # Avoid divide-by-zero
    if abs(ref_mean) < 1e-6:
        continue

    diff_ratio = abs(cur_mean - ref_mean) / abs(ref_mean)

    print(f"{col} drift ratio: {round(diff_ratio, 2)}")

    if diff_ratio > threshold:
        drift_count += 1

# Total columns
total_columns = len(reference.columns)

# Drift ratio
drift_ratio = drift_count / total_columns
# Alert threshold
alert_flag = "YES" if drift_ratio > 0.3 else "NO"
# Send alert if drift is high
if alert_flag == "YES":

    send_alert_email(
        drift_ratio,
        drift_count,
        total_columns
    )

print("\n===== DRIFT SUMMARY =====")
print("Drifted Columns:", drift_count)
print("Total Columns:", total_columns)
print("Drift Ratio:", round(drift_ratio, 2))
print("Alert Triggered:", alert_flag)

# =========================
# SAVE METRICS HISTORY
# =========================

os.makedirs("metrics", exist_ok=True)

metrics_file = "metrics/drift_history.csv"

file_exists = os.path.exists(metrics_file)

with open(metrics_file, mode="a", newline="") as f:

    writer = csv.writer(f)

    # Header only first time
    if not file_exists:
        writer.writerow([
            "date",
            "drifted_columns",
            "total_columns",
            "drift_ratio",
            "alert"
        ])

    # Save monitoring row
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        drift_count,
        total_columns,
        round(drift_ratio, 2),
        alert_flag
    ])

print("\n✅ Drift metrics saved to:", metrics_file)