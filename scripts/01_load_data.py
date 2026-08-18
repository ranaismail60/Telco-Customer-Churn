"""
Phase 1 — Load the raw Telco Customer Churn CSV into the normalized
SQLite database defined in sql/schema.sql.

Usage:
    python scripts/01_load_data.py

Expects the CSV at: data/telco_churn.csv
(download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
 and rename/place it there)
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "telco_churn.csv"
DB_PATH = ROOT / "data" / "nexatel.db"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"


def load_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {CSV_PATH}. Download the Kaggle Telco Churn CSV "
            f"and save it there as 'telco_churn.csv'."
        )
    df = pd.read_csv(CSV_PATH)

    # Known data quality issue in this dataset: TotalCharges is stored as
    # text and has blank strings for customers with tenure == 0 (brand new
    # customers who haven't been billed yet). Coerce to numeric, blanks -> NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def build_database(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # (Re)build schema
    with open(SCHEMA_PATH, "r") as f:
        cur.executescript(f.read())

    customers = df[["customerID", "gender", "SeniorCitizen", "Partner",
                     "Dependents", "tenure"]].rename(columns={
        "customerID": "customer_id",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
    })

    accounts = df[["customerID", "Contract", "PaperlessBilling",
                    "PaymentMethod", "MonthlyCharges", "TotalCharges"]].rename(columns={
        "customerID": "customer_id",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    })

    services = df[["customerID", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies"]].rename(columns={
        "customerID": "customer_id",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
    })

    churn_status = df[["customerID", "Churn"]].rename(columns={
        "customerID": "customer_id",
        "Churn": "churn",
    })

    customers.to_sql("customers", conn, if_exists="append", index=False)
    accounts.to_sql("accounts", conn, if_exists="append", index=False)
    services.to_sql("services", conn, if_exists="append", index=False)
    churn_status.to_sql("churn_status", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} customers into {DB_PATH}")


if __name__ == "__main__":
    data = load_csv()
    build_database(data)
