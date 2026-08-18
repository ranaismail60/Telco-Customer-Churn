"""
Phase 3 — Feature Engineering.
Builds derived features on top of the joined table and one-hot encodes
categoricals. Saves the engineered (but not yet scaled/split) dataset.

Usage:
    python scripts/03_feature_engineering.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "nexatel.db"
OUT_PATH = ROOT / "data" / "engineered.csv"


def load_joined_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT cu.*, a.contract, a.paperless_billing, a.payment_method,
               a.monthly_charges, a.total_charges,
               s.phone_service, s.multiple_lines, s.internet_service,
               s.online_security, s.online_backup, s.device_protection,
               s.tech_support, s.streaming_tv, s.streaming_movies,
               c.churn
        FROM customers cu
        JOIN accounts a ON cu.customer_id = a.customer_id
        JOIN services s ON cu.customer_id = s.customer_id
        JOIN churn_status c ON cu.customer_id = c.customer_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def fix_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges is blank for tenure==0 customers (haven't been billed
    # yet) -> fill with 0, since they've genuinely paid nothing so far.
    df["total_charges"] = df["total_charges"].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # tenure_group: bucket raw tenure into readable ranges
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 1000],
        labels=["0-12", "13-24", "25-48", "49+"]
    ).astype(str)

    # total_services: count of Yes-valued add-on services
    addon_cols = ["online_security", "online_backup", "device_protection",
                  "tech_support", "streaming_tv", "streaming_movies"]
    df["total_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    # avg_monthly_spend_ratio: total_charges / tenure, careful with tenure=0
    df["avg_monthly_spend_ratio"] = df.apply(
        lambda r: r["total_charges"] / r["tenure"] if r["tenure"] > 0 else r["monthly_charges"],
        axis=1
    )

    # combined_risk_flag: short tenure + month-to-month + no tech support
    df["combined_risk_flag"] = (
        (df["tenure"] < 12) &
        (df["contract"] == "Month-to-month") &
        (df["tech_support"] == "No")
    ).astype(int)

    # payment_risk_flag: manual/check payment methods correlate with churn
    df["payment_risk_flag"] = df["payment_method"].isin(
        ["Electronic check", "Mailed check"]
    ).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal: contract length genuinely has an order (month-to-month < 1yr < 2yr)
    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_ordinal"] = df["contract"].map(contract_order)

    # Target -> binary
    df["churn_binary"] = (df["churn"] == "Yes").astype(int)

    # Binary Yes/No-ish columns -> 0/1
    yesno_cols = ["partner", "dependents", "phone_service", "paperless_billing"]
    for col in yesno_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # Nominal categoricals -> one-hot
    nominal_cols = ["gender", "multiple_lines", "internet_service",
                     "online_security", "online_backup", "device_protection",
                     "tech_support", "streaming_tv", "streaming_movies",
                     "payment_method", "tenure_group"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # Drop columns we no longer need (raw contract text, id, raw churn text)
    df = df.drop(columns=["contract", "churn", "customer_id"])

    return df


if __name__ == "__main__":
    data = load_joined_data()
    data = fix_data_quality(data)
    data = engineer_features(data)
    data = encode_categoricals(data)
    data.to_csv(OUT_PATH, index=False)
    print(f"Engineered dataset saved to {OUT_PATH} — shape: {data.shape}")
    print(
        "\nNEXT (do yourself): for each new feature above, write 1-2 sentences "
        "on why you expect it to help and whether your Phase 2 EDA supports that. "
        "Also double check none of these leak the target (they don't here, "
        "but you should verify that reasoning yourself)."
    )
