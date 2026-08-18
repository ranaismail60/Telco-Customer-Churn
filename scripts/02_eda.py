"""
Phase 2 — Exploratory Data Analysis.
Pulls data from the SQLite DB (not the raw CSV), profiles it, and saves
plots to reports/figures/.

Usage:
    python scripts/02_eda.py
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save-to-file backend, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "nexatel.db"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


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


def data_quality_report(df: pd.DataFrame) -> None:
    print("\n--- Data Quality ---")
    print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("Duplicate customer_ids:", df["customer_id"].duplicated().sum())
    print("Rows with tenure == 0:", (df["tenure"] == 0).sum())


def univariate_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["tenure"], bins=30, ax=axes[0]).set_title("Tenure distribution")
    sns.histplot(df["monthly_charges"], bins=30, ax=axes[1]).set_title("Monthly Charges")
    sns.histplot(df["total_charges"].dropna(), bins=30, ax=axes[2]).set_title("Total Charges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "univariate_distributions.png")
    plt.close()

    churn_rate = (df["churn"] == "Yes").mean() * 100
    print(f"\nOverall churn rate: {churn_rate:.2f}%")


def bivariate_plots(df: pd.DataFrame) -> None:
    cat_cols = ["contract", "internet_service", "payment_method",
                "tech_support", "paperless_billing"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        churn_rate_by_cat = (
            df.groupby(col)["churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        sns.barplot(x=churn_rate_by_cat.index, y=churn_rate_by_cat.values, ax=axes[i])
        axes[i].set_title(f"Churn rate by {col}")
        axes[i].set_ylabel("Churn rate (%)")
        axes[i].tick_params(axis="x", rotation=45)
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bivariate_churn_by_category.png")
    plt.close()


def correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df[["tenure", "monthly_charges", "total_charges"]].copy()
    numeric_df["churn_binary"] = (df["churn"] == "Yes").astype(int)
    corr = numeric_df.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png")
    plt.close()

    print("\n--- Correlation with churn ---")
    print(corr["churn_binary"].sort_values(ascending=False))


def segment_analysis(df: pd.DataFrame) -> None:
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 1000],
        labels=["0-12", "13-24", "25-48", "49+"]
    )
    pivot = pd.crosstab(
        [df["contract"], df["tenure_bucket"]], df["churn"], normalize="index"
    ) * 100
    print("\n--- Churn % by Contract x Tenure bucket ---")
    print(pivot.round(2))
    pivot.to_csv(ROOT / "reports" / "segment_churn_contract_tenure.csv")


if __name__ == "__main__":
    (ROOT / "reports").mkdir(exist_ok=True)
    data = load_joined_data()
    data_quality_report(data)
    univariate_plots(data)
    bivariate_plots(data)
    correlation_heatmap(data)
    segment_analysis(data)
    print(f"\nFigures saved to {FIG_DIR}")
    print(
        "\nNEXT: open reports/figures/*.png and reports/segment_churn_*.csv, "
        "then write your one-page insights summary in your own words — "
        "which segment is riskiest, and what's the single most surprising finding?"
    )
