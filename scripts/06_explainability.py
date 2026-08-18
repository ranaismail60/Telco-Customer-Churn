"""
Phase 6 — Model Explainability with SHAP.
Computes global feature importance and a reusable function to get the
top 3 "reasons" for any single customer's prediction (used by the
Flask backend in Phase 7).

Usage:
    python scripts/06_explainability.py
"""

import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

model = joblib.load(MODEL_DIR / "model.pkl")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
except Exception:
    explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)

# Handle both binary-classifier output shapes across sklearn/xgboost versions
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# --- Global summary plot ---
plt.figure()
shap.summary_plot(sv, X_test, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()
print(f"Saved global SHAP summary to {FIG_DIR / 'shap_summary.png'}")


def top_reasons_for_customer(row_index: int, n: int = 3):
    """Return the top-N features pushing this one customer's prediction,
    with direction (toward churn / away from churn)."""
    row_shap = sv[row_index]
    feature_names = X_test.columns
    contributions = list(zip(feature_names, row_shap))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, val in contributions[:n]:
        direction = "increases" if val > 0 else "decreases"
        reasons.append(f"{feat} {direction} churn risk")
    return reasons


if __name__ == "__main__":
    example = top_reasons_for_customer(0)
    print("\nExample — top reasons for test customer #0:")
    for r in example:
        print(" -", r)
