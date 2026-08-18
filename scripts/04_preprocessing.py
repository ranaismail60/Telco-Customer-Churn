"""
Phase 4 — Preprocessing & Scaling.
Stratified split FIRST, then scale (fit on train only), then balance
classes with SMOTE on the training set only. Saves reproducible
train/test files plus the fitted scaler.

Usage:
    python scripts/04_preprocessing.py
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "engineered.csv"
OUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "churn_binary"

# These model families need scaled input; tree-based models
# (Random Forest, XGBoost, LightGBM) split on raw thresholds and don't
# care about feature scale, so scaling doesn't help or hurt them.
NEEDS_SCALING = ["Logistic Regression", "SVM", "KNN"]


def main():
    df = pd.read_csv(IN_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 1. Split FIRST, before anything touches the target distribution,
    # stratified because churn is imbalanced (~26% positive class).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Class balance before SMOTE:")
    print(y_train.value_counts(normalize=True).round(3))

    # 2. Scale — fit on TRAIN only, then transform both. Fitting on test
    # data would leak test-set statistics (mean/std) into training,
    # inflating your evaluation scores.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # 3. Balance classes with SMOTE — fit ONLY on training data, never
    # on test data (test data must reflect the real-world distribution).
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    print("Class balance after SMOTE:")
    print(y_train_bal.value_counts(normalize=True).round(3))

    # Save everything needed to reproduce this without re-running preprocessing
    X_train_bal.to_csv(OUT_DIR / "X_train.csv", index=False)
    y_train_bal.to_csv(OUT_DIR / "y_train.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)
    # Also keep the unscaled test set — useful for tree models where
    # scaling isn't needed and for sanity-checking predictions.
    X_test.to_csv(OUT_DIR / "X_test_unscaled.csv", index=False)

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")

    print(f"\nSaved processed data to {OUT_DIR}")
    print(f"Saved scaler to {MODEL_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
