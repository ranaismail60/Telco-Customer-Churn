"""
Phase 5 — Model Training & Evaluation.
Trains Logistic Regression, Random Forest, XGBoost, and KNN, compares
them on Accuracy/Precision/Recall/F1/ROC-AUC, plots ROC curves, tunes
the top 2 with RandomizedSearchCV, and saves the final chosen model.

Usage:
    python scripts/05_train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)

results = []
roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

# WHY Recall/F1 over Accuracy: churn is imbalanced (~26% positive class),
# so a model predicting "no churn" for everyone would already score ~74%
# accuracy while catching zero at-risk customers. Missing an actual churner
# (a false negative) costs NexaTel real recurring revenue, while a false
# alarm just costs one unnecessary retention offer -- so we weight Recall
# and F1 more heavily than Accuracy when picking a final model.

# --- ROC curve comparison plot ---
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curves.png")
plt.close()

# --- Tune the top 2 models by F1 ---
top_two = results_df["Model"].head(2).tolist()
print(f"\nTuning top 2 models: {top_two}")

tuned_models = {}

if "Random Forest" in top_two:
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), rf_grid,
        n_iter=4, scoring="f1", cv=3, random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    tuned_models["Random Forest"] = search.best_estimator_
    print("Random Forest best params:", search.best_params_)

if "XGBoost" in top_two and HAS_XGB:
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg_lambda": [1, 5, 10],
    }
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42), xgb_grid,
        n_iter=10, scoring="f1", cv=3, random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    tuned_models["XGBoost"] = search.best_estimator_
    print("XGBoost best params:", search.best_params_)

if "Logistic Regression" in top_two:
    lr_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]}
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=1000, random_state=42), lr_grid,
        n_iter=5, scoring="f1", cv=3, random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    tuned_models["Logistic Regression"] = search.best_estimator_
    print("Logistic Regression best params:", search.best_params_)

# --- Pick final model: highest F1 among tuned models ---
final_scores = {}
for name, model in tuned_models.items():
    preds = model.predict(X_test)
    final_scores[name] = f1_score(y_test, preds)

best_name = max(final_scores, key=final_scores.get)
best_model = tuned_models[best_name]
print(f"\nFinal model selected: {best_name} (F1={final_scores[best_name]:.4f})")

joblib.dump(best_model, MODEL_DIR / "model.pkl")
joblib.dump(best_name, MODEL_DIR / "model_name.pkl")
print(f"Saved final model to {MODEL_DIR / 'model.pkl'}")

print(
    "\nNEXT (do yourself): open reports/model_comparison.csv and write a short "
    "paragraph — which metric did you optimize for, why is that the right "
    "tradeoff for NexaTel, and what does the confusion matrix tell a "
    "non-technical stakeholder?"
)
