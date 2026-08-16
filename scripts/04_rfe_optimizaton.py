"""
Step 4: RFE optimization over different panel sizes
---------------------------------------------------
Compares RFE-selected panels of k = 5, 8, 10, 12, 15, 20,
25, 30, 39 miRNAs using a final logistic regression,
and reports test accuracy / AUC / sensitivity / specificity.

Input : results/data_ml.csv
Output: results/rfe_comparison.csv
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, confusion_matrix,
)

# ===================== CONFIG =====================
K_VALUES = [5, 8, 10, 12, 15, 20, 25, 30, 39]
RANDOM_STATE = 42
TEST_SIZE = 0.30
# ==================================================

# 1) Load and split
data_ml = pd.read_csv("results/data_ml.csv", sep="\t", index_col=0)
X = data_ml.drop(columns="Target")
y = data_ml["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 2) Impute missing values (fit on train only)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 3) Loop over candidate panel sizes
results = []

for k in K_VALUES:
    # Feature selection on train only
    base_estimator = LogisticRegression(
        solver="liblinear", max_iter=10000, random_state=RANDOM_STATE
    )
    selector = RFE(estimator=base_estimator,
                   n_features_to_select=k, step=1)
    selector.fit(X_train_imputed, y_train)

    X_train_reduced = selector.transform(X_train_imputed)
    X_test_reduced = selector.transform(X_test_imputed)

    # Final model on selected features
    final_model = LogisticRegression(
        solver="liblinear", max_iter=10000, random_state=RANDOM_STATE
    )
    final_model.fit(X_train_reduced, y_train)

    y_pred = final_model.predict(X_test_reduced)
    y_prob = final_model.predict_proba(X_test_reduced)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    sens = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    selected_features = X_train.columns[selector.support_]

    results.append({
        "k": k,
        "accuracy": acc,
        "roc_auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "features": list(selected_features),
    })
    print(f"[INFO] k={k:>2} | acc={acc:.3f} | auc={auc:.3f} | "
          f"sens={sens:.3f} | spec={spec:.3f}")

# 4) Rank by ROC-AUC
results_df = pd.DataFrame(results).sort_values(
    "roc_auc", ascending=False
)

print("\n[INFO] Ranked by ROC-AUC:")
print(results_df[["k", "accuracy", "roc_auc",
                  "sensitivity", "specificity"]])

# 5) Save
results_df.to_csv("results/rfe_comparison.csv", index=False)
print("\n[INFO] Saved to results/rfe_comparison.csv")
