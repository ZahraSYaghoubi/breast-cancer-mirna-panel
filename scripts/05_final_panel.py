"""
Step 5: Final 20-miRNA panel
----------------------------
Trains the final logistic regression on the 20 RFE-selected
miRNAs, computes test metrics, prints the coefficient table,
and saves a horizontal bar plot of coefficients.

Input : results/data_ml.csv
Output: figures/top_20_miRNA_panel_coefficients.png
        results/final_panel_coefficients.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    confusion_matrix, classification_report,
)

# ===================== CONFIG =====================
N_FEATURES = 20
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

# 2) Impute (fit on train only)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 3) RFE to select N_FEATURES
rfe_estimator = LogisticRegression(
    solver="liblinear", penalty="l1",
    max_iter=10000, random_state=RANDOM_STATE,
)
selector = RFE(estimator=rfe_estimator,
               n_features_to_select=N_FEATURES, step=1)
selector.fit(X_train_imputed, y_train)

selected_features_20 = X_train.columns[selector.support_].tolist()

print(f"[INFO] Selected {len(selected_features_20)} miRNAs:")
for i, mirna in enumerate(selected_features_20, start=1):
    print(f"  {i:02d}. {mirna}")

# 4) Build reduced datasets
X_train_20 = selector.transform(X_train_imputed)
X_test_20 = selector.transform(X_test_imputed)

# 5) Train final model (L2 penalty)
final_model_20 = LogisticRegression(
    solver="liblinear", penalty="l2",
    max_iter=10000, random_state=RANDOM_STATE,
)
final_model_20.fit(X_train_20, y_train)

# 6) Predict and evaluate
y_pred_20 = final_model_20.predict(X_test_20)
y_prob_20 = final_model_20.predict_proba(X_test_20)[:, 1]

accuracy_20 = accuracy_score(y_test, y_pred_20)
auc_20 = roc_auc_score(y_test, y_prob_20)
sensitivity_20 = recall_score(y_test, y_pred_20, pos_label=1)

cm_20 = confusion_matrix(y_test, y_pred_20)
tn, fp, fn, tp = cm_20.ravel()
specificity_20 = tn / (tn + fp)

print("\n--- Test Performance: 20-miRNA Panel ---")
print(f"Accuracy    : {accuracy_20:.3f}")
print(f"ROC-AUC     : {auc_20:.3f}")
print(f"Sensitivity : {sensitivity_20:.3f}")
print(f"Specificity : {specificity_20:.3f}")
print("\nConfusion Matrix:")
print(cm_20)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_20))

# 7) Coefficient table
coefficients_20 = final_model_20.coef_.ravel()
importance_20 = pd.DataFrame({
    "miRNA": selected_features_20,
    "Coefficient": coefficients_20,
})
importance_20["Absolute_Coefficient"] = (
    importance_20["Coefficient"].abs()
)
importance_20 = importance_20.sort_values(
    "Absolute_Coefficient", ascending=False
)

print("\nCoefficient table (sorted by |coef|):")
print(importance_20)

# 8) Save coefficients
importance_20.to_csv(
    "results/final_panel_coefficients.csv", index=False
)

# 9) Bar plot of coefficients
plt.figure(figsize=(10, 8))
colors = [
    "firebrick" if c < 0 else "steelblue"
    for c in importance_20["Coefficient"]
]
plt.barh(importance_20["miRNA"],
         importance_20["Coefficient"], color=colors)
plt.axvline(x=0, color="black", linewidth=1)
plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("miRNA")
plt.title(f"{N_FEATURES}-miRNA Diagnostic Panel")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    "figures/top_20_miRNA_panel_coefficients.png",
    dpi=300, bbox_inches="tight",
)
print("\n[INFO] Figure saved to figures/")
plt.show()
