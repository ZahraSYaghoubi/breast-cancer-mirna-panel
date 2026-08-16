"""
Step 3: Missing-value filtering + LASSO feature selection
---------------------------------------------------------
Filters out miRNAs present in fewer than 70% of samples,
splits data into train/test (stratified), and trains a
LASSO (L1) logistic regression using nested CV on train.

Input : results/data_ml.csv
Output: prints selected miRNAs
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

# ===================== CONFIG =====================
MISSING_THRESHOLD = 0.70   # keep miRNA present in >=70% of samples
TEST_SIZE = 0.30
RANDOM_STATE = 42
# ==================================================

# 1) Load the ML table
data_ml = pd.read_csv("results/data_ml.csv", sep="\t", index_col=0)

X = data_ml.drop(columns="Target")
y = data_ml["Target"]

print(f"[INFO] X shape: {X.shape}")
print("[INFO] Class distribution:")
print(y.value_counts())

# 2) Drop columns (miRNAs) with too much missingness
min_present = int(np.ceil(MISSING_THRESHOLD * len(X)))
X_filtered = X.dropna(axis=1, thresh=min_present)

print(f"\n[INFO] miRNAs before filter: {X.shape[1]}")
print(f"[INFO] miRNAs after filter : {X_filtered.shape[1]}")
print("\n[INFO] Missingness after filtering:")
print(X_filtered.isna().mean().describe())

# 3) Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\n[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Test shape : {X_test.shape}")
print("[INFO] Train class distribution:")
print(y_train.value_counts())
print("[INFO] Test class distribution:")
print(y_test.value_counts())

# 4) Nested cross-validation (inner CV only on train)
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

lasso_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "lasso_logistic",
        LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            Cs=np.logspace(-3, 2, 20),
            cv=inner_cv,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=10000,
            random_state=RANDOM_STATE,
            refit=True,
        ),
    ),
])

# 5) Fit on train only
lasso_model.fit(X_train, y_train)
print("\n[INFO] LASSO training completed.")

# 6) Extract selected (nonzero-coefficient) miRNAs
trained_lasso = lasso_model.named_steps["lasso_logistic"]
coefficients = trained_lasso.coef_.ravel()

selected_mirnas = X_train.columns[coefficients != 0]
selected_table = pd.DataFrame({
    "miRNA": selected_mirnas,
    "Coefficient": coefficients[coefficients != 0],
    "Abs_Coefficient": np.abs(coefficients[coefficients != 0]),
}).sort_values("Abs_Coefficient", ascending=False)

print(f"\n[INFO] Best C (internal CV): {trained_lasso.C_[0]:.4f}")
print(f"[INFO] Number of selected miRNAs: {len(selected_table)}")

print("\n[INFO] Selected miRNAs:")
print(selected_table)

# 7) Save for later steps
selected_table.to_csv("results/lasso_selected_mirnas.csv", index=False)
print("\n[INFO] Saved to results/lasso_selected_mirnas.csv")
