import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_PATH = "data/reference/german_credit_data.csv"
MODEL_PATH = "models/baseline_credit_model.joblib"

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

if "target" not in df.columns:
    raise ValueError("Reference dataset must contain 'target' (good/bad).")

# ---------------------------------------------------------------------------
# 2. Column groups
# ---------------------------------------------------------------------------
NUM_COLS = ["age", "credit_amount", "duration"]

# Ordinal categoricals: order matters
SAVINGS_ORDER  = ["none", "little", "moderate", "quite rich", "rich"]
CHECKING_ORDER = ["none", "little", "moderate", "rich"]
JOB_ORDER      = [0, 1, 2, 3]   # already numeric, kept as ordinal for clarity

ORDINAL_COLS = ["saving_accounts", "checking_account", "job"]
ORDINAL_ORDERS = [SAVINGS_ORDER, CHECKING_ORDER, JOB_ORDER]

# Nominal categoricals: no intrinsic order
NOMINAL_COLS = ["sex", "housing", "purpose"]

# ---------------------------------------------------------------------------
# 3. Clean the literal "NA" string -> "none" for ordinal columns
#    (keeps ordinal encoder happy; "none" is the lowest level anyway)
# ---------------------------------------------------------------------------
for col in ["saving_accounts", "checking_account"]:
    df[col] = df[col].astype(str).str.strip().replace({"NA": "none", "": "none"})

for col in NOMINAL_COLS:
    df[col] = df[col].astype(str).str.strip()

for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------------
# 4. Encode target: good -> 0, bad -> 1
# ---------------------------------------------------------------------------
df["target"] = df["target"].astype(str).str.strip().map({"good": 0, "bad": 1})
if df["target"].isna().any():
    raise ValueError("Unexpected target values; expected 'good' or 'bad'.")

df = df.dropna(subset=NUM_COLS + ORDINAL_COLS + NOMINAL_COLS + ["target"])

# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
X = df[NUM_COLS + ORDINAL_COLS + NOMINAL_COLS]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 6. Preprocessing
# ---------------------------------------------------------------------------
numeric_pipe = Pipeline([("scaler", StandardScaler())])

# OrdinalEncoder: unknown values at inference -> NaN, then we can fall back.
# Better: use handle_unknown='use_encoded_value' with unknown_value=-1
ordinal_pipe = Pipeline([
    ("ordinal", OrdinalEncoder(
        categories=ORDINAL_ORDERS,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
    ("scaler", StandardScaler()),
])

nominal_pipe = Pipeline([
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num",     numeric_pipe,   NUM_COLS),
        ("ordinal", ordinal_pipe,   ORDINAL_COLS),
        ("nominal", nominal_pipe,   NOMINAL_COLS),
    ],
    remainder="drop",
)

# ---------------------------------------------------------------------------
# 7. Model
# ---------------------------------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )),
])

model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]
pred  = model.predict(X_test)

print(f"Test ROC-AUC : {roc_auc_score(y_test, proba):.4f}")
print(f"Test PR-AUC  : {average_precision_score(y_test, proba):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, digits=4))

# ---------------------------------------------------------------------------
# 9. Persist
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")