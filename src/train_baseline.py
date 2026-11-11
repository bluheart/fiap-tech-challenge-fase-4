import json
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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

DATA_PATH  = "data/reference/german_credit_data.csv"
MODEL_PATH = "models/baseline_credit_model.joblib"
LOG_PATH   = "logs/pipeline.jsonl"

MLFLOW_EXPERIMENT = "credit-risk-baseline"
MLFLOW_RUN_NAME   = "logreg-baseline"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# ---------------------------------------------------------------------------
# Helper de log no mesmo formato usado em validate_data.py
# ---------------------------------------------------------------------------
def log_event(step: str, status: str, **extra) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {"step": step, "status": status, **extra}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
t0 = time.perf_counter()

try:
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if "target" not in df.columns:
        raise ValueError("Reference dataset must contain 'target' (good/bad).")

    log_event("train", "started", file=DATA_PATH, rows=len(df))

    # -----------------------------------------------------------------------
    # 2. Column groups
    # -----------------------------------------------------------------------
    NUM_COLS = ["age", "credit_amount", "duration"]

    SAVINGS_ORDER  = ["none", "little", "moderate", "quite rich", "rich"]
    CHECKING_ORDER = ["none", "little", "moderate", "rich"]
    JOB_ORDER      = [0, 1, 2, 3]

    ORDINAL_COLS   = ["saving_accounts", "checking_account", "job"]
    ORDINAL_ORDERS = [SAVINGS_ORDER, CHECKING_ORDER, JOB_ORDER]
    NOMINAL_COLS   = ["sex", "housing", "purpose"]

    # -----------------------------------------------------------------------
    # 3. Cleaning
    # -----------------------------------------------------------------------
    for col in ["saving_accounts", "checking_account"]:
        df[col] = df[col].astype(str).str.strip().replace({"NA": "none", "": "none"})

    for col in NOMINAL_COLS:
        df[col] = df[col].astype(str).str.strip()

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------------------------
    # 4. Encode target
    # -----------------------------------------------------------------------
    df["target"] = df["target"].astype(str).str.strip().map({"good": 0, "bad": 1})
    if df["target"].isna().any():
        raise ValueError("Unexpected target values; expected 'good' or 'bad'.")

    df = df.dropna(subset=NUM_COLS + ORDINAL_COLS + NOMINAL_COLS + ["target"])

    # -----------------------------------------------------------------------
    # 5. Split
    # -----------------------------------------------------------------------
    X = df[NUM_COLS + ORDINAL_COLS + NOMINAL_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------------
    # 6. Preprocessing
    # -----------------------------------------------------------------------
    numeric_pipe = Pipeline([("scaler", StandardScaler())])

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

    # -----------------------------------------------------------------------
    # 7. Model
    # -----------------------------------------------------------------------
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    # =======================================================================
    # 7.1 Treino com MLflow (rastreamento do experimento)
    # =======================================================================
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:

        # ---- Parâmetros do modelo (o que define o experimento) ----
        mlflow.log_params({
            "model_type":     "LogisticRegression",
            "solver":         "lbfgs",
            "max_iter":       2000,
            "class_weight":   "balanced",
            "random_state":   42,
            "test_size":      0.25,
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_features":     X.shape[1],
        })

        # ---- Treino ----
        model.fit(X_train, y_train)

        # ---- Avaliação ----
        proba = model.predict_proba(X_test)[:, 1]
        pred  = model.predict(X_test)

        auc    = float(roc_auc_score(y_test, proba))
        pr_auc = float(average_precision_score(y_test, proba))

        print(f"Test ROC-AUC : {auc:.4f}")
        print(f"Test PR-AUC  : {pr_auc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, pred, digits=4))

        # ---- Métricas no MLflow ----
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("pr_auc",  pr_auc)

        # ---- Persistência local + MLflow ----
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"\nModel saved to: {MODEL_PATH}")

        # Registra o modelo como artefato do run
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="credit-risk-baseline",  # registry opcional
        )

        # Anexa o joblib também (útil para o script de drift reusar)
        mlflow.log_artifact(MODEL_PATH, artifact_path="joblib")

        # ---- Log de sucesso no JSONL (com run_id do MLflow) ----
        log_event(
            "train",
            "success",
            file=DATA_PATH,
            rows=len(df),
            duration_s=round(time.perf_counter() - t0, 3),
            auc=round(auc, 4),
            pr_auc=round(pr_auc, 4),
            model_path=MODEL_PATH,
            mlflow_run_id=run.info.run_id,
            mlflow_experiment=MLFLOW_EXPERIMENT,
            errors=[],
        )

except Exception as exc:
    log_event(
        "train",
        "failed",
        file=DATA_PATH,
        duration_s=round(time.perf_counter() - t0, 3),
        errors=[f"{type(exc).__name__}: {exc}"],
    )
    raise