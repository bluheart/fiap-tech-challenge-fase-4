"""
detect_drift.py
---------------
Detecção de drift usando Evidently AI como biblioteca central.

Gera:
  1. reports/drift_report.html          -> relatório visual completo
  2. reports/metrics/drift_metrics.json -> métricas extraídas do Evidently

Uso:
    python src/detect_drift.py
"""

import json
import os
import warnings
import sys
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

REFERENCE_PATH  = "data/reference/german_credit_data.csv"
PRODUCTION_PATH = "data/production/credit_production.csv"

REPORT_HTML  = "reports/drift_report.html"
METRICS_JSON = "reports/metrics/drift_metrics.json"

# ---------------------------------------------------------------------------
# 1. Carregar e normalizar
# ---------------------------------------------------------------------------
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    for col in ["saving_accounts", "checking_account"]:
        df[col] = df[col].astype(str).str.strip().replace({"NA": "none", "": "none"})
    for col in ["sex", "housing", "purpose"]:
        df[col] = df[col].astype(str).str.strip()

    return df

ref  = load(REFERENCE_PATH)
prod = load(PRODUCTION_PATH)

# Remover target da comparação (não é feature; evita label leak)
FEATURES = [
    "age", "credit_amount", "duration",
    "saving_accounts", "checking_account", "job",
    "sex", "housing", "purpose",
]
ref_eval  = ref[FEATURES].copy()
prod_eval = prod[FEATURES].copy()

# ---------------------------------------------------------------------------
# 2. Executar Evidently DataDriftPreset
# ---------------------------------------------------------------------------
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_eval, current_data=prod_eval)

# ---------------------------------------------------------------------------
# 3. Salvar HTML
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(REPORT_HTML), exist_ok=True)
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=ref_eval, current_data=prod_eval)
snapshot.save_html(REPORT_HTML)

# ---------------------------------------------------------------------------
# 4. Extrair resultados do Evidently (estrutura v2/0.7.x) para JSON
# ---------------------------------------------------------------------------
result_dict = snapshot.dict()   # já é dict, não precisa json.loads

metrics_list = result_dict.get("metrics", [])

feature_results = []
summary = {
    "dataset_drift": False,
    "number_of_drifted_columns": 0,
    "share_of_drifted_columns": 0.0,
    "drift_share_threshold": None,
}

for m in metrics_list:
    name  = m.get("metric_name", "")
    cfg   = m.get("config", {})
    value = m.get("value", None)

    # --- (a) métrica de contagem de colunas com drift ---
    if name.startswith("DriftedColumnsCount"):
        summary["dataset_drift"]          = bool(value["share"] >= cfg.get("drift_share", 0.5))
        summary["number_of_drifted_columns"] = int(value["count"])
        summary["share_of_drifted_columns"]  = round(float(value["share"]), 4)
        summary["drift_share_threshold"]     = cfg.get("drift_share", 0.5)
        continue

    # --- (b) métrica de drift por coluna ---
    if name.startswith("ValueDrift"):
        column   = cfg.get("column")
        method   = cfg.get("method", "")
        threshold = cfg.get("threshold", 0.05)

        # Todos os métodos neste output são p-valor → drift se p < threshold
        p_value = float(value)
        drift_detected = p_value < threshold

        feature_results.append({
            "feature":        column,
            "method":         method,
            "drift_score":    round(p_value, 8),   # p-valor
            "threshold":      threshold,
            "drift_detected": bool(drift_detected),
        })

# Ordena: drifted primeiro, depois por nome
feature_results.sort(key=lambda r: (not r["drift_detected"], r["feature"]))
features_drifted = [r["feature"] for r in feature_results if r["drift_detected"]]

# Sanity check: se não encontrou nada, falha com mensagem útil
if not feature_results and not summary["number_of_drifted_columns"]:
    raise RuntimeError(
        "Estrutura inesperada do Evidently. "
        f"metric_names encontrados: {[m.get('metric_name') for m in metrics_list]}"
    )

# ---------------------------------------------------------------------------
# 5. Montar JSON final (compatível com o dashboard)
# ---------------------------------------------------------------------------
metrics = {
    "pipeline_status":              "success",
    "reference_rows":               int(len(ref_eval)),
    "production_rows":              int(len(prod_eval)),
    "dataset_drift":                summary["dataset_drift"],
    "n_features_total":             len(feature_results),
    "n_features_drifted":           summary["number_of_drifted_columns"],
    "share_drifted":                summary["share_of_drifted_columns"],
    "drift_share_threshold":        summary["drift_share_threshold"],
    "features_drifted":             features_drifted,
    "feature_results":              feature_results,
    "evidently_report":             REPORT_HTML,
}

os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 6. Resumo no console
# ---------------------------------------------------------------------------
print("=" * 78)
print(f"{'FEATURE':<22}{'METHOD':<22}{'P-VALUE':>12}{'DRIFT':>8}")
print("=" * 78)
for r in feature_results:
    print(
        f"{r['feature']:<22}{r['method']:<22}"
        f"{r['drift_score']:>12.3e}"
        f"{'  YES' if r['drift_detected'] else '   no':>8}"
    )
print("=" * 78)
print(f"Dataset drift        : {'YES' if metrics['dataset_drift'] else 'no'}")
print(f"Columns drifted      : {metrics['n_features_drifted']} / {metrics['n_features_total']}")
print(f"Share drifted        : {metrics['share_drifted']:.2%}")
print(f"Features com drift   : {features_drifted}")
print(f"HTML report          : {REPORT_HTML}")
print(f"JSON metrics         : {METRICS_JSON}")