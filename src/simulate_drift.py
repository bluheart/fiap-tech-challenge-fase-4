"""
simulate_drift.py
-----------------
Creates a "production" dataset from the reference dataset by intentionally
shifting the distribution of several features to emulate real-world drift
over time.

Usage:
    python src/simulate_drift.py
Output:
    data/production/credit_production.csv
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

REFERENCE_PATH  = "data/reference/german_credit_data.csv"
PRODUCTION_PATH = "data/production/credit_production.csv"

SEED = 42
rng  = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Load reference
# ---------------------------------------------------------------------------
ref = pd.read_csv(REFERENCE_PATH)
ref = ref.loc[:, ~ref.columns.str.startswith("Unnamed")]

# Normalize the literal "NA" string once, so drift simulation works on
# real categories rather than the string "NA".
for col in ["saving_accounts", "checking_account"]:
    ref[col] = ref[col].astype(str).str.strip().replace({"NA": "none", "": "none"})

prod = ref.copy()

# ---------------------------------------------------------------------------
# 2. Numeric drift
# ---------------------------------------------------------------------------
# 2a. age: shift mean upward by ~10 years, keep inside [18, 100]
prod["age"] = np.clip(
    prod["age"] + rng.normal(loc=10, scale=4, size=len(prod)),
    18,
    100,
).round().astype(int)

# 2b. credit_amount: inflate by 40–80% (inflation + bigger loan requests)
prod["credit_amount"] = np.clip(
    prod["credit_amount"] * rng.uniform(1.4, 1.8, size=len(prod)),
    100,
    100_000,
)

# 2c. duration: lengthen by 30–60%
prod["duration"] = np.clip(
    prod["duration"] * rng.uniform(1.3, 1.6, size=len(prod)),
    1,
    120,
).round().astype(int)

# ---------------------------------------------------------------------------
# 3. Ordinal-categorical drift
#    Shift a fraction of rows down to the lowest tier, emulating
#    an economic downturn where customers deplete savings.
# ---------------------------------------------------------------------------
SAVINGS_ORDER  = ["none", "little", "moderate", "quite rich", "rich"]
CHECKING_ORDER = ["none", "little", "moderate", "rich"]

def downgrade_ordinal(series, order, frac=0.30):
    s = series.copy()
    mask = rng.random(len(s)) < frac
    s.loc[mask] = order[0]   # force "none"
    return s

prod["saving_accounts"]  = downgrade_ordinal(prod["saving_accounts"],  SAVINGS_ORDER,  frac=0.35)
prod["checking_account"] = downgrade_ordinal(prod["checking_account"], CHECKING_ORDER, frac=0.30)

# ---------------------------------------------------------------------------
# 4. Nominal-categorical drift
#    Reweight "purpose" so that "car" and "business" grow, "radio/TV" shrinks.
# ---------------------------------------------------------------------------
purpose_weights = {
    "radio/TV":             0.15,
    "education":            0.10,
    "furniture/equipment":  0.15,
    "car":                  0.35,
    "business":             0.15,
    "domestic appliances":  0.03,
    "repairs":              0.04,
    "vacation/others":      0.03,
}
# Only draw from purposes actually present in the reference
present = [p for p in purpose_weights if p in ref["purpose"].unique()]
w = np.array([purpose_weights[p] for p in present], dtype=float)
w /= w.sum()
prod["purpose"] = rng.choice(present, size=len(prod), p=w)

# Mild housing shift toward "rent" (affordability pressure)
housing_probs = {"own": 0.50, "rent": 0.40, "free": 0.10}
prod["housing"] = rng.choice(
    list(housing_probs.keys()),
    size=len(prod),
    p=list(housing_probs.values()),
)

# ---------------------------------------------------------------------------
# 5. Keep `job` and `sex` unchanged  (control variables)
#    -> They should NOT be flagged as drifted; good for the report narrative.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. Re-encode target as good/bad if present (don't drift the label!)
# ---------------------------------------------------------------------------
# We intentionally leave `target` untouched. Drift is a feature-level
# phenomenon; in a real scenario we may not even have labels in production.
# If you want to *also* show concept drift later, recompute target from
# a shifted rule; that's out of scope for Stage 2.

# ---------------------------------------------------------------------------
# 7. Sanity: enforce hard bounds so the contract still passes
# ---------------------------------------------------------------------------
prod["age"]           = prod["age"].clip(18, 100)
prod["credit_amount"] = prod["credit_amount"].clip(100, 100_000)
prod["duration"]      = prod["duration"].clip(1, 120)

# Ratio rule from the Pandera contract:
#   credit_amount / duration <= 5000
# After inflating both, some rows may violate it. Cap them.
ratio_violation = prod["credit_amount"] / prod["duration"] > 5_000
if ratio_violation.any():
    prod.loc[ratio_violation, "duration"] = np.ceil(
        prod.loc[ratio_violation, "credit_amount"] / 5_000
    ).astype(int)

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
import os
os.makedirs(os.path.dirname(PRODUCTION_PATH), exist_ok=True)
prod.to_csv(PRODUCTION_PATH, index=False)

print(f"Reference rows : {len(ref)}")
print(f"Production rows: {len(prod)}")
print(f"Saved          : {PRODUCTION_PATH}")
print("\n--- Drift summary (mean shift) ---")
for col in ["age", "credit_amount", "duration"]:
    print(f"{col:15s} ref={ref[col].mean():10.2f}  prod={prod[col].mean():10.2f}")

print("\n--- saving_accounts distribution ---")
print(pd.DataFrame({
    "ref":  ref["saving_accounts"].value_counts(normalize=True),
    "prod": prod["saving_accounts"].value_counts(normalize=True),
}).fillna(0).round(3))

print("\n--- purpose distribution ---")
print(pd.DataFrame({
    "ref":  ref["purpose"].value_counts(normalize=True),
    "prod": prod["purpose"].value_counts(normalize=True),
}).fillna(0).round(3))