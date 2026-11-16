"""
augment_dataset.py
------------------
Aumenta o dataset de referência de 1000 para 5000 registros
via replicação com ruído controlado, preservando:
  - os limites do contrato Pandera (age 18-100, duration 1-120,
    credit_amount <= 100000, credit_amount/duration <= 5000)
  - as categorias válidas (ordinais e nominais)
  - a proporção original da variável target
  - ausência de duplicatas exatas

Uso:
    python src/augment_dataset.py
Saída:
    data/reference/german_credit_data_5000.csv
"""

import numpy as np
import pandas as pd

INPUT_PATH  = "data/reference/german_credit_data.csv"
OUTPUT_PATH = "data/reference/german_credit_data_5000.csv"
TARGET_ROWS = 5000
SEED        = 42

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Carregar original
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Normaliza "NA" -> "none" nas colunas ordinais (consistente com o pipeline)
for col in ["saving_accounts", "checking_account"]:
    df[col] = df[col].astype(str).str.strip().replace({"NA": "none", "": "none"})

n_original = len(df)
n_needed   = TARGET_ROWS - n_original

print(f"Original: {n_original} linhas")
print(f"Necessário gerar: {n_needed} linhas sintéticas")

# ---------------------------------------------------------------------------
# 2. Amostragem estratificada por target (preserva a proporção good/bad)
# ---------------------------------------------------------------------------
# Sorteia linhas do dataset original com reposição, estratificando por target.
# Isso preserva a proporção good/bad na parte sintética.
synthetic = (
    df.groupby("target", group_keys=False)
      .apply(lambda g: g.sample(
          n=int(round(n_needed * len(g) / n_original)),
          replace=True,
          random_state=SEED,
      ))
      .reset_index(drop=True)
)

# Ajuste fino: se o arredondamento deixou a mais/menos, corta ou completa
if len(synthetic) > n_needed:
    synthetic = synthetic.sample(n=n_needed, random_state=SEED).reset_index(drop=True)
elif len(synthetic) < n_needed:
    extra = df.sample(n=n_needed - len(synthetic), replace=True, random_state=SEED)
    synthetic = pd.concat([synthetic, extra], ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Adicionar ruído controlado nas colunas numéricas
# ---------------------------------------------------------------------------
# Idade: ruído gaussiano de ±2 anos, clipado em [18, 100]
synthetic["age"] = np.clip(
    synthetic["age"] + rng.normal(0, 2, len(synthetic)),
    18, 100,
).round().astype(int)

# credit_amount: ruído multiplicativo de ±10%, clipado em [100, 100000]
synthetic["credit_amount"] = np.clip(
    synthetic["credit_amount"] * rng.uniform(0.9, 1.1, len(synthetic)),
    100, 100_000,
)

# duration: ruído de ±2 meses, clipado em [1, 120]
synthetic["duration"] = np.clip(
    synthetic["duration"] + rng.integers(-2, 3, len(synthetic)),
    1, 120,
).astype(int)

# ---------------------------------------------------------------------------
# 4. Garantir a regra credit_amount / duration <= 5000
# ---------------------------------------------------------------------------
ratio_violation = synthetic["credit_amount"] / synthetic["duration"] > 5_000
if ratio_violation.any():
    synthetic.loc[ratio_violation, "duration"] = np.ceil(
        synthetic.loc[ratio_violation, "credit_amount"] / 5_000
    ).astype(int)

# ---------------------------------------------------------------------------
# 5. Garantir ausência de duplicatas exatas
# ---------------------------------------------------------------------------
combined = pd.concat([df, synthetic], ignore_index=True)

# Se ainda houver duplicatas (raro, mas possível após ruído), remove
# mantendo a primeira ocorrência e reamostra o restante.
dupes = combined.duplicated()
if dupes.any():
    print(f"Removendo {dupes.sum()} duplicatas exatas...")
    combined = combined[~dupes].reset_index(drop=True)

# Se após remoção ficou abaixo de 5000, completa com novas amostras com ruído
while len(combined) < TARGET_ROWS:
    extra_n = TARGET_ROWS - len(combined)
    extra = df.sample(n=extra_n, replace=True, random_state=rng.integers(0, 10_000))
    extra["age"] = np.clip(
        extra["age"] + rng.normal(0, 2, len(extra)), 18, 100
    ).round().astype(int)
    extra["credit_amount"] = np.clip(
        extra["credit_amount"] * rng.uniform(0.9, 1.1, len(extra)), 100, 100_000
    )
    extra["duration"] = np.clip(
        extra["duration"] + rng.integers(-2, 3, len(extra)), 1, 120
    ).astype(int)
    ratio_v = extra["credit_amount"] / extra["duration"] > 5_000
    if ratio_v.any():
        extra.loc[ratio_v, "duration"] = np.ceil(
            extra.loc[ratio_v, "credit_amount"] / 5_000
        ).astype(int)
    combined = pd.concat([combined, extra], ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)

# Se ultrapassou 5000, corta aleatoriamente
if len(combined) > TARGET_ROWS:
    combined = combined.sample(n=TARGET_ROWS, random_state=SEED).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Salvar
# ---------------------------------------------------------------------------
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)

# ---------------------------------------------------------------------------
# 7. Resumo
# ---------------------------------------------------------------------------
print(f"\nSalvo em: {OUTPUT_PATH}")
print(f"Total de linhas: {len(combined)}")
print(f"\nDistribuição target (original vs aumentado):")
print("Original:")
print(df["target"].value_counts(normalize=True).round(3))
print("Aumentado:")
print(combined["target"].value_counts(normalize=True).round(3))

# Verificações de contrato
print("\nVerificações de contrato:")
print(f"  age min/max:           {combined['age'].min()} / {combined['age'].max()}")
print(f"  credit_amount min/max: {combined['credit_amount'].min():.0f} / {combined['credit_amount'].max():.0f}")
print(f"  duration min/max:      {combined['duration'].min()} / {combined['duration'].max()}")
print(f"  ratio max:             {(combined['credit_amount'] / combined['duration']).max():.2f} (limite 5000)")
print(f"  duplicatas exatas:     {combined.duplicated().sum()}")