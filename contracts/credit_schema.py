import pandera.pandas as pa
from pandera.typing import Series

# Allowed categories observed in your data (extend if your full CSV has more)
HOUSING = ["own", "rent", "free"]
SAVINGS = ["none", "little", "moderate", "quite rich", "rich"]
CHECKING = ["none", "little", "moderate", "rich"]
PURPOSE = [
    "radio/TV", "education", "furniture/equipment",
    "car", "business", "domestic appliances",
    "repairs", "vacation/others",
]
SEX = ["male", "female"]
TARGET = ["good", "bad"]

credit_schema = pa.DataFrameSchema(
    columns={
        # ---- identity / not required, but validated if present ----
        "age": pa.Column(
            int,
            checks=[pa.Check.ge(18), pa.Check.le(100)],
            nullable=False,
            coerce=True,
        ),
        "sex": pa.Column(
            str,
            checks=pa.Check.isin(SEX),
            nullable=False,
            coerce=True,
        ),
        "job": pa.Column(
            int,
            checks=pa.Check.isin([0, 1, 2, 3]),
            nullable=False,
            coerce=True,
        ),
        "housing": pa.Column(
            str,
            checks=pa.Check.isin(HOUSING),
            nullable=False,
            coerce=True,
        ),
        "saving_accounts": pa.Column(
            str,
            checks=pa.Check.isin(SAVINGS),
            nullable=False,
            coerce=True,
        ),
        "checking_account": pa.Column(
            str,
            checks=pa.Check.isin(CHECKING),
            nullable=False,
            coerce=True,
        ),
        "credit_amount": pa.Column(
            float,
            checks=[pa.Check.gt(0), pa.Check.le(100_000)],
            nullable=False,
            coerce=True,
        ),
        "duration": pa.Column(
            int,
            checks=[pa.Check.ge(1), pa.Check.le(120)],   # months
            nullable=False,
            coerce=True,
        ),
        "purpose": pa.Column(
            str,
            checks=pa.Check.isin(PURPOSE),
            nullable=False,
            coerce=True,
        ),
        # ---- optional: only validate if present (for training baseline) ----
        "target": pa.Column(
            str,
            checks=pa.Check.isin(TARGET),
            nullable=False,
            coerce=True,
            required=False,
        ),
    },
    checks=[
        # 1) no duplicate rows
        pa.Check(
            lambda df: ~df.duplicated().any(),
            error="Duplicatas encontradas no lote.",
        ),
        # 2) credit_amount must be plausible vs. monthly capacity proxy:
        #    duration cannot be zero and credit_amount/duration ratio bounded
        pa.Check(
            lambda df: (df["credit_amount"] / df["duration"]).le(5_000).all(),
            error="Relação crédito/duração acima do limite (possível outlier).",
        ),
        # 3) job level 0 (unskilled, non-resident) must be <= 3% of batch
        pa.Check(
            lambda df: (df["job"] == 0).mean() <= 0.03,
            error="Proporção anômala de job=0 no lote.",
        ),
    ],
    coerce=True,
    strict=False,   # ignore extra columns (e.g., id) if present
)