import pandas as pd
import sys
from pathlib import Path
import pandera.pandas as pa

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from contracts.credit_schema import credit_schema


df = pd.read_csv("data/reference/german_credit_data.csv")
credit_schema.validate(df)   # should pass

# Force a failure: duplicate + out-of-range age + huge credit_amount
bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
bad.loc[0, "age"] = 150
bad.loc[1, "credit_amount"] = 999_999

try:
    credit_schema.validate(bad, lazy=True)
except pa.errors.SchemaErrors as e:
    print("BLOQUEADO:")
    print(e.failure_cases[["column", "check", "failure_case"]])