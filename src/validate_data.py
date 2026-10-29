import argparse
import json
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from contracts.credit_schema import credit_schema


def validate(path: str) -> int:
    df = pd.read_csv(path)

    try:
        credit_schema.validate(df, lazy=True)
        status = "success"
        exit_code = 0
        errors = []
    except Exception as exc:
        status = "failed"
        exit_code = 1
        errors = [str(exc)]

    with open("logs/pipeline.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step": "data_validation",
            "file": path,
            "status": status,
            "rows": len(df),
            "errors": errors,
        }, ensure_ascii=False) + "\n")

    if exit_code != 0:
        print("Contrato de dados FALHOU. Ingestão bloqueada.")
        print(errors[0])
        sys.exit(exit_code)

    print("Contrato de dados validado com sucesso.")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    validate(args.path)