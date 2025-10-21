#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

def load_bundle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}. Rode antes: python src/train.py")
    bundle = joblib.load(model_path)
    return (
        bundle["model"],
        bundle["feature_columns"],
        bundle["species_to_int"],
        bundle["int_to_species"],
    )

def parse_values_arg(values_str: str):
    parts = [p.strip() for p in values_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Forneça exatamente 4 valores separados por vírgula.")
    return [float(x) for x in parts]

def main():
    parser = argparse.ArgumentParser(description="Predição Iris (GaussianNB).")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models" / "iris_nb.joblib"),
                        help="Caminho do arquivo .joblib do modelo.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--values", help='Quatro valores CSV na ordem salva pelo modelo. Ex: "5.1,3.5,1.4,0.2"')
    group.add_argument("--json", help='JSON com chaves exatamente iguais à ordem das features.')
    args = parser.parse_args()

    clf, feature_columns, species_to_int, int_to_species = load_bundle(Path(args.model))

    if args.values:
        arr = parse_values_arg(args.values)
    elif args.json:
        obj = json.loads(args.json)
        arr = [float(obj[c]) for c in feature_columns]
    else:
        print("Insira 4 valores (floats) nesta ordem:")
        print(", ".join(feature_columns))
        arr = [float(input(f"{c}: ").strip()) for c in feature_columns]

    X = np.array(arr, dtype=float).reshape(1, -1)
    y_pred = int(clf.predict(X)[0])
    result = {
        "input_order": feature_columns,
        "input_values": arr,
        "pred_label": y_pred,
        "pred_species": int_to_species[y_pred],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
