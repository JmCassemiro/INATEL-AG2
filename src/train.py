import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import (
    SPECIES_TO_INT,
    INT_TO_SPECIES,
    normalize_species_name,
    resolve_feature_columns,
)


def main():
    parser = argparse.ArgumentParser(
        description="Treino Iris com GaussianNB (80/20, shuffle=True) usando Series.replace."
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "iris.csv"),
        help="Caminho do iris.csv (padrão: data/iris.csv).",
    )
    parser.add_argument(
        "--model-out",
        default=str(Path(__file__).resolve().parents[1] / "models" / "iris_nb.joblib"),
        help="Arquivo .joblib de saída.",
    )
    parser.add_argument(
        "--mapping-out",
        default=str(
            Path(__file__).resolve().parents[1] / "models" / "species_mapping.json"
        ),
        help="Arquivo JSON de mapeamentos.",
    )
    parser.add_argument(
        "--metrics-out",
        default=str(Path(__file__).resolve().parents[1] / "models" / "metrics.json"),
        help="Arquivo JSON de métricas.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if "species" not in df.columns:
        for c in df.columns:
            if c.lower().strip() in ("species", "target", "class", "variety", "label"):
                df = df.rename(columns={c: "species"})
                break
    if "species" not in df.columns:
        raise KeyError("Coluna 'species' não encontrada no CSV.")

    df["species"] = df["species"].map(normalize_species_name)
    bad = set(df["species"].unique()) - set(SPECIES_TO_INT.keys())
    if bad:
        raise ValueError(f"Valores inesperados em 'species': {sorted(bad)}")

    df["species"] = (
        df["species"].replace(SPECIES_TO_INT).infer_objects(copy=False).astype("int64")
    )

    cols = resolve_feature_columns(df)
    X = df[cols].to_numpy(dtype=float)
    y = df["species"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    labels_sorted = sorted(INT_TO_SPECIES.keys())  # [1,2,3]
    target_names = [INT_TO_SPECIES[i] for i in labels_sorted]
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred, labels=labels_sorted, target_names=target_names
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

   
    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_columns": cols,
            "species_to_int": SPECIES_TO_INT,
            "int_to_species": INT_TO_SPECIES,
        },
        out_model,
    )

    mapping_json = {
        "species_to_int": SPECIES_TO_INT,
        "int_to_species": INT_TO_SPECIES,
        "feature_columns": cols,
        "csv_source": str(csv_path),
        "label_column": "species",
        "label_mapping_method": "pandas.Series.replace",
    }
    Path(args.mapping_out).write_text(
        json.dumps(mapping_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    
    metrics = {
        "accuracy": acc,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "labels": labels_sorted,
        "target_names": target_names,
        "feature_columns": cols,
        "test_size": 0.2,
        "shuffle": True,
        "random_state": 42,
        "model": "GaussianNB",
        "label_mapping_method": "pandas.Series.replace",
    }
    Path(args.metrics_out).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nModelo salvo em: {out_model}")
    print(f"Mapeamentos salvos em: {args.mapping_out}")
    print(f"Métricas salvas em: {args.metrics_out}")


if __name__ == "__main__":
    main()
