#!/usr/bin/env python3
# Predição Iris com UX melhorada (tabelas, cores, validação e probabilidades)
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.align import Align

console = Console()

def load_bundle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}. Rode antes: python src/train.py")
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feature_columns = bundle["feature_columns"]
    species_to_int = bundle["species_to_int"]
    int_to_species = bundle["int_to_species"]
    return clf, feature_columns, species_to_int, int_to_species

def ask_float(label: str) -> float:
    while True:
        raw = Prompt.ask(f"[bold]{label}[/] (ex: 5.1)")
        try:
            return float(str(raw).strip().replace(",", "."))
        except ValueError:
            console.print("[red]Valor inválido. Digite um número (use ponto ou vírgula).[/]")

def parse_values_arg(values_str: str):
    parts = [p.strip() for p in values_str.split(",")]
    if len(parts) != 4:
        raise ValueError("Forneça exatamente 4 valores separados por vírgula.")
    vals = []
    for x in parts:
        vals.append(float(str(x).replace(",", ".")))
    return vals

def parse_json_arg(json_str: str, feature_columns):
    obj = json.loads(json_str)
    vals = []
    for col in feature_columns:
        # aceita chaves iguais (recomendado); se quiser, pode acrescentar aliases aqui
        if col not in obj:
            raise KeyError(f"Chave ausente no JSON para '{col}'")
        vals.append(float(str(obj[col]).replace(",", ".")))
    return vals

def predict_pretty(arr, feature_columns, clf, int_to_species, show_probs=True):
    X = np.array(arr, dtype=float).reshape(1, -1)
    y_pred = int(clf.predict(X)[0])
    species = int_to_species[y_pred]

    # Painel com o resultado principal
    result_panel = Panel(
        Align.center(
            f"[bold white]Predição:[/]\n\n[bold green]{species.upper()}[/]  (label = {y_pred})",
            vertical="middle",
        ),
        title="[bold]Iris — Resultado[/]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(result_panel)

    # Tabela com os inputs
    t_inputs = Table(title="Entrada do usuário (na ordem do modelo)", show_lines=True)
    t_inputs.add_column("Feature", style="cyan", no_wrap=True)
    t_inputs.add_column("Valor", style="white")
    for c, v in zip(feature_columns, arr):
        t_inputs.add_row(c, str(v))
    console.print(t_inputs)

    # Probabilidades por classe (GaussianNB tem predict_proba)
    if show_probs and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]  # array na ordem interna das classes
        # Precisamos descobrir a ordem das classes no modelo
        # Em nosso treinamento, as classes são [1,2,3]; vamos garantir isso:
        classes = getattr(clf, "classes_", None)
        if classes is None:
            classes = np.array(sorted(int_to_species.keys()))
        t_probs = Table(title="Probabilidade por classe", show_lines=True)
        t_probs.add_column("Classe", style="magenta")
        t_probs.add_column("Espécie", style="magenta")
        t_probs.add_column("Probabilidade", style="magenta")
        for cls, p in zip(classes, proba):
            name = int_to_species[int(cls)]
            t_probs.add_row(str(int(cls)), name, f"{p:.4f}")
        console.print(t_probs)

    return {"pred_label": y_pred, "pred_species": species}

def main():
    parser = argparse.ArgumentParser(description="Predição Iris (UX aprimorada com rich).")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models" / "iris_nb.joblib"),
                        help="Caminho do arquivo .joblib do modelo.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--values", help='Quatro valores CSV na ordem do modelo. Ex: "5.1,3.5,1.4,0.2"')
    group.add_argument("--json", help="JSON com chaves exatamente iguais à ordem das features.")
    parser.add_argument("--no-probs", action="store_true", help="Não mostrar tabela de probabilidades.")
    args = parser.parse_args()

    console.print(Panel("[bold]Iris Classifier — Interface CLI Amigável[/]\n"
                        "Preencha 4 valores ou use --values / --json.\n",
                        border_style="blue"))

    clf, feature_columns, species_to_int, int_to_species = load_bundle(Path(args.model))

    # Se nada passado, modo interativo
    if not args.values and not args.json:
        console.print("[bold]Ordem das features:[/] " + ", ".join(feature_columns))
        vals = [ask_float(c) for c in feature_columns]
    elif args.values:
        vals = parse_values_arg(args.values)
    else:
        vals = parse_json_arg(args.json, feature_columns)

    result = predict_pretty(vals, feature_columns, clf, int_to_species, show_probs=not args.no_probs)

    # Também imprime um JSON final (útil para testes automatizados)
    console.print(Panel.fit(json.dumps({
        "input_order": feature_columns,
        "input_values": vals,
        **result
    }, ensure_ascii=False, indent=2), title="Saída JSON", border_style="cyan"))

if __name__ == "__main__":
    main()
