#!/usr/bin/env python3
# Relatório de métricas em modo bonito (lendo models/metrics.json)
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def main():
    metrics_path = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {metrics_path}\nRode: python src/train.py")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Painel principal
    top = Panel(f"[bold]Modelo:[/] {data.get('model')}\n"
                f"[bold]Accuracy:[/] {data.get('accuracy'):.4f}\n"
                f"[bold]Split:[/] test_size={data.get('test_size')}  shuffle={data.get('shuffle')}  random_state={data.get('random_state')}\n"
                f"[bold]Features:[/] {', '.join(data.get('feature_columns', []))}",
                title="Resumo", border_style="green")
    console.print(top)

    # Classification report (macro view)
    cr = data.get("classification_report", {})
    # Tabela por classe
    per_class = Table(title="Relatório por classe", show_lines=True)
    per_class.add_column("Classe", style="magenta")
    per_class.add_column("precision", justify="right")
    per_class.add_column("recall", justify="right")
    per_class.add_column("f1-score", justify="right")
    per_class.add_column("support", justify="right")
    for name in data.get("target_names", []):
        row = cr.get(name, {})
        per_class.add_row(name,
                          f"{row.get('precision', 0):.4f}",
                          f"{row.get('recall', 0):.4f}",
                          f"{row.get('f1-score', 0):.4f}",
                          str(row.get("support", 0)))
    console.print(per_class)

    # Averages
    for avg_key in ["macro avg", "weighted avg"]:
        row = cr.get(avg_key, None)
        if row:
            avg_table = Table(title=f"{avg_key}", show_lines=True)
            avg_table.add_column("precision", justify="right")
            avg_table.add_column("recall", justify="right")
            avg_table.add_column("f1-score", justify="right")
            avg_table.add_row(f"{row.get('precision',0):.4f}",
                              f"{row.get('recall',0):.4f}",
                              f"{row.get('f1-score',0):.4f}")
            console.print(avg_table)

    # Confusion matrix
    cm = data.get("confusion_matrix", [])
    labels = data.get("target_names", [])
    if cm and labels:
        cm_table = Table(title="Matriz de Confusão (rows=true, cols=pred)", show_lines=True)
        cm_table.add_column("")
        for lbl in labels:
            cm_table.add_column(lbl, justify="right")
        for i, row in enumerate(cm):
            cm_table.add_row(labels[i], *[str(x) for x in row])
        console.print(cm_table)

if __name__ == "__main__":
    main()
