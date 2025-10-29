from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Iris Classifier — GaussianNB",
    page_icon="🌸",
    layout="centered",
)

# -----------------------------
# Helpers
# -----------------------------
def load_bundle(model_path: Path):
    if not model_path.exists():
        st.error(f"Modelo não encontrado: {model_path}\nRode antes: `python src/train.py`.")
        st.stop()
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feature_columns = bundle["feature_columns"]         # ordem das colunas usada no treino
    species_to_int = bundle["species_to_int"]
    int_to_species = bundle["int_to_species"]
    return clf, feature_columns, species_to_int, int_to_species

def load_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def base_feature_key(name: str) -> str:
    """Normaliza o nome salvo no modelo para uma chave base (sem _cm etc)."""
    name = name.replace(".", "_").replace(" ", "_").lower()
    if name.endswith("_cm"):
        name = name[:-3]
    return name

def parse_float(text: str):
    """Aceita 5.1 ou 5,1; retorna (ok, valor|msg_erro)."""
    try:
        val = float(str(text).strip().replace(",", "."))
        return True, val
    except Exception:
        return False, "Digite um número válido (use ponto ou vírgula)."

# -----------------------------
# Carregamento do modelo e métricas
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_nb.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

clf, feature_columns, species_to_int, int_to_species = load_bundle(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

# Classes na ordem interna do modelo (para predict_proba)
classes_in_model = getattr(clf, "classes_", np.array(sorted(int_to_species.keys())))
class_names = [int_to_species[int(c)] for c in classes_in_model]

# -----------------------------
# Sidebar (informações)
# -----------------------------
with st.sidebar:
    st.header("ℹ️ Sobre o modelo")
    st.write("**Algoritmo:** Gaussian Naive Bayes")
    if metrics:
        st.metric(label="Accuracy (test)", value=f"{metrics.get('accuracy', 0):.4f}")
        st.caption(
            f"Split: test_size={metrics.get('test_size')} | shuffle={metrics.get('shuffle')} | "
            f"random_state={metrics.get('random_state')}"
        )
    st.write("**Ordem das features no modelo:**")
    st.code(", ".join(feature_columns), language="text")
    st.write("**Mapa de rótulos:**")
    st.code(json.dumps(species_to_int, indent=2, ensure_ascii=False), language="json")
    st.write("**Caminhos:**")
    st.code(str(MODEL_PATH), language="text")
    st.code(str(METRICS_PATH), language="text")

st.title("🌸 Iris Classifier")
st.caption("Interface web para predição — GaussianNB (Iris)")

# -----------------------------
# Entradas por DIGITAÇÃO (sem sliders)
# Defaults "confortáveis" próximos a setosa
# -----------------------------
defaults = {
    "sepal_length": "5.1",
    "sepal_width":  "3.5",
    "petal_length": "1.4",
    "petal_width":  "0.2",
}

with st.form("predict_form", clear_on_submit=False):
    st.write("**Digite os valores (aceita vírgula ou ponto):**")
    col1, col2 = st.columns(2)
    with col1:
        sepal_length_txt = st.text_input("Sepal Length (cm)", value=defaults["sepal_length"])
        petal_length_txt = st.text_input("Petal Length (cm)", value=defaults["petal_length"])
    with col2:
        sepal_width_txt  = st.text_input("Sepal Width (cm)",  value=defaults["sepal_width"])
        petal_width_txt  = st.text_input("Petal Width (cm)",  value=defaults["petal_width"])

    submitted = st.form_submit_button("🔮 Prever (Enter)")

# Processamento da submissão
if submitted:
    fields = {
        "sepal_length": sepal_length_txt,
        "sepal_width":  sepal_width_txt,
        "petal_length": petal_length_txt,
        "petal_width":  petal_width_txt,
    }

    ui_values = {}
    errors = []
    for k, txt in fields.items():
        ok, val = parse_float(txt)
        if not ok:
            errors.append(f"**{k.replace('_',' ').title()}**: {val}")
        else:
            ui_values[k] = float(val)

    if errors:
        st.error("Erros na entrada:\n\n- " + "\n- ".join(errors))
        st.stop()

    # Reordena segundo a ordem EXATA salva no modelo
    ordered_vals = []
    for col in feature_columns:
        base_key = base_feature_key(col)   # ex: sepal_length_cm -> sepal_length
        if base_key not in ui_values:
            st.error(f"Entrada ausente para a feature '{col}'.")
            st.stop()
        ordered_vals.append(ui_values[base_key])

    # Predição
    X = np.array(ordered_vals, dtype=float).reshape(1, -1)
    y_pred = int(clf.predict(X)[0])
    species = int_to_species[y_pred]

    st.success(f"**Predição:** {species.upper()}  —  (label = {y_pred})")

    # Tabela com as entradas (na ordem do modelo)
    df_inputs = pd.DataFrame([ordered_vals], columns=feature_columns)
    with st.expander("Ver entradas usadas na predição"):
        st.dataframe(df_inputs, use_container_width=True)

    # Probabilidades por classe (se disponível)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        df_prob = pd.DataFrame({"classe": classes_in_model, "espécie": class_names, "probabilidade": proba})
        df_prob = df_prob.set_index("espécie")[["probabilidade"]]
        st.subheader("Probabilidade por classe")
        st.bar_chart(df_prob)

    # Download do JSON com o resultado
    result_json = json.dumps({
        "input_order": feature_columns,
        "input_values": ordered_vals,
        "pred_label": y_pred,
        "pred_species": species,
    }, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Baixar resultado (JSON)", data=result_json, file_name="iris_prediction.json", mime="application/json")

# Rodapé opcional
st.divider()
st.caption("Dica rápida: *petal_length* < ~2.5 → setosa; 3–5 (e *petal_width* ≤ ~1.8) → versicolor; > ~5 ou *petal_width* > ~1.8 → virginica.")
