# AG2 — Classificação de Íris (Gaussian Naive Bayes)

Projeto de **Análise e Classificação** do conjunto de dados **Iris** (UCI). Implementa **pipeline completo** em Python: leitura do CSV, pré-processamento, divisão **80/20 com embaralhamento**, treino de **GaussianNB**, **métricas de avaliação**, e **predição** a partir de valores informados pelo usuário (CLI).  
> **Mapeamento de rótulos (obrigatório):** `setosa → 1`, `versicolor → 2`, `virginica → 3`.

---

## 📌 Objetivos (conforme enunciado)
- Ler o **CSV do Iris** (link do PDF) com **Pandas**.
- **Normalizar** a coluna `species` e **mapear** para inteiros (1/2/3) na ordem especificada.
- Separar **treino/teste** com `train_test_split(test_size=0.2, shuffle=True)` e `random_state` fixo.
- Treinar **1 modelo** dentre os listados (usamos **GaussianNB**).
- Exibir **métricas de avaliação**.
- Disponibilizar **predição** a partir de dados informados pelo usuário.
- Entrega: **ZIP ou GitHub** + **vídeo (≤ 7 min)** demonstrando todo o fluxo.

---

## 🗂️ Estrutura do Projeto

```
ag2-iris/
  data/
    iris.csv                  # <- coloque o CSV aqui
  models/
    iris_nb.joblib            # gerado pelo treino
    species_mapping.json      # gerado pelo treino
    metrics.json              # gerado pelo treino
  src/
    __init__.py               # opcional
    utils.py                  # helpers (mapeamento e resolução de colunas)
    train.py                  # treino + métricas + salvamento de artefatos
    predict.py                # predição via CLI (--values / --json ou interativo)
  requirements.txt
  README.md
  .gitignore
```

---

## 🧰 Requisitos

- **Python 3.10+** (recomendado)
- Bibliotecas: `scikit-learn`, `pandas`, `numpy`, `joblib`  
  (já listadas em `requirements.txt`)

---

## ⚙️ Setup Rápido

### 1) Criar ambiente e instalar dependências
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux (bash/zsh):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Colocar o CSV
- Salve o arquivo **`iris.csv`** em `ag2-iris/data/iris.csv`.

**Formato esperado (flexível):**
- Coluna de rótulo: `species` (aceita também `target`, `class`, `variety`, `label` → renomeadas internamente).
- Nomes das 4 features podem variar (`sepal_length`, `sepal.length`, `sepal length`, `sepal_length_cm`, etc.). O código **resolve automaticamente** para:
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.

### 3) Treinar o modelo
```powershell
python src/train.py
```

Saídas:
- **Console:** Accuracy, Classification Report, Confusion Matrix.
- **Arquivos:**
  - `models/iris_nb.joblib`
  - `models/species_mapping.json`
  - `models/metrics.json`

---

## 🧪 Métricas de Avaliação

O script imprime e também **salva** em `models/metrics.json`:
- `accuracy`
- `classification_report` (precision, recall, f1-score, support por classe)
- `confusion_matrix`
- Metadados (labels, target_names, feature_columns, split, random_state, modelo).

Exemplo (console):
```
Accuracy: 1.0000

Classification report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

Confusion matrix (rows=true, cols=pred):
[[10  0  0]
 [ 0  9  0]
 [ 0  0  11]]
```

---

## 🔮 Predição (CLI)

### Opção A — 4 valores em linha
```powershell
python src/predict.py --values "5.1,3.5,1.4,0.2"
```

### Opção B — JSON
```powershell
python src/predict.py --json "{\"sepal_length_cm\": 5.1, \"sepal_width_cm\": 3.5, \"petal_length_cm\": 1.4, \"petal_width_cm\": 0.2}"
```

### Opção C — Interativo
```powershell
python src/predict.py
```
O script exibirá a **ordem exata** das features conforme aprendida no treino (por exemplo: `sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm`). Digite 4 valores nessa ordem.

**Saída (JSON):**
```json
{
  "input_order": ["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"],
  "input_values": [5.1,3.5,1.4,0.2],
  "pred_label": 1,
  "pred_species": "setosa"
}
```

---

## 🧭 Guia de Testes (valores → espécie esperada)

Use a **ordem exibida** pelo `predict.py` (pode incluir `_cm` conforme seu CSV).
- **Setosa**
  - `5.1,3.5,1.4,0.2`
  - `4.9,3.0,1.4,0.2`
- **Versicolor**
  - `6.0,2.9,4.5,1.5`
  - `5.5,2.6,4.4,1.2`
- **Virginica**
  - `6.5,3.0,5.2,2.0`
  - `7.1,3.0,5.9,2.1`

> **Regrinha mental rápida:**  
> `petal_length < ~2.5` → **setosa**  
> `petal_length 3–5` e `petal_width ≤ ~1.8` → **versicolor**  
> `petal_length > ~5` ou `petal_width > ~1.8` → **virginica**

---

## 🔧 Detalhes de Implementação

- **Modelo escolhido:** `GaussianNB` (simples, rápido e cumpre os requisitos).
- **Split:** `train_test_split(test_size=0.2, shuffle=True, random_state=42)`.
- **Normalização de rótulos:**  
  - Remoção de prefixos `Iris-`, `Iris `, `Iris_` → compara em minúsculas.  
  - Mapeamento final: `{"setosa":1, "versicolor":2, "virginica":3}`.
- **Resolução de colunas:** robusta a variações (`pontos`, `espaços`, `snake_case`, sufixos `_cm`, etc).
- **Artefatos salvos:**
  - `iris_nb.joblib` (modelo + ordem das colunas + mapeamentos).
  - `species_mapping.json` (mapeamentos e colunas).
  - `metrics.json` (todas as métricas + metadados).

---

## 🧱 Reprodutibilidade
- `random_state=42` no split.
- Fixe versões no `requirements.txt` (ex.: `scikit-learn==1.7.2`).
- **Atenção:** abrir `.joblib` gerado em versão diferente do scikit-learn pode emitir **InconsistentVersionWarning**. Melhor **treinar localmente** com suas versões para o vídeo/entrega.

---

## 🩹 Troubleshooting

- **`FileNotFoundError: CSV não encontrado`**  
  Garanta que o arquivo está em `data/iris.csv` (ou passe `--csv` no `train.py`).

- **`KeyError: 'species' não encontrada`**  
  Renomeie sua coluna de classe para `species` ou use `target/class/variety/label`. O código renomeia para `species`.

- **`Coluna não encontrada para 'sepal_length'`**  
  Verifique os nomes das features no CSV. O código aceita variações (`sepal.length`, `sepal length`, `_cm`, etc). Se ainda falhar, ajuste no `utils.py` (dicionário `_FEATURE_SYNONYMS`).

- **Aviso de versão do scikit-learn (InconsistentVersionWarning)**  
  Retreine localmente (`python src/train.py`) com a sua versão ou alinhe a versão do sklearn no `requirements.txt`.

---

## 🎥 Roteiro do Vídeo (≤ 7 min)

1. **Contexto rápido** (dataset Iris, 3 classes, 4 features).  
2. **Abrir o projeto** e mostrar `data/iris.csv`.  
3. **Rodar o treino** (`python src/train.py`): comentar split 80/20, shuffle, GaussianNB.  
4. Mostrar **métricas** no console e o arquivo `models/metrics.json`.  
5. **Predição**: `python src/predict.py --values "..."`
6. Encerrar recapitulando o mapeamento (setosa=1, versicolor=2, virginica=3).

---

## 📦 Entrega

- **GitHub (público)** ou **ZIP** do projeto completo.
- Garanta que o revisor consegue:
  1) instalar (`pip install -r requirements.txt`),  
  2) treinar (`python src/train.py`),  
  3) prever (`python src/predict.py --values "..."`).

---

## ✅ Checklist (AG2)

- [x] CSV lido com **Pandas**  
- [x] `species` normalizada e mapeada para **1/2/3** na ordem correta  
- [x] `train_test_split(0.2, shuffle=True, random_state=42)`  
- [x] Modelo **GaussianNB** treinado  
- [x] **Métricas** exibidas + salvas em `metrics.json`  
- [x] **Predição** via CLI (valores do usuário)  
- [x] **Artefatos** do modelo salvos  
- [x] Pronto para **ZIP/GitHub** + **vídeo** demonstrativo

---

## 📜 Licença
Uso acadêmico/educacional, conforme diretrizes da disciplina. Ajuste se necessário.
