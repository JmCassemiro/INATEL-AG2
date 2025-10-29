# AG2 — Classificação de Íris (Gaussian Naive Bayes)

Projeto de Análise e Classificação do dataset Iris (UCI). Pipeline completo em Python: leitura do CSV, pré-processamento, split 80/20, treino de GaussianNB, métricas, e predição via CLI e UI Web (Streamlit).

Mapeamento de rótulos (exigido): setosa → 1, versicolor → 2, virginica → 3.

## ✅ O que este projeto entrega

- Leitura do CSV com Pandas
- Normalização de species e mapeamento com Series.replace → 1/2/3 
- Split 80/20 com shuffle=True e random_state=42
- Treino com Gaussian Naive Bayes (GaussianNB)
- Métricas no console e em models/metrics.json
- Predição:
  - CLI rápida (predict.py)
  - CLI amigável com rich (predict_pretty.py)
  - UI Web com Streamlit (app_streamlit.py)

## 🗂️ Estrutura

```
ag2-iris/
  data/
    iris.csv
  models/
    iris_nb.joblib
    species_mapping.json
    metrics.json
  src/
    utils.py
    train.py
    predict.py            
    predict_pretty.py     
    train_report.py       
    app_streamlit.py      
  requirements.txt
  README.md
```

## 🧰 Requisitos

- Python 3.10+ (recomendado)

**requirements.txt:**
```
scikit-learn==1.7.2
pandas
numpy
joblib
rich
streamlit
```

## ⚙️ Setup

### 1) Ambiente e dependências

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Dataset
Coloque o iris.csv em `data/iris.csv`.

## 🏋️‍♀️ Treinar o modelo

```bash
python src/train.py
```

Saídas:
- Console: Accuracy, Classification Report, Confusion Matrix
- Arquivos gerados em `models/`

## 🔮 Predição

### CLI Rápida
```bash
python src/predict.py --values "5.1,3.5,1.4,0.2"
# ou modo interativo
python src/predict.py
```

### CLI Rica (com Rich)
```bash
python src/predict_pretty.py
# ou com valores
python src/predict_pretty.py --values "5.1,3.5,1.4,0.2"
```

### UI Web (Streamlit)
```bash
streamlit run src/app_streamlit.py
```

## 🧭 Casos de Teste

**Setosa**
- 5.1,3.5,1.4,0.2
- 4.9,3.0,1.4,0.2

**Versicolor**
- 6.0,2.9,4.5,1.5
- 5.5,2.6,4.4,1.2

**Virginica**
- 6.5,3.0,5.2,2.0
- 7.1,3.0,5.9,2.1

## 💡 Heurística
- petal_length < ~2.5 → setosa
- petal_length 3–5 e petal_width ≤ ~1.8 → versicolor
- petal_length > ~5 ou petal_width > ~1.8 → virginica

## 📜 Licença
Uso acadêmico/educacional, conforme diretrizes da disciplina.