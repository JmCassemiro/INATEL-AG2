# AG2 — Iris (GaussianNB)

Estrutura inicial pronta. Próximos passos:
1) Crie um virtualenv e instale dependências:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```
2) Coloque o arquivo `iris.csv` em `data/iris.csv`.
3) Depois implementaremos `src/train.py` (GaussianNB) e `src/predict.py`.

Estrutura:
```
ag2-iris/
  data/                # CSV do Iris (colocar aqui)
  models/              # Modelos treinados (.joblib)
  src/                 # Código-fonte
    __init__.py
    train.py           # (a implementar) treino GaussianNB
    predict.py         # (a implementar) predição interativa
    utils.py           # (a implementar) helpers
  requirements.txt
  .gitignore
  README.md
```
