SPECIES_TO_INT = {
    "setosa": 1,
    "versicolor": 2,
    "virginica": 3,
}
INT_TO_SPECIES = {v: k for k, v in SPECIES_TO_INT.items()}

def normalize_species_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("iris-", "").replace("iris ", "").replace("iris_", "")
    return s

def _norm_key(c: str) -> str:
    c = str(c).lower()
    return "".join(ch for ch in c if ch.isalnum())


_FEATURE_SYNONYMS = {
    "sepal_length": {
        "sepal_length", "sepal.length", "sepal length", "sepal_length_cm",
        "sepal length (cm)", "sepallength", "sepallengthcm"
    },
    "sepal_width": {
        "sepal_width", "sepal.width", "sepal width", "sepal_width_cm",
        "sepal width (cm)", "sepalwidth", "sepalwidthcm"
    },
    "petal_length": {
        "petal_length", "petal.length", "petal length", "petal_length_cm",
        "petal length (cm)", "petallength", "petallengthcm"
    },
    "petal_width": {
        "petal_width", "petal.width", "petal width", "petal_width_cm",
        "petal width (cm)", "petalwidth", "petalwidthcm"
    },
}

def resolve_feature_columns(df):
    norm_lookup = {_norm_key(c): c for c in df.columns}
    resolved = []
    for base, variants in _FEATURE_SYNONYMS.items():
        found = None
        for v in variants:
            nk = _norm_key(v)
            if nk in norm_lookup:
                found = norm_lookup[nk]
                break
        if not found:
            raise KeyError(f"Coluna não encontrada para '{base}'. Colunas no CSV: {list(df.columns)}")
        resolved.append(found)
    return resolved  
