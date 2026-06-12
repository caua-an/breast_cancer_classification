import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "wdbc.data"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

columns = ['id', 'diagnosis']

for stat in ['mean', 'se', 'worst']:
    for feat in features:
        columns.append(f'{feat}_{stat}')

# Carregar o dataset
df = pd.read_csv(RAW_DATA_PATH, header=None, names=columns)

# Remover coluna de ID
df = df.drop(columns=['id'])

# Salvar CSV tratado
df.to_csv(PROCESSED_DIR / "wdbc_data_cleaned.csv", index=False)


# processo de normalização do df
scaler = RobustScaler()
df_scaled = df.copy()

# normaliza as colunas numéricas, exceto a primeira coluna diagnosis
df_scaled.iloc[:, 1:] = scaler.fit_transform(df_scaled.iloc[:, 1:])

df_scaled.to_csv(PROCESSED_DIR / "wdbc_data_normalized.csv", index=False)


print(f"Shape: {df.shape}")
print(f"Distribuição:\n{df['diagnosis'].value_counts()}")