import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "wdbc.data"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def remove_outliers(df: pd.DataFrame):
    # seleciona as colunas numericas
    colunas_num = df.select_dtypes(include=["number"]).columns
    mask = pd.Series(True, index=df.index)

    for coluna in colunas_num:
        # primeiro e terceiro quartil
        q1 = df[coluna].quantile(0.25)
        q3 = df[coluna].quantile(0.75)
        iqr = q3 - q1
        limite_sup = q1 - 1.5 * iqr
        limite_inf = q3 + 1.5 * iqr
        # mask vai acumulando os outliers com o &=
        mask &= df[coluna].between(limite_inf, limite_sup, inclusive="both")

    return df.loc[mask].copy()

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

# Criar versão sem outliers
df_no_outliers = remove_outliers(df)
df_no_outliers.to_csv(PROCESSED_DIR / "wdbc_data_no_outliers.csv", index=False)

# processo de normalização do df
scaler = RobustScaler()
df_scaled = df.copy()

# normaliza as colunas numéricas, exceto a primeira coluna diagnosis
df_scaled.iloc[:, 1:] = scaler.fit_transform(df_scaled.iloc[:, 1:])

df_scaled.to_csv(PROCESSED_DIR / "wdbc_data_normalized.csv", index=False)


print(f"Shape: {df.shape}")
print(f"Distribuição:\n{df['diagnosis'].value_counts()}")