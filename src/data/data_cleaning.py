import pandas as pd

features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

columns = ['id', 'diagnosis']

for stat in ['mean', 'se', 'worst']:
    for feat in features:
        columns.append(f'{feat}_{stat}')

# Carregar o dataset
df = pd.read_csv('wdbc.data', header=None, names=columns)

# Remover coluna de ID
df = df.drop(columns=['id'])

# Salvar CSV tratado
df.to_csv('wdbc_data_cleaned.csv', index=False)

print(f"Shape: {df.shape}")
print(f"Distribuição:\n{df['diagnosis'].value_counts()}")