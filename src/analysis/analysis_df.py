import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from pathlib import Path

"""Análise exploratória do dataset WDBC para identificar padrões entre tumores benignos e malignos.
O intuito do módulo analysis concentra-se exclusivamente na análise dos dados brutos, com foco em:
- distribuição das classes;
- comparação visual entre benignos e malignos;
- investigação de atributos potencialmente discriminativos;
- correlação entre variáveis;
- interpretação dos padrões observados.
"""

ATRIBUTOS_IMPORTANTES = [
    "radius_mean", 
    "area_mean", 
    "perimeter_mean",
    "texture_mean"
]

def load_dataset(path: Path) -> pd.DataFrame:
    """ Carrega o dataset WDBC a partir do caminho especificado. """
    colunas = ["id", 
               "diagnosis", 
               *ATRIBUTOS_IMPORTANTES]
    df = pd.read_csv(path, header=None, names=colunas)
    df["diagnosis_label"] = df["diagnosis"].map({"B": "Benigno", "M": "Maligno"})

    return df

""" Testando as funções """
def main():
    path = Path("data/raw/wdbc.data")
    df = load_dataset(path)
    print(df)

if __name__ == "__main__":
    main()        