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

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "docs" / "figures" 

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    # Carrega o dataset WDBC a partir do caminho especificado. 
    colunas = ["id", 
               "diagnosis", 
               *ATRIBUTOS_IMPORTANTES]
    df = pd.read_csv(path, header=None, names=colunas)
    df["diagnosis_label"] = df["diagnosis"].map({"B": "Benigno", "M": "Maligno"})

    return df

def save_plot(grafico: plt.figure, nome_grafico: str, dataset_type: str):
    # Salvar o grafico, e guarda em uma pasta de graficos do dataset raw ou type

    if(dataset_type not in {"raw", "processed"}):
        # So deve permitir graficos do tipo de dataset raw ou processed
        raise ValueError(f"Tipo do dataset invalido: {dataset_type}")


    output_path = OUTPUT_DIR / dataset_type
    output_path.mkdir(parents=True, exist_ok=True)

    grafico.savefig(output_path / nome_grafico, dpi = 300, bbox_inches = "tight")
    plt.close(grafico)

""" Testando as funções """
def main():
    path = Path("data/raw/wdbc.data")
    df = load_dataset(path)
    print(df)

if __name__ == "__main__":
    main()        