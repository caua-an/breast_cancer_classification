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
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "docs" / "figures" 


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

def plot_class_distribution(df: pd.DataFrame, dataset_type: str):
    # modela o grafico
    fig, ax = plt.subplots(figsize=(6,4))
    sb.countplot(data=df, x="diagnosis_label", order=["Benigno", "Maligno"], ax=ax)
    ax.set_title("Distribuição de Tumores")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")
    ax.set_axisbelow(True)
    ax.margins(y=0.10)
    # escreve o numero que foi contado pelo countplot()
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=4)
    # salva o grafico
    save_plot(fig, "distribuicao_BM.png", dataset_type)

""" Testando as funções """
def main():
    path_raw = Path("data/raw/wdbc.data")
    df = load_dataset(path_raw)
    plot_class_distribution(df, "raw")

if __name__ == "__main__":
    main()        