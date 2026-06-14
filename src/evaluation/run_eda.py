from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.features.wrangle import TARGET_COLUMN, build_quality_table, wrangle


sns.set_theme(style="whitegrid")


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    dataframe = wrangle()

    figures_dir = Path("reports/figures")
    tables_dir = Path("reports/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    quality_table = build_quality_table(dataframe)
    quality_table.to_csv(tables_dir / "data_quality.csv")

    numeric_summary = dataframe.describe(include="number").T.round(4)
    numeric_summary.to_csv(tables_dir / "numeric_summary.csv")

    category_summary = pd.DataFrame(
        {
            "n_unique": dataframe.select_dtypes(exclude="number").nunique(),
            "top_category": dataframe.select_dtypes(exclude="number").mode().iloc[0],
        }
    )
    category_summary.to_csv(tables_dir / "categorical_summary.csv")

    plt.figure(figsize=(9, 5))
    sns.histplot(dataframe[TARGET_COLUMN], bins=30, kde=True, color="#1f77b4")
    plt.title("Distribuicao de peso medio final")
    plt.xlabel("peso_medio_kg")
    save_figure(figures_dir / "target_distribution.png")

    plt.figure(figsize=(9, 5))
    top_missing = quality_table.query("missing_count > 0").head(10)
    if top_missing.empty:
        top_missing = quality_table.head(10)
    sns.barplot(x=top_missing["missing_pct"], y=top_missing.index, color="#e15759")
    plt.title("Percentual de nulos por coluna")
    plt.xlabel("missing_pct")
    plt.ylabel("coluna")
    save_figure(figures_dir / "missing_values.png")

    plt.figure(figsize=(10, 7))
    corr_columns = [
        TARGET_COLUMN,
        "racao_total_por_ave",
        "duracao_ciclo_dias",
        "idade_media_dias",
        "peso35",
        "peso42",
        "densidade_aves_m2",
        "pct_mort49",
        "sobra_racao_pct",
    ]
    corr = dataframe[corr_columns].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("Correlacoes numericas com foco no alvo")
    save_figure(figures_dir / "correlation_heatmap.png")

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=dataframe,
        x="racao_total_por_ave",
        y=TARGET_COLUMN,
        hue="regiao",
        alpha=0.5,
        legend=False,
    )
    plt.title("Peso medio x racao total por ave")
    save_figure(figures_dir / "target_vs_feed_per_bird.png")

    plt.figure(figsize=(10, 5))
    region_order = (
        dataframe.groupby("regiao")[TARGET_COLUMN].median().sort_values(ascending=False).index
    )
    sns.boxplot(data=dataframe, x="regiao", y=TARGET_COLUMN, order=region_order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Peso medio por regiao")
    save_figure(figures_dir / "target_by_region.png")

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=dataframe,
        x="pct_mort49",
        y=TARGET_COLUMN,
        alpha=0.5,
        color="#59a14f",
    )
    plt.title("Peso medio x mortalidade acumulada")
    save_figure(figures_dir / "target_vs_mortality.png")

    print("EDA concluida.")
    print(f"Tabela de qualidade salva em: {tables_dir / 'data_quality.csv'}")
    print(f"Figuras salvas em: {figures_dir}")


if __name__ == "__main__":
    main()