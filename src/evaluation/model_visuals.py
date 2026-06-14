from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.features.wrangle import get_modeling_frame, wrangle
from src.models.train_and_compare import MODELS, build_preprocessor, get_best_model_name


sns.set_theme(style="whitegrid")


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("model", MODELS[model_name]),
        ]
    )


def main() -> None:
    figures_dir = Path("reports/figures")
    tables_dir = Path("reports/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataframe = wrangle()
    X, y = get_modeling_frame(dataframe)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    comparison_rows: list[dict[str, float | str]] = []
    fold_rows: list[dict[str, float | int]] = []

    best_model_name = get_best_model_name(dataframe)

    for model_name in MODELS:
        pipeline = build_pipeline(model_name, X)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        )
        comparison_rows.append(
            {
                "model": model_name,
                "r2_mean": scores["test_r2"].mean(),
                "rmse_mean": -scores["test_neg_root_mean_squared_error"].mean(),
                "mae_mean": -scores["test_neg_mean_absolute_error"].mean(),
            }
        )

        if model_name == best_model_name:
            for fold_index in range(5):
                fold_rows.append(
                    {
                        "fold": fold_index + 1,
                        "r2": scores["test_r2"][fold_index],
                        "rmse": -scores["test_neg_root_mean_squared_error"][fold_index],
                        "mae": -scores["test_neg_mean_absolute_error"][fold_index],
                    }
                )

    comparison = pd.DataFrame(comparison_rows).sort_values("r2_mean", ascending=False)
    comparison.to_csv(tables_dir / "model_comparison_visuals.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(tables_dir / "best_model_folds.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="model", y="r2_mean", palette="Blues_d")
    plt.title("Comparacao de modelos por R2 medio")
    plt.xlabel("modelo")
    plt.ylabel("R2 medio")
    save_figure(figures_dir / "model_r2_comparison.png")

    error_comparison = comparison.melt(
        id_vars="model",
        value_vars=["rmse_mean", "mae_mean"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=error_comparison, x="model", y="value", hue="metric", palette="Set2")
    plt.title("Comparacao de erros medios por modelo")
    plt.xlabel("modelo")
    plt.ylabel("erro")
    save_figure(figures_dir / "model_error_comparison.png")

    folds_long = folds.melt(id_vars="fold", value_vars=["r2", "rmse", "mae"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=folds_long, x="fold", y="value", hue="metric", marker="o")
    plt.title(f"Estabilidade do {best_model_name.upper()} nos 5 folds")
    plt.xlabel("fold")
    plt.ylabel("valor da metrica")
    save_figure(figures_dir / "best_model_fold_stability.png")

    best_pipeline = build_pipeline(best_model_name, X)
    predictions = cross_val_predict(best_pipeline, X, y, cv=cv)
    prediction_frame = pd.DataFrame({"real": y, "predito": predictions})
    prediction_frame["residuo"] = prediction_frame["real"] - prediction_frame["predito"]
    prediction_frame.to_csv(tables_dir / "best_model_predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=prediction_frame, x="real", y="predito", alpha=0.45)
    min_value = min(prediction_frame["real"].min(), prediction_frame["predito"].min())
    max_value = max(prediction_frame["real"].max(), prediction_frame["predito"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="crimson", linestyle="--")
    plt.title(f"Predito vs real com validacao cruzada - {best_model_name.upper()}")
    plt.xlabel("peso medio real")
    plt.ylabel("peso medio predito")
    save_figure(figures_dir / "best_model_actual_vs_predicted.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(prediction_frame["residuo"], bins=30, kde=True, color="#4e79a7")
    plt.title(f"Distribuicao dos residuos - {best_model_name.upper()}")
    plt.xlabel("residuo")
    save_figure(figures_dir / "best_model_residuals.png")

    print(f"Melhor modelo: {best_model_name}")
    print(f"Figuras salvas em: {figures_dir}")


if __name__ == "__main__":
    main()