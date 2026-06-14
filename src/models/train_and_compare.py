from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.wrangle import TARGET_COLUMN, get_modeling_frame, wrangle, wrangle_dataframe

MODELS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=2.0),
    "lasso": Lasso(alpha=0.0005, max_iter=20000),
    "bayesian_ridge": BayesianRidge(),
    "knn": KNeighborsRegressor(n_neighbors=11),
}


def get_best_model_name(dataframe: pd.DataFrame | None = None) -> str:
    comparison = evaluate_models(dataframe)
    return str(comparison.loc[0, "model"])


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = feature_frame.select_dtypes(include="number").columns.tolist()
    categorical_columns = feature_frame.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def evaluate_models(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    dataframe = wrangle() if dataframe is None else dataframe
    feature_frame, target = get_modeling_frame(dataframe)

    preprocessor = build_preprocessor(feature_frame)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict[str, float | str]] = []

    for model_name, model in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            feature_frame,
            target,
            cv=cv,
            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
            error_score="raise",
        )
        results.append(
            {
                "model": model_name,
                "r2_mean": scores["test_r2"].mean(),
                "rmse_mean": -scores["test_neg_root_mean_squared_error"].mean(),
                "mae_mean": -scores["test_neg_mean_absolute_error"].mean(),
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        ["r2_mean", "rmse_mean", "mae_mean"],
        ascending=[False, True, True],
    )
    return results_frame.reset_index(drop=True)


def fit_best_model(dataframe: pd.DataFrame | None = None) -> tuple[Pipeline, pd.DataFrame, str]:
    dataframe = wrangle() if dataframe is None else dataframe
    feature_frame, target = get_modeling_frame(dataframe)
    best_model_name = get_best_model_name(dataframe)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_frame)),
            ("model", MODELS[best_model_name]),
        ]
    )
    pipeline.fit(feature_frame, target)
    return pipeline, feature_frame, best_model_name


def predict_from_dataframe(new_data: pd.DataFrame, pipeline: Pipeline | None = None) -> pd.DataFrame:
    if pipeline is None:
        pipeline, _, _ = fit_best_model()

    wrangled_data = wrangle_dataframe(new_data)
    feature_frame = wrangled_data.drop(columns=[column for column in [TARGET_COLUMN] if column in wrangled_data.columns])
    feature_frame = feature_frame.drop(columns=[column for column in ["peso_abatido_kg", "aves_abatidas", "peso_medio_calculado_kg", "ganho_peso_kg", "data_alojamento", "data_abate"] if column in feature_frame.columns])
    predictions = pipeline.predict(feature_frame)
    result = wrangled_data.copy()
    result["predicted_peso_medio_kg"] = predictions
    return result


def extract_lasso_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    lasso: Lasso = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": lasso.coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients[coefficients["abs_coefficient"] > 0]
    return coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def main() -> None:
    dataframe = wrangle()
    reports_tables = Path("reports/tables")
    models_dir = Path("models")
    reports_tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    comparison = evaluate_models(dataframe)
    comparison.to_csv(reports_tables / "model_comparison.csv", index=False)

    best_pipeline, _, best_model_name = fit_best_model(dataframe)
    dump(best_pipeline, models_dir / "best_model.joblib")

    if best_model_name == "lasso":
        coefficients = extract_lasso_coefficients(best_pipeline)
        coefficients.to_csv(reports_tables / "lasso_coefficients.csv", index=False)
    else:
        coefficients = pd.DataFrame()

    print(comparison.to_string(index=False))
    print(f"\nMelhor modelo: {best_model_name}")
    if not coefficients.empty:
        print("\nTop Lasso coefficients:")
        print(coefficients.head(15).to_string(index=False))
    print("\nModelo salvo em: models/best_model.joblib")


if __name__ == "__main__":
    main()