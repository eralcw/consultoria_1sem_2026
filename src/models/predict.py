from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from src.features.wrangle import TARGET_COLUMN
from src.models.train_and_compare import predict_from_dataframe


DEFAULT_MODEL_PATH = Path("models/best_model.joblib")


def load_input_table(file_path: Path, sheet_name: str = "Base_Dados") -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError("Formato nao suportado. Use .xlsx, .xls ou .csv.")


def build_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_predicoes.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera predicoes de peso medio usando o melhor modelo treinado.")
    parser.add_argument("input_file", help="Arquivo de entrada (.xlsx, .xls ou .csv)")
    parser.add_argument("--sheet", default="Base_Dados", help="Nome da aba quando o arquivo for Excel")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Caminho do modelo salvo em joblib")
    parser.add_argument("--output", default=None, help="Caminho do CSV de saida")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else build_output_path(input_path)

    raw_data = load_input_table(input_path, sheet_name=args.sheet)
    pipeline = load(model_path)
    predicted = predict_from_dataframe(raw_data, pipeline)
    predicted.to_csv(output_path, index=False)

    print(f"Arquivo lido: {input_path}")
    print(f"Modelo usado: {model_path}")
    print(f"Arquivo salvo: {output_path}")

    if TARGET_COLUMN in predicted.columns:
        error = (predicted[TARGET_COLUMN] - predicted["predicted_peso_medio_kg"]).abs().mean()
        print(f"MAE medio no arquivo informado: {error:.6f}")

    print("\nPrimeiras predicoes:")
    columns_to_show = [column for column in [TARGET_COLUMN, "predicted_peso_medio_kg"] if column in predicted.columns]
    print(predicted[columns_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()