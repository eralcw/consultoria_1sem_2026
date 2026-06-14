from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/Base_Dados_Modelos_peso.xlsx")
TARGET_COLUMN = "peso_medio_kg"
LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "peso_abatido_kg",
    "aves_abatidas",
    "peso_medio_calculado_kg",
    "ganho_peso_kg",
}


def normalize_column_name(column_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(column_name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("%", "pct_")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("²", "2")
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_").lower()


def wrangle_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.rename(columns=normalize_column_name)
    dataframe = dataframe.drop_duplicates().copy()

    for column_name in ("data_alojamento", "data_abate"):
        dataframe[column_name] = pd.to_datetime(dataframe[column_name], errors="coerce")

    dataframe["duracao_ciclo_dias"] = (
        dataframe["data_abate"] - dataframe["data_alojamento"]
    ).dt.days
    dataframe["data_alojamento_ano"] = dataframe["data_alojamento"].dt.year
    dataframe["data_alojamento_mes"] = dataframe["data_alojamento"].dt.month
    dataframe["data_alojamento_dia"] = dataframe["data_alojamento"].dt.day
    dataframe["data_abate_ano"] = dataframe["data_abate"].dt.year
    dataframe["data_abate_mes"] = dataframe["data_abate"].dt.month
    dataframe["data_abate_dia"] = dataframe["data_abate"].dt.day
    dataframe["peso_medio_calculado_kg"] = (
        dataframe["peso_abatido_kg"] / dataframe["aves_abatidas"]
    )
    dataframe["racao_total_por_ave"] = dataframe["racao_total"] / dataframe["aves_alojadas"]
    dataframe["sobra_racao_pct"] = np.where(
        dataframe["racao_total"].gt(0),
        dataframe["sobra_racao"] / dataframe["racao_total"],
        np.nan,
    )
    dataframe["ganho_peso_kg"] = dataframe["peso_medio_kg"] - dataframe["peso_pintinho_kg"]
    dataframe["taxa_abate"] = dataframe["aves_abatidas"] / dataframe["aves_alojadas"]
    dataframe["mortalidade_total_pct"] = dataframe["pct_mort49"]

    return dataframe.sort_values(["data_alojamento", "produtor", "num_galpao"]).reset_index(drop=True)


def wrangle(file_path: str | Path = DEFAULT_DATA_PATH, sheet_name: str = "Base_Dados") -> pd.DataFrame:
    dataframe = pd.read_excel(file_path, sheet_name=sheet_name)
    return wrangle_dataframe(dataframe)


def build_quality_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    quality_table = pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing_count": dataframe.isna().sum(),
            "missing_pct": dataframe.isna().mean().mul(100).round(2),
            "n_unique": dataframe.nunique(dropna=False),
        }
    )
    return quality_table.sort_values(["missing_pct", "n_unique"], ascending=[False, False])


def get_modeling_frame(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    blocked_columns = set(drop_columns or set()) | LEAKAGE_COLUMNS
    blocked_columns |= {"data_alojamento", "data_abate"}
    feature_frame = dataframe.drop(columns=[column for column in blocked_columns if column in dataframe.columns])
    return feature_frame, dataframe[target_column]