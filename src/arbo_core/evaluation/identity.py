# -*- coding: utf-8 -*-
"""Identidade temporal das previsões diretas por horizonte.

Origem: ``dengue_pipeline/modeling/forecast_identity.py`` (commit 402f845).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FORECAST_ORIGIN_COLUMN = "forecast_origin"
TARGET_WEEK_COLUMN = "target_epi_sunday"
TARGET_VALUE_COLUMN = "target_k"
OBSERVED_CASES_COLUMN = "observed_cases"


def preparar_amostras_horizonte(
    df: pd.DataFrame,
    target_col: str,
    horizonte: int,
) -> pd.DataFrame:
    """Liga cada origem à semana-alvo e à observação futura da mesma RA.

    ``epi_sunday`` permanece sendo a data de origem enquanto as amostras são
    usadas no treino e na calibração. A semana e os valores observados no alvo
    são deslocados juntos, evitando que previsão, observação e unidade espacial
    percam sua associação.
    """
    if horizonte < 1:
        raise ValueError("O horizonte deve ser maior ou igual a 1.")

    required = {"epi_sunday", "RA", "cases", target_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colunas ausentes para identidade temporal: {missing}")

    result = df.copy()
    result["epi_sunday"] = pd.to_datetime(result["epi_sunday"])
    result = result.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)
    grouped = result.groupby("RA", sort=False)

    result[FORECAST_ORIGIN_COLUMN] = result["epi_sunday"]
    result[TARGET_WEEK_COLUMN] = grouped["epi_sunday"].shift(-horizonte)
    result[TARGET_VALUE_COLUMN] = grouped[target_col].shift(-horizonte)
    result[OBSERVED_CASES_COLUMN] = grouped["cases"].shift(-horizonte)

    for column in ("incidencia_100k", "populacao"):
        if column in result.columns:
            result[f"target_{column}"] = grouped[column].shift(-horizonte)

    return result.dropna(
        subset=[TARGET_WEEK_COLUMN, TARGET_VALUE_COLUMN, OBSERVED_CASES_COLUMN]
    ).reset_index(drop=True)


def materializar_resultado_forecast(
    amostras: pd.DataFrame,
    predictions: Sequence[float] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Materializa previsões na semana e escala observadas no alvo."""
    required = {
        FORECAST_ORIGIN_COLUMN,
        TARGET_WEEK_COLUMN,
        OBSERVED_CASES_COLUMN,
        "RA",
    }
    missing = sorted(required.difference(amostras.columns))
    if missing:
        raise ValueError(f"Colunas ausentes para materializar forecast: {missing}")

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or len(values) != len(amostras):
        raise ValueError(
            "Predições devem ser unidimensionais e ter uma entrada por amostra."
        )

    result = pd.DataFrame(
        {
            FORECAST_ORIGIN_COLUMN: amostras[FORECAST_ORIGIN_COLUMN].to_numpy(),
            "epi_sunday": amostras[TARGET_WEEK_COLUMN].to_numpy(),
            "RA": amostras["RA"].to_numpy(),
            "cases": amostras[OBSERVED_CASES_COLUMN].astype(float).to_numpy(),
            "prediction": values,
        }
    )

    for column in ("incidencia_100k", "populacao"):
        target_column = f"target_{column}"
        if target_column in amostras.columns:
            result[column] = amostras[target_column].to_numpy()

    return result
