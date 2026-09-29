# -*- coding: utf-8 -*-
"""
Divisões temporais do protocolo de comparação.

- ``dividir_treino_teste_temporal``: esquema do baseline antigo (treino < ano de teste, teste
  no ano inteiro). Origem: ``dengue_pipeline/modeling/train_tuning.py`` (commit 402f845).
  É o esquema usado na validação de equivalência (T9).
- ``dividir_cv_temporal``: dobras do tuning antigo, ``TimeSeriesSplit(n_splits=5, gap=4)``,
  reescritas sem scikit-learn com a mesma semântica.
- ``origens_expansivas`` e ``amostras_ate_origem``: origem expansiva com re-treino, proposta
  para o protocolo de pesquisa depois da equivalência (decisão aberta D-T1-1).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

from arbo_core.contracts import DEFAULT_NOTIFICATION_LAG_WEEKS
from arbo_core.evaluation.identity import TARGET_WEEK_COLUMN


def dividir_treino_teste_temporal(df: pd.DataFrame, ano_teste: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino: semanas antes de 1º de janeiro de ``ano_teste``. Teste: o ano ``ano_teste``."""
    treino = df[df["epi_sunday"] < pd.Timestamp(f"{ano_teste}-01-01")].copy()
    teste = df[
        (df["epi_sunday"] >= pd.Timestamp(f"{ano_teste}-01-01"))
        & (df["epi_sunday"] < pd.Timestamp(f"{ano_teste + 1}-01-01"))
    ].copy()
    return treino, teste


def dividir_cv_temporal(
    n_amostras: int, n_splits: int = 5, gap: int = DEFAULT_NOTIFICATION_LAG_WEEKS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Índices (treino, validação) com a semântica de ``sklearn.model_selection.TimeSeriesSplit``.

    Validação de tamanho ``n_amostras // (n_splits + 1)``, sempre ao fim; o treino termina
    ``gap`` posições antes da validação (atraso de notificação de ~4 semanas).
    """
    tamanho = n_amostras // (n_splits + 1)
    if tamanho < 1 or n_amostras - n_splits * tamanho - gap <= 0:
        raise ValueError(f"Amostras insuficientes ({n_amostras}) para {n_splits} dobras com gap={gap}.")
    indices = np.arange(n_amostras)
    for inicio in range(n_amostras - n_splits * tamanho, n_amostras, tamanho):
        yield indices[: inicio - gap], indices[inicio : inicio + tamanho]


def origens_expansivas(
    datas: Sequence[pd.Timestamp], primeira_origem: pd.Timestamp, passo_semanas: int = 4
) -> list[pd.Timestamp]:
    """Origens semanais a partir de ``primeira_origem``, de ``passo_semanas`` em ``passo_semanas``."""
    datas = sorted(pd.to_datetime(pd.Index(datas)).unique())
    elegiveis = [d for d in datas if d >= pd.Timestamp(primeira_origem)]
    return elegiveis[::passo_semanas]


def amostras_ate_origem(
    amostras: pd.DataFrame, origem: pd.Timestamp, gap: int = DEFAULT_NOTIFICATION_LAG_WEEKS
) -> pd.DataFrame:
    """
    Amostras de treino para uma origem: só as que tinham o alvo observado na origem.

    Uma amostra (origem ``t'``, alvo ``t'+k``) entra se ``t'+k <= origem - gap`` semanas; o gap
    reproduz o atraso de notificação.
    """
    limite = pd.Timestamp(origem) - pd.Timedelta(weeks=gap)
    return amostras[pd.to_datetime(amostras[TARGET_WEEK_COLUMN]) <= limite].copy()
