# -*- coding: utf-8 -*-
"""
Painel RA×semana com casos, incidência e features defasadas.

Origem: ``dengue_pipeline/modeling/feature_engineering.construir_dataset_consolidado``
(commit 402f845). A lógica foi preservada; a diferença é a separação entre leitura de
fontes (``construir_painel``) e transformação pura (``montar_painel``), para permitir
testes com dados sintéticos. Grupos de features de outras fontes entram como
``enriquecedores``: funções ``DataFrame -> DataFrame`` aplicadas depois dos lags de casos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from arbo_core.epi_calendar import calcular_epi_year_week
from arbo_core.contracts import ALVO_PADRAO, INICIO_SERIE_ESPACIAL, TemporalContext
from arbo_core.etl.cases import mascaras_target
from arbo_core.ras import padronizar_regioes_administrativas

logger = logging.getLogger(__name__)

CASE_LAGS: list[int] = [1, 2, 3, 4]

Enriquecedor = Callable[[pd.DataFrame], pd.DataFrame]


def agregar_casos(
    casos_brutos: pd.DataFrame,
    contexto: TemporalContext,
    target_name: str = ALVO_PADRAO,
    lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Filtra o alvo, a UF e a RA, e conta casos por semana epidemiológica e RA.

    Registro sem RA identificada é excluído aqui: ausência de identificação territorial
    não é zero observado.
    """
    masks = mascaras_target(casos_brutos)
    if target_name not in masks:
        raise ValueError(f"Filtro de target epidemiológico desconhecido: {target_name}")

    df = casos_brutos.loc[masks[target_name]].copy()
    df = df[df["uf_norm"].eq("DF")].copy()
    df["RA"] = df["i_desc_radf_res"].map(lambda x: padronizar_regioes_administrativas(x, lookup))
    df = df[df["RA"].notna() & df["epi_sunday"].notna()].copy()

    availability_field = contexto.availability_time_field
    if availability_field and availability_field in df.columns:
        available_at = pd.to_datetime(df[availability_field], errors="coerce", utc=True).dt.tz_localize(None)
        df = df[available_at.le(contexto.as_of_date)].copy()

    cases = df.groupby(["epi_sunday", "RA"]).size().reset_index(name="cases")
    if cases.empty:
        raise ValueError("Dataset sem casos válidos após filtros de target, UF, RA e disponibilidade temporal.")
    return cases


def montar_grade(cases: pd.DataFrame, populacao: pd.DataFrame, contexto: TemporalContext) -> pd.DataFrame:
    """
    Grade completa RA×semana desde ``INICIO_SERIE_ESPACIAL``, com população e incidência.

    Célula sem caso de uma RA identificada vale 0: é contagem observada. O universo de RAs
    vem da tabela de população.
    """
    valid_ras = sorted(populacao["RA"].unique())
    last_event_time = pd.Timestamp(cases["epi_sunday"].max()).normalize()
    end = contexto.resolved_cutoff(last_event_time)
    all_sundays = pd.date_range(start=INICIO_SERIE_ESPACIAL, end=end, freq="W-SUN")
    grid = pd.MultiIndex.from_product([all_sundays, valid_ras], names=["epi_sunday", "RA"]).to_frame(index=False)

    dataset = grid.merge(cases, on=["epi_sunday", "RA"], how="left")
    dataset["cases"] = dataset["cases"].fillna(0).astype(float)

    # Guarda fail-loud: RA com 0 casos em toda a série indica falha de normalização/join.
    casos_por_ra = dataset.groupby("RA")["cases"].sum()
    ras_mortas = casos_por_ra[casos_por_ra == 0].index.tolist()
    if ras_mortas:
        raise ValueError(
            f"RAs com 0 casos em toda a série (provável falha de join/normalização): {ras_mortas}"
        )

    epi_parts = calcular_epi_year_week(dataset["epi_sunday"])
    dataset["epi_year"] = epi_parts["epi_year"].astype(int)
    dataset["epi_week"] = epi_parts["epi_week"].astype(int)
    dataset["ano"] = dataset["epi_year"].astype(int)

    dataset = dataset.merge(populacao[["RA", "ano", "populacao"]], on=["RA", "ano"], how="inner")
    dataset["incidencia_100k"] = dataset["cases"] / dataset["populacao"] * 100000
    return dataset.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)


def adicionar_lags_casos(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Lags 1–4 de casos e incidência, calculados dentro de cada RA, e tendência de curto prazo.

    O ``groupby("RA")`` isola cada série antes do ``shift``: sem ele, a primeira semana de
    uma RA puxaria a última semana da RA anterior.
    """
    dataset = dataset.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)
    for col in ["cases", "incidencia_100k"]:
        for lag in CASE_LAGS:
            dataset[f"{col}_lag_{lag}"] = dataset.groupby("RA")[col].shift(lag)

    idx_primeira_semana = dataset.groupby("RA")["epi_sunday"].idxmin()
    if not dataset.loc[idx_primeira_semana, "cases_lag_1"].isna().all():
        raise AssertionError(
            "Leakage detectado: cases_lag_1 não-NaN na 1ª semana de alguma RA — "
            "o groupby('RA') do shift de lags foi quebrado."
        )

    dataset["cases_delta_1"] = dataset["cases_lag_1"] - dataset["cases_lag_2"]
    dataset["cases_delta_2"] = dataset["cases_lag_2"] - dataset["cases_lag_3"]
    # +1 no denominador evita divisão por zero em semanas sem casos.
    dataset["cases_growth_rate"] = (dataset["cases_lag_1"] + 1) / (dataset["cases_lag_2"] + 1)
    return dataset


def adicionar_sazonais(dataset: pd.DataFrame) -> pd.DataFrame:
    """Semana e mês do ano com codificação cíclica seno/cosseno."""
    week = dataset["epi_week"].astype(int)
    month = dataset["epi_sunday"].dt.month.astype(int)
    dataset["week_of_year"] = week
    dataset["month"] = month
    dataset["sin_week"] = np.sin(2 * np.pi * week / 53)
    dataset["cos_week"] = np.cos(2 * np.pi * week / 53)
    dataset["sin_month"] = np.sin(2 * np.pi * month / 12)
    dataset["cos_month"] = np.cos(2 * np.pi * month / 12)
    return dataset


def montar_painel(
    casos_brutos: pd.DataFrame,
    populacao: pd.DataFrame,
    contexto: TemporalContext,
    target_name: str = ALVO_PADRAO,
    lookup: dict[str, str] | None = None,
    enriquecedores: Sequence[Enriquecedor] = (),
) -> pd.DataFrame:
    """Transformação pura: casos brutos e população em painel com features defasadas."""
    cases = agregar_casos(casos_brutos, contexto, target_name, lookup)
    dataset = montar_grade(cases, populacao, contexto)
    dataset = adicionar_lags_casos(dataset)
    for enriquecer in enriquecedores:
        dataset = enriquecer(dataset)
    dataset = adicionar_sazonais(dataset)
    for key, value in contexto.metadata().items():
        dataset[key] = value
    return dataset


def construir_painel(
    contexto: TemporalContext,
    target_name: str = ALVO_PADRAO,
    enriquecedores: Sequence[Enriquecedor] = (),
) -> pd.DataFrame:
    """Lê as fontes via ``diretorio_fonte`` e monta o painel."""
    from arbo_core.etl.cases import ingestar_dados_saude_local
    from arbo_core.ras import busca_ra_canonica, carregar_historico_populacao

    casos_brutos = ingestar_dados_saude_local()
    populacao = carregar_historico_populacao()
    lookup = busca_ra_canonica()
    return montar_painel(casos_brutos, populacao, contexto, target_name, lookup, enriquecedores)
