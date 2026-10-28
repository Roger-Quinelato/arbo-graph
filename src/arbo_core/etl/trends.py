# -*- coding: utf-8 -*-
"""
Origem: ``dengue_pipeline/etl/trends_ingestion.py`` (commit 402f845).

Ingestão do índice semanal de busca do Google Trends para o Distrito Federal.

Fonte: ``Google Trends/*.csv`` — uma linha por semana epidemiológica, separador ``,``,
recorte geográfico ``BR-DF``. A coluna usada é ``google_trends_dengue``: o volume
relativo de buscas pelo termo "dengue" na semana, já reescalonado entre anos.

Por que o reescalonamento existe: o Google Trends normaliza cada consulta ao próprio
máximo (100), de modo que uma consulta anual de 2021 (ano de baixa) e uma de 2024
(epidemia) chegam ambas com pico 100 e não são comparáveis entre si. O arquivo corrige
isso multiplicando cada ano por ``annual_calibration_factor``, encadeado a um ano de
referência (2024) e validado contra uma consulta mensal âncora — daí as colunas de QA
``monthly_anchor_correlation`` (0,97 a 0,998) e ``monthly_anchor_rmse``.

CAVEAT operacional documentado — motivo do contrato ``GOOGLE_TRENDS_CONTRACT``:
o ``annual_calibration_factor`` é constante dentro do ano e derivado da consulta anual
FECHADA. Para o ano corrente esse fator não existe ainda, então usar esta série em modo
operacional injetaria informação do futuro no nível da própria covariável. A série é
liberada apenas em ``ForecastMode.RETROSPECTIVE``; a porta é aplicada em
``arbo_core.features.GoogleTrends``, não aqui — este módulo só lê e valida.

Segunda limitação, análoga à do clima de ponto único: a consulta é ``BR-DF``, uma série
só para as 31 RAs. Informa o TIMING do surto, não discrimina entre regiões.

Os demais termos do arquivo (chikungunya, zika, sintomas_dengue, arboviroses) ficam de
fora: são esparsos demais para virar feature — 61%, 73%, 47% e 99% de semanas em zero,
contra 0% de ``google_trends_dengue``.
"""

from pathlib import Path
import logging

import pandas as pd

from arbo_core.epi_calendar import calcular_epi_year_week

logger = logging.getLogger(__name__)

# Recorte geográfico esperado. Qualquer outro valor indica arquivo trocado.
GEO_ESPERADO = "BR-DF"

# Única coluna de termo promovida a feature (ver docstring do módulo).
COLUNA_TERMO = "google_trends_dengue"


def _diretorio_trends() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("Google Trends")


def carregar_indice_google_trends() -> pd.DataFrame:
    """
    Lê os CSVs do Google Trends e devolve o índice semanal do termo "dengue" no DF.

    Retorna:
        pd.DataFrame: colunas ``epi_sunday`` (domingo da semana epidemiológica) e
            ``google_trends_dengue``, ordenado por ``epi_sunday``.

    Levanta:
        FileNotFoundError: se o diretório não existir ou não contiver CSVs.
        ValueError: se o recorte geográfico não for ``BR-DF``, se houver semanas
            duplicadas, se ``week_start`` não cair em domingo, ou se o calendário
            epidemiológico do arquivo divergir de ``arbo_core.epi_calendar``.
    """
    diretorio = _diretorio_trends()
    arquivos = sorted(diretorio.glob("*.csv")) if diretorio.exists() else []
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV do Google Trends encontrado em {diretorio}")

    frames = []
    for arquivo in arquivos:
        parcial = pd.read_csv(arquivo, encoding="utf-8")
        parcial["source_file"] = arquivo.name
        frames.append(parcial)
    df = pd.concat(frames, ignore_index=True)

    faltantes = {"week_start", "epi_year", "epi_week", "geo", COLUNA_TERMO} - set(df.columns)
    if faltantes:
        raise ValueError(f"Google Trends: colunas ausentes no CSV: {sorted(faltantes)}")

    # Guarda fail-loud 1: recorte geográfico. Um arquivo de outra UF passaria
    # despercebido no merge (que é só por epi_sunday) e contaminaria a série.
    geos = sorted(df["geo"].dropna().astype(str).unique())
    if geos != [GEO_ESPERADO]:
        raise ValueError(
            f"Google Trends: recorte geográfico inesperado {geos} (esperado ['{GEO_ESPERADO}'])"
        )

    df["epi_sunday"] = pd.to_datetime(df["week_start"], errors="coerce")
    if df["epi_sunday"].isna().any():
        raise ValueError("Google Trends: há week_start não parseável como data.")

    # Guarda fail-loud 2: uma linha por semana. Duplicata viraria fan-out no merge,
    # multiplicando linhas do painel RA×semana.
    duplicadas = df["epi_sunday"].duplicated().sum()
    if duplicadas:
        raise ValueError(f"Google Trends: {duplicadas} semana(s) duplicada(s) em week_start.")

    # Guarda fail-loud 3: o merge com o painel é por epi_sunday, que no dataset é
    # sempre domingo. Um arquivo com semanas em segunda desalinharia tudo em silêncio.
    nao_domingo = df.loc[df["epi_sunday"].dt.dayofweek != 6, "epi_sunday"]
    if len(nao_domingo):
        raise ValueError(
            f"Google Trends: {len(nao_domingo)} semana(s) não começam no domingo "
            f"(ex.: {nao_domingo.iloc[0].date()})."
        )

    # Guarda fail-loud 4: o calendário epidemiológico do arquivo tem de ser o mesmo do
    # pipeline (regra SINAN domingo-sábado, majoritária de janeiro). Divergência aqui
    # significaria que a série está deslocada de uma semana em relação aos casos.
    esperado = calcular_epi_year_week(df["epi_sunday"])
    divergentes = (
        esperado["epi_year"].ne(df["epi_year"].astype(int))
        | esperado["epi_week"].ne(df["epi_week"].astype(int))
    )
    if divergentes.any():
        exemplo = df.loc[divergentes, "epi_sunday"].iloc[0].date()
        raise ValueError(
            f"Google Trends: {int(divergentes.sum())} semana(s) com epi_year/epi_week "
            f"divergentes do calendário do pacote (ex.: {exemplo}). A série está deslocada."
        )

    df[COLUNA_TERMO] = pd.to_numeric(df[COLUNA_TERMO], errors="coerce")
    if df[COLUNA_TERMO].isna().any():
        raise ValueError(f"Google Trends: valores não numéricos em {COLUNA_TERMO}.")
    if (df[COLUNA_TERMO] < 0).any():
        raise ValueError(f"Google Trends: valores negativos em {COLUNA_TERMO}.")

    indice = (
        df[["epi_sunday", COLUNA_TERMO]]
        .sort_values("epi_sunday")
        .reset_index(drop=True)
    )
    logger.info(
        "Google Trends: %d semanas (%s a %s), %s médio=%.2f máx=%.2f",
        len(indice),
        indice["epi_sunday"].min().date(),
        indice["epi_sunday"].max().date(),
        COLUNA_TERMO,
        indice[COLUNA_TERMO].mean(),
        indice[COLUNA_TERMO].max(),
    )
    return indice
