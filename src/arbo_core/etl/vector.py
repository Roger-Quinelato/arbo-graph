# -*- coding: utf-8 -*-
"""
Origem: ``dengue_pipeline/etl/vector_ingestion.py`` (commit 402f845).

Ingestão da vigilância entomológica por ovitrampas (SES-DF).

Fonte: ``ovitrampas/*.csv`` — uma linha por leitura de armadilha, separador ``;``,
colunas relevantes: ``Município`` (nome da RA com sufixo "(RA ...)"), ``Data``
(data da coleta), ``Ovos`` (contagem) e ``Resultado`` (Positiva/Negativa).

Índices semanais derivados por RA (nomenclatura da vigilância entomológica):
    - ``ovi_ipo``: Índice de Positividade de Ovitrampas (fração de armadilhas positivas)
    - ``ovi_ovos_media``: média de ovos por armadilha lida (proxy do IDO)
    - ``ovi_n_armadilhas``: nº de leituras na semana×RA (denominador/QA)

CAVEAT operacional documentado: a coluna "Tempo de Envio da Contagem" mostra que
parte das leituras é digitada com semanas/meses de atraso. As features derivadas
usam a DATA DE COLETA com lags ≥ 2 semanas (janela biológica ovo→adulto→transmissão),
mas em operação real a disponibilidade pode ser menor — mesma classe de limitação
do atraso de notificação do SINAN, já tratada por ``TemporalContext``.
"""

from pathlib import Path
import logging

import pandas as pd

from arbo_core.epi_calendar import calcular_semana_epidemiologica
from arbo_core.ras import consolidar_ra, sanitizar_texto

logger = logging.getLogger(__name__)

# Aliases específicos da base de ovitrampas (nomes fora do padrão do registry).
_ALIAS_OVITRAMPAS: dict[str, str] = {
    "BRASILIA": "PLANO PILOTO",
    "SCIA/ESTRUTURAL": "SCIA",
    "SCIA (ESTRUTURAL)": "SCIA",
}


def _diretorio_ovitrampas() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("ovitrampas")


def carregar_indices_ovitrampas() -> pd.DataFrame:
    """
    Lê todos os CSVs de ovitrampas e agrega em índices semanais por RA.

    Retorna:
        pd.DataFrame: colunas ``epi_sunday``, ``RA`` (canônica consolidada),
            ``ovi_n_armadilhas``, ``ovi_ipo``, ``ovi_ovos_media``.

    Levanta:
        FileNotFoundError: se o diretório não existir ou não contiver CSVs.
    """
    diretorio = _diretorio_ovitrampas()
    arquivos = sorted(diretorio.glob("*.csv")) if diretorio.exists() else []
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV de ovitrampas encontrado em {diretorio}")

    frames = []
    for arquivo in arquivos:
        df = pd.read_csv(arquivo, sep=";", encoding="utf-8", low_memory=False)
        df["source_file"] = arquivo.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    ra_bruta = (
        df["Município"].astype(str)
        .str.replace(r"\s*\(RA[^)]*\)", "", regex=True)
        .map(sanitizar_texto)
    )
    df["RA"] = ra_bruta.map(lambda nome: consolidar_ra(_ALIAS_OVITRAMPAS.get(nome, nome)))

    df["data_coleta"] = pd.to_datetime(df["Data"], errors="coerce")
    df["ovos"] = pd.to_numeric(df["Ovos"], errors="coerce")
    df["positiva"] = df["Resultado"].map(sanitizar_texto).eq("POSITIVA").astype(float)
    df["epi_sunday"] = calcular_semana_epidemiologica(df["data_coleta"])

    invalidas = df["epi_sunday"].isna() | df["RA"].isna()
    if invalidas.any():
        logger.warning(
            "Ovitrampas: descartando %d leitura(s) sem data ou RA válida.", int(invalidas.sum())
        )
    df = df[~invalidas]

    agregado = (
        df.groupby(["epi_sunday", "RA"])
        .agg(
            ovi_n_armadilhas=("ovos", "size"),
            ovi_ipo=("positiva", "mean"),
            ovi_ovos_media=("ovos", "mean"),
        )
        .reset_index()
    )
    logger.info(
        "Ovitrampas: %d leituras → %d semanas×RA (%s a %s, %d RAs)",
        len(df), len(agregado),
        agregado["epi_sunday"].min().date(), agregado["epi_sunday"].max().date(),
        agregado["RA"].nunique(),
    )
    return agregado
