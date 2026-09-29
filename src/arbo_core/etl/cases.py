# -*- coding: utf-8 -*-
"""Origem: ``dengue_pipeline/etl/case_ingestion.py`` (commit 402f845)."""
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from arbo_core.epi_calendar import calcular_semana_epidemiologica
from arbo_core.ras import FAMILIA_DENGUE, sanitizar_texto

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from arbo_core.contracts import TemporalContext


def _diretorio_info_saude() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("info-saude")

COLUNAS_INFO_SAUDE = [
    "i_class_final",
    "i_desc_classificacao",
    "i_desc_uf_res",
    "i_desc_radf_res",
    "i_data_prim_sintomas",
]

COLUNAS_DISPONIBILIDADE_CANDIDATAS = [
    "data_arquivo",
    "i_data_notificacao",
    "i_data_notificacao_sinannet",
    "i_data_digitacao",
    "i_data_encerramento",
]

def ingestar_dados_saude_local() -> pd.DataFrame:
    """
    Lê os arquivos CSV do diretório de vigilância de saúde, consolidando-os.
    Aplica normalização de strings e calcula o domingo epidemiológico para cada caso.
    
    Retorna:
        pd.DataFrame: DataFrame contendo todos os casos e colunas auxiliares de data e RA normalizadas.
        
    Premissas críticas:
        Os arquivos no diretório info-saude devem ser separados por ';' e ter codificação UTF-8.
    """
    diretorio_info_saude = _diretorio_info_saude()
    if not diretorio_info_saude.exists():
        raise FileNotFoundError(f"Diretório de dados info-saude não encontrado: {diretorio_info_saude}")
        
    frames = []
    # F5 (segurança): estes CSVs são dados locais de fonte oficial (info-saude/
    # SINAN), tratados como confiáveis. ``pd.read_csv`` não executa código, mas
    # ``usecols`` abaixo restringe a leitura às colunas esperadas — evita ingerir
    # colunas inesperadas e limita o consumo de memória. Se um dia esta função
    # passar a ler uploads de usuários, validar tamanho/esquema antes de ler.
    for file in sorted(diretorio_info_saude.glob("*.csv")):
        try:
            header = pd.read_csv(file, sep=";", encoding="utf-8", nrows=0).columns
            usecols = COLUNAS_INFO_SAUDE + [c for c in COLUNAS_DISPONIBILIDADE_CANDIDATAS if c in header]
            df = pd.read_csv(file, sep=";", encoding="utf-8", usecols=usecols)
        except UnicodeDecodeError:
            logger.warning(f"  [AVISO] Encoding UTF-8 falhou em {file.name}. Tentando latin-1...")
            header = pd.read_csv(file, sep=";", encoding="latin-1", nrows=0).columns
            usecols = COLUNAS_INFO_SAUDE + [c for c in COLUNAS_DISPONIBILIDADE_CANDIDATAS if c in header]
            df = pd.read_csv(file, sep=";", encoding="latin-1", usecols=usecols)
        df["source_file"] = file.name
        frames.append(df)
        
    if not frames:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {diretorio_info_saude}")
        
    return normalizar_casos(pd.concat(frames, ignore_index=True))


def normalizar_casos(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza textos e datas dos registros brutos e calcula o domingo epidemiológico.

    Separada da leitura de arquivos para permitir testes com registros sintéticos.
    """
    df_all = df_all.copy()
    df_all["class_norm"] = df_all["i_class_final"].map(sanitizar_texto)
    df_all["disease_norm"] = df_all["i_desc_classificacao"].map(sanitizar_texto)
    df_all["uf_norm"] = df_all["i_desc_uf_res"].map(sanitizar_texto)
    df_all["ra_norm_raw"] = df_all["i_desc_radf_res"].map(sanitizar_texto)
    df_all["date"] = pd.to_datetime(df_all["i_data_prim_sintomas"], errors="coerce", utc=True)
    for col in COLUNAS_DISPONIBILIDADE_CANDIDATAS:
        if col in df_all.columns:
            df_all[col] = pd.to_datetime(df_all[col], errors="coerce", utc=True)
    df_all["ano"] = df_all["date"].dt.year
    df_all["epi_sunday"] = calcular_semana_epidemiologica(df_all["i_data_prim_sintomas"])

    # Higiene de datas (4b): descarta registros com ano de primeiros sintomas
    # anterior ao início da série útil de vigilância. Anos como 1900/1938/.../2016
    # têm 1 caso cada e são erros de digitação do SINAN — removê-los evita poluir
    # a formalização do target e a grade temporal (proíbe tratamento silencioso).
    # Registros com data nula (NaT → ano NaN) NÃO são afetados aqui; o filtro de
    # disponibilidade temporal cuida deles.
    ANO_MINIMO_SERIE = 2017
    espurios = df_all["ano"].notna() & (df_all["ano"] < ANO_MINIMO_SERIE)
    n_espurios = int(espurios.sum())
    if n_espurios:
        anos_removidos = sorted(df_all.loc[espurios, "ano"].dropna().astype(int).unique().tolist())
        logger.warning(
            "Higiene de datas: removendo %d registro(s) com ano < %d (datas espúrias SINAN): %s",
            n_espurios, ANO_MINIMO_SERIE, anos_removidos,
        )
        df_all = df_all[~espurios].copy()

    return df_all

def mascaras_target(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Gera máscaras booleanas para diferentes definições de filtragem epidemiológica do target.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os casos de dengue.
        
    Retorna:
        dict[str, pd.Series]: Dicionário indexado pelos nomes das configurações contendo a série booleana correspondente.
    """
    simples = df["class_norm"].eq("CASO PROVAVEL")
    dengue_exata = simples & df["disease_norm"].eq("DENGUE")
    familia_dengue = simples & df["disease_norm"].isin(FAMILIA_DENGUE)
    return {
        "simples_caso_provavel": simples,
        "duplo_dengue_exata": dengue_exata,
        "familia_dengue": familia_dengue,
    }

def aplicar_disponibilidade_temporal(df: pd.DataFrame, contexto: "TemporalContext") -> pd.DataFrame:
    """
    Filtra o dataset para remover dados não observáveis na data de corte simulada,
    aplicando o watermark de atraso (right-censoring operacional).
    """
    initial_len = len(df)
    
    # Remover e registrar casos com datas nulas (NaT) que quebram o fluxo temporal
    valid_dates = df["date"].notna()
    if not valid_dates.all():
        n_invalidos = (~valid_dates).sum()
        logger.warning(f"Ignorando {n_invalidos} casos ({n_invalidos/initial_len:.1%}) por falta de 'data_prim_sintomas' válida.")
        df = df[valid_dates].copy()
        
    from arbo_core.contracts import ForecastMode

    if contexto.modo_avaliacao == ForecastMode.RETROSPECTIVE:
        mask = df["date"].dt.date <= contexto.data_cutoff
        return df[mask].copy()
    
    mask_event = df["date"].dt.date <= contexto.data_cutoff
    
    if contexto.availability_time_field and contexto.availability_time_field in df.columns:
        valid_avail = df[contexto.availability_time_field].notna()
        if not valid_avail.all():
            n_invalidos = (~valid_avail).sum()
            logger.warning(f"Ignorando {n_invalidos} casos por falta de campo '{contexto.availability_time_field}'.")
        mask_avail = valid_avail & (df[contexto.availability_time_field].dt.date <= contexto.as_of_date)
    else:
        from datetime import timedelta
        limite_observacao = contexto.as_of_date - timedelta(weeks=contexto.notification_lag_weeks)
        mask_avail = df["date"].dt.date <= limite_observacao
        
    return df[mask_event & mask_avail].copy()
