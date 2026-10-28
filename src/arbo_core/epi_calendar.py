# -*- coding: utf-8 -*-
"""Origem: ``dengue_pipeline/shared_kernel/epi_calendar.py`` (commit 402f845)."""
import pandas as pd

def calcular_semana_epidemiologica(series: pd.Series) -> pd.Series:
    """
    Calcula o domingo da semana epidemiológica correspondente para uma série de datas.
    
    Parâmetros:
        series: Pandas Series contendo as datas de sintomas ou registros.
        
    Retorna:
        Pandas Series com os domingos correspondentes normalizados e sem timezone.
        
    Premissas críticas:
        Assume que as datas de entrada podem ser convertidas pelo pd.to_datetime.
        A semana epidemiológica inicia no domingo.
    """
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    sundays = dates - pd.to_timedelta((dates.dt.weekday + 1) % 7, unit="D")
    return sundays.dt.tz_localize(None).dt.normalize()


def primeiro_domingo_epidemiologico(ano: int) -> pd.Timestamp:
    """
    Retorna o domingo inicial da semana epidemiológica 1 do ano.

    Para semanas domingo-sábado, a semana 1 é a que contém a maior parte dos
    dias em janeiro. Equivalentemente, é a semana que contém 4 de janeiro.
    """
    jan4 = pd.Timestamp(year=int(ano), month=1, day=4)
    return (jan4 - pd.Timedelta(days=(jan4.weekday() + 1) % 7)).normalize()


def calcular_epi_year_week(series: pd.Series) -> pd.DataFrame:
    """
    Calcula ano e semana epidemiológica sob convenção domingo-sábado.

    O ano epidemiológico é o ano da quarta-feira da semana, que é o dia que
    determina em qual ano cai a maioria da semana domingo-sábado.
    """
    epi_sunday = calcular_semana_epidemiologica(series)
    anchor = epi_sunday + pd.Timedelta(days=3)
    epi_year = anchor.dt.year.astype("Int64")
    first_sundays = epi_year.map(lambda y: primeiro_domingo_epidemiologico(int(y)) if pd.notna(y) else pd.NaT)
    epi_week = ((epi_sunday - first_sundays).dt.days // 7 + 1).astype("Int64")
    return pd.DataFrame({"epi_year": epi_year, "epi_week": epi_week})
