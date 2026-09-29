# -*- coding: utf-8 -*-
"""
Contratos temporais e operacionais.

Origem: ``dengue_pipeline/shared_kernel/contracts.py`` (commit 402f845), mais
``INICIO_SERIE_ESPACIAL``, que vivia em ``modeling/feature_engineering.py``.
Camada sem dependências internas: importada por todos, nunca importa ninguém.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


# Início do painel RA×semana. Regra inviolável.
#
# Antes de 2020 a ficha do InfoSaúde traz "Não Informado" em quase todos os registros
# (99,9% em 2017, 99,3% em 2019). Como a grade preenche com zero as células sem caso,
# começar antes fabricaria anos de calmaria inexistente. A partir de 2020 a ausência de
# identificação territorial cai para 17%–29% e é tratada por exclusão de linha, não por zero.
INICIO_SERIE_ESPACIAL = pd.Timestamp("2020-01-01")

# Atraso de notificação usado nas janelas operacionais e no gap da validação.
DEFAULT_NOTIFICATION_LAG_WEEKS = 4

# Alvo epidemiológico oficial. Regra inviolável.
#
# `simples_caso_provavel` = i_class_final == "CASO PROVAVEL", sem filtrar a descrição da
# doença: a definição do Ministério da Saúde ("notificado suspeito exceto descartado").
# `familia_dengue` exige ainda descrição DENGUE / COM SINAIS DE ALARME / GRAVE e exclui os
# INCONCLUSIVOS (cerca de 27% dos casos prováveis do DF). Fica só como análise de
# sensibilidade: um ganho que só existe sob definição não padrão não é reportável.
ALVO_PADRAO = "simples_caso_provavel"
ALVO_SENSIBILIDADE = "familia_dengue"

ARTIFACT_SCHEMA_VERSION = "tdd11.v1"
FORECAST_PROBLEM_VERSION = "tdd11.forecast_problem.v1"
TEMPORAL_CONTEXT_VERSION = "tdd11.temporal_context.v1"
QUANTILE_SCHEMA_VERSION = "tdd11.quantile_schema.v1"


class ForecastMode(str, Enum):
    RETROSPECTIVE = "RETROSPECTIVE"
    ROLLING_NOWCAST = "ROLLING_NOWCAST"
    ROLLING_DIRECT_K = "ROLLING_DIRECT_K"
    CLOSED_FORECAST_FROM_CUTOFF = "CLOSED_FORECAST_FROM_CUTOFF"


class WeatherPolicy(str, Enum):
    RETROSPECTIVE_INTERPOLATION = "RETROSPECTIVE_INTERPOLATION"
    OPERATIONAL_OBSERVED_ONLY = "OPERATIONAL_OBSERVED_ONLY"
    OPERATIONAL_FORECAST_SOURCE = "OPERATIONAL_FORECAST_SOURCE"


class PopulationScope(str, Enum):
    URBANA = "URBANA"
    URBANA_RURAL = "URBANA_RURAL"
    UNKNOWN_BLOCKED = "UNKNOWN_BLOCKED"


class WatermarkPolicy(str, Enum):
    CONSERVATIVE_NOTIFICATION_LAG = "CONSERVATIVE_NOTIFICATION_LAG"
    AVAILABILITY_TIME = "AVAILABILITY_TIME"
    RETROSPECTIVE_COMPLETE = "RETROSPECTIVE_COMPLETE"


@dataclass(frozen=True)
class ForecastProblemContract:
    target_name: str = ALVO_PADRAO
    target_event_time: str = "i_data_prim_sintomas"
    spatial_level: str = "RA+DF_TOTAL"
    expected_spatial_units: tuple[str, ...] = field(default_factory=tuple)
    frequency: str = "epi_week_sunday_saturday"
    horizon_family: tuple[int, ...] = (1, 2, 3, 4)
    use_case: str = "RETROSPECTIVE"
    epi_week_policy: str = "SINAN_sunday_saturday_majority_january"


@dataclass(frozen=True)
class TemporalContext:
    run_id: str
    as_of_date: pd.Timestamp
    data_cutoff: pd.Timestamp | None = None
    event_time_field: str = "i_data_prim_sintomas"
    availability_time_field: str | None = None
    processing_time: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(datetime.now()))
    watermark_policy: WatermarkPolicy = WatermarkPolicy.CONSERVATIVE_NOTIFICATION_LAG
    modo_avaliacao: ForecastMode = ForecastMode.RETROSPECTIVE
    notification_lag_weeks: int = 4
    weather_policy: WeatherPolicy = WeatherPolicy.RETROSPECTIVE_INTERPOLATION
    population_scope: PopulationScope = PopulationScope.UNKNOWN_BLOCKED
    target_name: str = ALVO_PADRAO

    @classmethod
    def retrospective(
        cls,
        target_name: str = ALVO_PADRAO,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> "TemporalContext":
        as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
        return cls(
            run_id=f"retrospective-{as_of.date()}",
            as_of_date=as_of,
            modo_avaliacao=ForecastMode.RETROSPECTIVE,
            watermark_policy=WatermarkPolicy.RETROSPECTIVE_COMPLETE,
            notification_lag_weeks=0,
            weather_policy=WeatherPolicy.RETROSPECTIVE_INTERPOLATION,
            target_name=target_name,
        )

    @classmethod
    def operational(
        cls,
        as_of_date: str | pd.Timestamp,
        target_name: str = ALVO_PADRAO,
        notification_lag_weeks: int = 4,
        modo_avaliacao: ForecastMode = ForecastMode.ROLLING_NOWCAST,
    ) -> "TemporalContext":
        as_of = pd.Timestamp(as_of_date).normalize()
        data_cutoff = as_of - pd.Timedelta(weeks=notification_lag_weeks)
        return cls(
            run_id=f"{modo_avaliacao.value.lower()}-{as_of.date()}",
            as_of_date=as_of,
            data_cutoff=data_cutoff,
            modo_avaliacao=modo_avaliacao,
            notification_lag_weeks=notification_lag_weeks,
            weather_policy=WeatherPolicy.OPERATIONAL_OBSERVED_ONLY,
            target_name=target_name,
        )

    @property
    def is_operational(self) -> bool:
        return self.modo_avaliacao is not ForecastMode.RETROSPECTIVE

    def resolved_cutoff(self, last_event_time: pd.Timestamp) -> pd.Timestamp:
        last_event_time = pd.Timestamp(last_event_time).normalize()
        if self.data_cutoff is not None:
            cutoff = pd.Timestamp(self.data_cutoff).normalize()
        elif self.is_operational:
            cutoff = self.as_of_date - pd.Timedelta(weeks=self.notification_lag_weeks)
        else:
            cutoff = last_event_time
        return min(cutoff, last_event_time)

    def metadata(self) -> dict[str, str | int | None]:
        return {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "forecast_problem_version": FORECAST_PROBLEM_VERSION,
            "temporal_context_version": TEMPORAL_CONTEXT_VERSION,
            "quantile_schema_version": QUANTILE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "as_of_date": self.as_of_date.date().isoformat(),
            "data_cutoff": None if self.data_cutoff is None else pd.Timestamp(self.data_cutoff).date().isoformat(),
            "event_time_field": self.event_time_field,
            "availability_time_field": self.availability_time_field,
            "processing_time": self.processing_time.isoformat(),
            "watermark_policy": self.watermark_policy.value,
            "modo_avaliacao": self.modo_avaliacao.value,
            "notification_lag_weeks": self.notification_lag_weeks,
            "weather_policy": self.weather_policy.value,
            "population_scope": self.population_scope.value,
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class ExogenousCovariateContract:
    source: str
    availability_lag_weeks: int
    allowed_modes: tuple[ForecastMode, ...]
    biological_lag_range: tuple[int, int]
    future_policy: str

    def permite(self, modo: ForecastMode) -> bool:
        """Indica se a covariável pode ser usada sob o modo de avaliação dado."""
        return modo in self.allowed_modes


# Google Trends BR-DF — primeira fonte a declarar contrato exógeno explícito.
#
# `availability_lag_weeks=0`: o Trends publica em tempo quase real, bem antes do SINAN
# (cujo atraso de notificação o pipeline trata com `notification_lag_weeks=4`). O gargalo
# não é a publicação e sim a CALIBRAÇÃO: o `annual_calibration_factor` do arquivo é
# constante por ano e derivado da consulta anual fechada, que não existe para o ano
# corrente. Por isso `allowed_modes` fica restrito a RETROSPECTIVE — em modo operacional
# o nível da própria covariável carregaria informação do futuro.
#
# `biological_lag_range=(1, 4)`: medido no dado. A correlação bruta com casos é 0,989 em
# lag 0, mas quase toda ela é persistência; controlando `cases_lag_1` e `cases_lag_2`, a
# correlação parcial cai para +0,25 (lag 1), +0,10 (lag 2) e vira negativa (−0,24 a −0,30)
# nos lags 3 a 6 — sinal de reversão, não de antecedência.
GOOGLE_TRENDS_CONTRACT = ExogenousCovariateContract(
    source="google_trends_br_df",
    availability_lag_weeks=0,
    allowed_modes=(ForecastMode.RETROSPECTIVE,),
    biological_lag_range=(1, 4),
    future_policy="BLOCKED_OUTSIDE_RETROSPECTIVE",
)
