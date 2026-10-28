# -*- coding: utf-8 -*-
"""
Grupos de features defasadas de fontes exógenas, aplicados ao painel como enriquecedores.

Origem: ``dengue_pipeline/modeling/feature_engineering.py`` (commit 402f845). Constantes de
lag e lógica de cada grupo preservadas. Cada enriquecedor declara as fontes que lê
(``fontes``) e registra avisos de cobertura em ``caveats``, que o manifesto publica.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from arbo_core.contracts import GOOGLE_TRENDS_CONTRACT, TemporalContext, WeatherPolicy

logger = logging.getLogger(__name__)

# Janelas de lag do clima de ponto único. Todo shift é estritamente retrospectivo.
# precip_sum 2–11: a correlação de Spearman precipitação↔casos sobe até o pico no lag 11
# (rho=0,28) e estabiliza; a janela antiga 2–6 cortava cerca de 24% do sinal.
LAG_WINDOWS: dict[str, range] = {
    "temp_mean": range(2, 5),
    "precip_sum": range(2, 12),
    "umidmed": range(1, 4),
}

# Clima por RA (INMET, 5 estações, IDW). Mesmas janelas do ponto único, para que uma
# eventual vantagem venha da espacialização e não de defasagens a mais.
CLIMA_RA_LAGS: dict[str, range] = {
    "temp_mean_ra": LAG_WINDOWS["temp_mean"],
    "precip_sum_ra": LAG_WINDOWS["precip_sum"],
    "umid_med_ra": LAG_WINDOWS["umidmed"],
}


# Ovitrampas: janela biológica ovo→adulto→transmissão de cerca de 2–4 semanas.
OVI_LAGS: list[int] = [2, 3, 4]

# Lags de casos das RAs vizinhas (difusão inter-RA de curto prazo).
VIZ_LAGS: list[int] = [1, 2]

# Google Trends BR-DF (``GOOGLE_TRENDS_CONTRACT.biological_lag_range``). Controlando
# cases_lag_1 e cases_lag_2, a correlação parcial é +0,25 no lag 1, +0,10 no lag 2 e negativa
# nos lags 3 a 6: os lags 3 e 4 entram como sinal de reversão.
GT_LAGS: list[int] = [1, 2, 3, 4]


def colunas_vizinhanca() -> list[str]:
    return [f"viz_cases_lag_{lag}" for lag in VIZ_LAGS]


def colunas_ovitrampas() -> list[str]:
    ovi = [f"ovi_ipo_lag_{lag}" for lag in OVI_LAGS]
    ovi += [f"ovi_ovos_media_lag_{lag}" for lag in OVI_LAGS]
    return ovi + ["ovi_cobertura"]


def colunas_google_trends() -> list[str]:
    return [f"google_trends_dengue_lag_{lag}" for lag in GT_LAGS] + ["gt_cobertura"]


def colunas_clima_ponto_unico() -> list[str]:
    return [f"{col}_lag_{lag}" for col, lags in LAG_WINDOWS.items() for lag in lags]


def colunas_clima_ra() -> list[str]:
    return [f"{col}_lag_{lag}" for col, lags in CLIMA_RA_LAGS.items() for lag in lags] + ["clima_ra_cobertura"]


class ClimaPontoUnico:
    """
    Clima de ponto único do DF (Open-Meteo + umidade do InfoDengue), defasado antes do merge.

    Série única para o DF, então o ``shift`` é feito na própria série climática, que pode
    começar antes de 2020: os primeiros lags de 2020 usam clima de 2019, já observado.

    Limitação herdada: sob ``WeatherPolicy.RETROSPECTIVE_INTERPOLATION`` a umidade faltante é
    interpolada nos dois sentidos, o que usa valores futuros. Para o protocolo de pesquisa use
    ``OPERATIONAL_OBSERVED_ONLY`` (preenchimento só para frente).
    """

    fontes: tuple[str, ...] = ("clima_cache", "InfoDengue")

    def __init__(
        self,
        weather_policy: WeatherPolicy | None = None,
        as_of_date: pd.Timestamp | None = None,
        clima: pd.DataFrame | None = None,
    ) -> None:
        self.weather_policy = weather_policy
        self.as_of_date = as_of_date
        self.clima = clima
        self.caveats: dict = {}

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        clima = self.clima
        if clima is None:
            from arbo_core.etl.climate import carregar_cache_climatico

            clima = carregar_cache_climatico(weather_policy=self.weather_policy, as_of_date=self.as_of_date)
        clima = clima.sort_values("epi_sunday").reset_index(drop=True).copy()
        for col, lags in LAG_WINDOWS.items():
            for lag in lags:
                clima[f"{col}_lag_{lag}"] = clima[col].shift(lag)
        dataset = dataset.merge(clima, on="epi_sunday", how="left")
        return dataset.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)


class ClimaRA:
    """
    Clima por RA (INMET interpolado por IDW), defasado dentro de cada RA.

    O ``shift`` é feito por RA depois do merge; sem o ``groupby`` a primeira semana de uma RA
    herdaria o clima da última semana da RA anterior. As colunas contemporâneas são
    descartadas. Degrada com aviso quando a fonte falta, como no repo antigo; o motivo fica
    em ``caveats``.
    """

    fontes: tuple[str, ...] = ("inmet", "dados_geo")

    def __init__(self, clima_ra: pd.DataFrame | None = None) -> None:
        self.clima_ra = clima_ra
        self.caveats: dict = {}

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        try:
            clima = self.clima_ra
            if clima is None:
                from arbo_core.etl.climate import carregar_clima_por_ra

                clima = carregar_clima_por_ra()
            self.caveats["clima_ra_ras_distantes_km"] = clima.attrs.get("ras_distantes_km", [])

            colunas = ["epi_sunday", "RA", *CLIMA_RA_LAGS]
            dataset = dataset.merge(clima[colunas], on=["epi_sunday", "RA"], how="left")
            dataset = dataset.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)
            for col, lags in CLIMA_RA_LAGS.items():
                for lag in lags:
                    dataset[f"{col}_lag_{lag}"] = dataset.groupby("RA")[col].shift(lag)

            # 1 quando havia leitura na semana de referência mais próxima: separa "sem estação"
            # de "estação registrou zero". Cada variável tem sua janela; usa a da temperatura.
            menor_lag_temp = min(CLIMA_RA_LAGS["temp_mean_ra"])
            dataset["clima_ra_cobertura"] = dataset[f"temp_mean_ra_lag_{menor_lag_temp}"].notna().astype(float)
            dataset = dataset.drop(columns=list(CLIMA_RA_LAGS))
            self.caveats["clima_ra_cobertura"] = float(dataset["clima_ra_cobertura"].mean())
        except Exception as e:  # noqa: BLE001 — fonte opcional; o motivo vai para o manifesto
            logger.warning("Features de clima por RA puladas: %s", e)
            self.caveats["clima_ra_pulado"] = str(e)
        return dataset


class Vizinhanca:
    """
    Média dos casos defasados das RAs adjacentes (``viz_cases_lag_1``, ``viz_cases_lag_2``).

    Usa só ``cases_lag_k`` já defasado das vizinhas. RA sem vizinha no painel fica NaN,
    nunca 0. RAs do painel sem polígono e polígonos sem série vão para ``caveats``.
    Degrada com aviso quando o geo.json falta.
    """

    fontes: tuple[str, ...] = ("dados_geo",)

    def __init__(self, adjacencia: dict[str, frozenset[str]] | None = None) -> None:
        self.adjacencia = adjacencia
        self.caveats: dict = {}

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        from arbo_core.graph import carregar_adjacencia_ras, mapear_nos

        try:
            adjacencia = self.adjacencia if self.adjacencia is not None else carregar_adjacencia_ras()
        except FileNotFoundError as e:
            logger.warning("Features de vizinhança puladas: %s", e)
            self.caveats["vizinhanca_pulado"] = str(e)
            return dataset

        mapa = mapear_nos(dataset["RA"].unique(), adjacencia)
        self.caveats["grafo_ras_sem_poligono"] = list(mapa.ras_sem_poligono)
        self.caveats["grafo_poligonos_sem_painel"] = list(mapa.poligonos_sem_painel)

        ras_dataset = set(dataset["RA"].unique())
        for lag in VIZ_LAGS:
            larga = dataset.pivot(index="epi_sunday", columns="RA", values=f"cases_lag_{lag}")
            viz_larga = pd.DataFrame(index=larga.index)
            for ra in ras_dataset:
                vizinhas = [v for v in adjacencia.get(ra, frozenset()) if v in larga.columns]
                # np.nan em vez de pd.NA (repo antigo): no pandas 3 a coluna de pd.NA não
                # converte para float. O valor continua sendo ausência, nunca 0.
                viz_larga[ra] = larga[vizinhas].mean(axis=1) if vizinhas else np.nan
            longa = viz_larga.reset_index().melt(
                id_vars="epi_sunday", var_name="RA", value_name=f"viz_cases_lag_{lag}"
            )
            dataset = dataset.merge(longa, on=["epi_sunday", "RA"], how="left")
            dataset[f"viz_cases_lag_{lag}"] = dataset[f"viz_cases_lag_{lag}"].astype(float)
        return dataset


class Ovitrampas:
    """
    Índices de ovitrampas (IPO e média de ovos) defasados 2–4 semanas dentro de cada RA.

    Exceção documentada à regra "ausência não vira zero" (decisão 5): os lags ausentes são
    preenchidos com 0 e ``ovi_cobertura`` marca se havia leitura no menor lag. A árvore separa
    "sem vigilância" (cobertura 0) de "vigilância negativa" (IPO 0, cobertura 1). Degrada com
    aviso quando a fonte falta.
    """

    fontes: tuple[str, ...] = ("ovitrampas",)

    def __init__(self, indices: pd.DataFrame | None = None) -> None:
        self.indices = indices
        self.caveats: dict = {}

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        try:
            ovi = self.indices
            if ovi is None:
                from arbo_core.etl.vector import carregar_indices_ovitrampas

                ovi = carregar_indices_ovitrampas()
        except FileNotFoundError as e:
            logger.warning("Features de ovitrampas puladas: %s", e)
            self.caveats["ovitrampas_pulado"] = str(e)
            return dataset

        dataset = dataset.merge(
            ovi[["epi_sunday", "RA", "ovi_ipo", "ovi_ovos_media"]], on=["epi_sunday", "RA"], how="left"
        )
        dataset = dataset.sort_values(["RA", "epi_sunday"]).reset_index(drop=True)
        for col in ["ovi_ipo", "ovi_ovos_media"]:
            for lag in OVI_LAGS:
                dataset[f"{col}_lag_{lag}"] = dataset.groupby("RA")[col].shift(lag)
        menor_lag = min(OVI_LAGS)
        dataset["ovi_cobertura"] = dataset[f"ovi_ipo_lag_{menor_lag}"].notna().astype(float)
        cols_lag = [f"{col}_lag_{lag}" for col in ["ovi_ipo", "ovi_ovos_media"] for lag in OVI_LAGS]
        dataset[cols_lag] = dataset[cols_lag].fillna(0.0)
        self.caveats["ovitrampas_cobertura"] = float(dataset["ovi_cobertura"].mean())
        return dataset


class GoogleTrends:
    """
    Índice semanal de busca "dengue" (BR-DF) defasado 1–4 semanas.

    Porta de contrato: ``GOOGLE_TRENDS_CONTRACT`` só libera a fonte em modo RETROSPECTIVE,
    porque o fator de calibração anual não existe para o ano corrente. Fora disso as colunas
    não são criadas. Série única para o DF: o ``shift`` é feito antes do merge.

    Exceção documentada: lags ausentes viram 0 ao lado de ``gt_cobertura``, definida pelo MAIOR
    lag, para que a virada 2019/2020 não pareça "ninguém pesquisou dengue". Degrada com aviso
    quando a fonte falta.
    """

    fontes: tuple[str, ...] = ("Google Trends",)

    def __init__(self, contexto: TemporalContext, indice: pd.DataFrame | None = None) -> None:
        self.contexto = contexto
        self.indice = indice
        self.caveats: dict = {}

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not GOOGLE_TRENDS_CONTRACT.permite(self.contexto.modo_avaliacao):
            motivo = (
                f"contrato {GOOGLE_TRENDS_CONTRACT.source} permite apenas "
                f"{[m.value for m in GOOGLE_TRENDS_CONTRACT.allowed_modes]}; modo "
                f"{self.contexto.modo_avaliacao.value}"
            )
            logger.warning("Features do Google Trends PULADAS: %s", motivo)
            self.caveats["google_trends_pulado"] = motivo
            return dataset

        from arbo_core.etl.trends import COLUNA_TERMO

        try:
            trends = self.indice
            if trends is None:
                from arbo_core.etl.trends import carregar_indice_google_trends

                trends = carregar_indice_google_trends()
        except FileNotFoundError as e:
            logger.warning("Features do Google Trends puladas: %s", e)
            self.caveats["google_trends_pulado"] = str(e)
            return dataset

        trends = trends.sort_values("epi_sunday").reset_index(drop=True).copy()
        cols_lag = [f"{COLUNA_TERMO}_lag_{lag}" for lag in GT_LAGS]
        for lag in GT_LAGS:
            trends[f"{COLUNA_TERMO}_lag_{lag}"] = trends[COLUNA_TERMO].shift(lag)
        dataset = dataset.merge(trends[["epi_sunday", *cols_lag]], on="epi_sunday", how="left")

        maior_lag = max(GT_LAGS)
        dataset["gt_cobertura"] = dataset[f"{COLUNA_TERMO}_lag_{maior_lag}"].notna().astype(float)
        dataset[cols_lag] = dataset[cols_lag].fillna(0.0)
        self.caveats["google_trends_cobertura"] = float(dataset["gt_cobertura"].mean())
        return dataset


def enriquecedores_padrao(contexto: TemporalContext) -> list:
    """Grupos de features do painel completo, na ordem de aplicação do repo antigo."""
    return [
        ClimaPontoUnico(weather_policy=contexto.weather_policy, as_of_date=contexto.as_of_date),
        Vizinhanca(),
        Ovitrampas(),
        ClimaRA(),
        GoogleTrends(contexto),
    ]
