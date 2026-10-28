"""Clima: ETL do INMET com IDW, ausência como NaN, caveat de 15 km e lags por RA."""

import json

import numpy as np
import pandas as pd
import pytest

import sintetico
from arbo_core.contracts import TemporalContext, WeatherPolicy
from arbo_core.etl.climate import carregar_cache_climatico, carregar_clima_por_ra
from arbo_core.features import LAG_WINDOWS, colunas_clima_ponto_unico, colunas_clima_ra, enriquecedores_padrao
from arbo_core.panel import construir_painel
from arbo_core.prepare import preparar


def test_inmet_idw_por_ra(fontes_completas):
    clima = carregar_clima_por_ra().set_index(["epi_sunday", "RA"])
    semana = pd.Timestamp("2020-02-02")
    # Ceilândia fica sobre a A001: o IDW devolve praticamente a leitura dela. O centroide é a
    # média dos vértices (o de fechamento conta duas vezes), então não cai exatamente no centro.
    assert clima.loc[(semana, "CEILANDIA"), "temp_mean_ra"] == pytest.approx(25.0, abs=0.1)
    # Plano Piloto, longe das duas, recebe um valor entre as leituras.
    assert 15.0 < clima.loc[(semana, "PLANO PILOTO"), "temp_mean_ra"] < 25.0
    assert clima.loc[(semana, "GAMA"), "n_estacoes"] == 2


def test_inmet_ausencia_nao_vira_zero(fontes_completas):
    clima = carregar_clima_por_ra().set_index(["epi_sunday", "RA"])
    linha = clima.loc[(sintetico.SEMANA_SEM_A042, "GAMA")]
    # Sem a A042 na semana, só a A001 entra na ponderação: nada de zero puxando a média.
    assert linha["n_estacoes"] == 1
    assert linha["temp_mean_ra"] == pytest.approx(25.0)


def test_inmet_lista_ras_distantes(fontes_completas):
    distantes = dict(carregar_clima_por_ra().attrs["ras_distantes_km"])
    assert set(distantes) == {"PLANO PILOTO"}
    assert distantes["PLANO PILOTO"] > 15


def test_painel_com_clima_lags_por_ra(fontes_completas):
    painel = construir_painel(TemporalContext.retrospective(),
                              enriquecedores=enriquecedores_padrao(TemporalContext.retrospective()))
    assert set(colunas_clima_ponto_unico() + colunas_clima_ra()) <= set(painel.columns)
    for coluna in ("temp_mean_ra", "precip_sum_ra", "umid_med_ra"):
        assert coluna not in painel.columns  # contemporâneas descartadas
    for _, serie in painel.sort_values("epi_sunday").groupby("RA"):
        assert serie["temp_mean_ra_lag_2"].iloc[:2].isna().all()  # shift não cruza RAs


def test_ponto_unico_usa_clima_anterior_a_2020(fontes_completas):
    painel = construir_painel(TemporalContext.retrospective(),
                              enriquecedores=enriquecedores_padrao(TemporalContext.retrospective()))
    cache, _ = sintetico.gerar_clima_cache()
    primeira = painel[painel["epi_sunday"].eq(sintetico.INICIO)].iloc[0]
    esperado = cache.set_index("epi_sunday").loc[sintetico.INICIO - pd.Timedelta(weeks=2), "temp_mean"]
    assert primeira["temp_mean_lag_2"] == esperado
    assert max(LAG_WINDOWS["precip_sum"]) == 11


def test_manifesto_registra_caveat_de_distancia(fontes_completas, tmp_path):
    _, caminho_manifesto = preparar(saida=tmp_path / "saida")
    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    assert [ra for ra, _ in manifesto["caveats"]["clima_ra_ras_distantes_km"]] == ["PLANO PILOTO"]
    assert {"clima_cache", "InfoDengue", "inmet", "dados_geo"} <= set(manifesto["fontes"])


def test_clima_ra_ausente_degrada_com_caveat(fontes_sinteticas, tmp_path):
    sintetico.escrever_clima_cache(tmp_path / "dados")
    _, caminho_manifesto = preparar(saida=tmp_path / "saida")
    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    assert "clima_ra_pulado" in manifesto["caveats"]
    assert "inmet" in manifesto["caveats"]["fontes_ausentes"]


@pytest.mark.parametrize("politica", list(WeatherPolicy))
def test_umidade_sem_semanas_vazias(fontes_completas, politica):
    clima = carregar_cache_climatico(weather_policy=politica, as_of_date=pd.Timestamp("2021-06-01"))
    assert clima["umidmed"].notna().all()
    assert np.isfinite(clima["temp_mean"]).all()
