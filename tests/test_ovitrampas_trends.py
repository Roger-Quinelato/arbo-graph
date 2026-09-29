"""Ovitrampas e Google Trends: lags, cobertura, porta de contrato e degradação."""

import pandas as pd
import pytest

import sintetico
from arbo_core.contracts import ForecastMode, TemporalContext
from arbo_core.etl.trends import carregar_indice_google_trends
from arbo_core.features import GoogleTrends, Ovitrampas, colunas_google_trends, colunas_ovitrampas
from arbo_core.panel import construir_painel


def _painel(enriquecedores, contexto=None):
    return construir_painel(contexto or TemporalContext.retrospective(), enriquecedores=enriquecedores)


def _linha(painel, i, ra):
    return painel.set_index(["epi_sunday", "RA"]).loc[(sintetico.semana(i), ra)]


def test_ovitrampas_lag_e_cobertura(fontes_completas):
    painel = _painel([Ovitrampas()])
    positiva = _linha(painel, 7, "CEILANDIA")  # leituras da semana 5, lag 2
    assert positiva["ovi_ipo_lag_2"] == 0.5
    assert positiva["ovi_ovos_media_lag_2"] == 5.0
    assert positiva["ovi_cobertura"] == 1.0


def test_ovitrampas_zero_observado_difere_de_sem_vigilancia(fontes_completas):
    painel = _painel([Ovitrampas()])
    leitura_zero = _linha(painel, 10, "CEILANDIA")  # semana 8: armadilha negativa
    sem_leitura = _linha(painel, 12, "CEILANDIA")   # semana 10: nenhuma armadilha
    assert leitura_zero["ovi_ipo_lag_2"] == sem_leitura["ovi_ipo_lag_2"] == 0.0
    assert leitura_zero["ovi_cobertura"] == 1.0
    assert sem_leitura["ovi_cobertura"] == 0.0


def test_ovitrampas_alias_brasilia_vira_plano_piloto(fontes_completas):
    painel = _painel([Ovitrampas()])
    assert _linha(painel, 7, "PLANO PILOTO")["ovi_ipo_lag_2"] == 1.0


def test_trends_lags_a_partir_de_1(fontes_completas):
    painel = _painel([GoogleTrends(TemporalContext.retrospective())])
    indice = sintetico.gerar_trends().set_index(pd.to_datetime(sintetico.gerar_trends()["week_start"]))
    linha = _linha(painel, 10, "GAMA")
    for lag in (1, 2, 3, 4):
        assert linha[f"google_trends_dengue_lag_{lag}"] == indice.loc[sintetico.semana(10 - lag), "google_trends_dengue"]
    # Cobertura pelo maior lag: as 4 primeiras semanas não têm o índice completo.
    cobertura = painel.groupby("epi_sunday")["gt_cobertura"].first()
    assert cobertura.iloc[:4].eq(0).all() and cobertura.iloc[4:].eq(1).all()


def test_trends_bloqueado_fora_do_modo_retrospectivo(fontes_completas):
    contexto = TemporalContext.operational(as_of_date="2021-03-07", modo_avaliacao=ForecastMode.ROLLING_DIRECT_K)
    enriquecedor = GoogleTrends(contexto)
    painel = _painel([enriquecedor], contexto)
    assert not set(colunas_google_trends()) & set(painel.columns)
    assert "google_trends_pulado" in enriquecedor.caveats


def test_fontes_ausentes_degradam_sem_colunas(fontes_sinteticas):
    ovi, gt = Ovitrampas(), GoogleTrends(TemporalContext.retrospective())
    painel = _painel([ovi, gt])
    assert not set(colunas_ovitrampas() + colunas_google_trends()) & set(painel.columns)
    assert "ovitrampas_pulado" in ovi.caveats and "google_trends_pulado" in gt.caveats


def test_trends_rejeita_recorte_geografico_errado(fontes_sinteticas, tmp_path):
    sintetico.escrever_trends(tmp_path / "dados", geo="BR-GO")
    with pytest.raises(ValueError, match="recorte geográfico"):
        carregar_indice_google_trends()
