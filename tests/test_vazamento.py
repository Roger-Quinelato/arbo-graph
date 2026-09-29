"""Vazamento futuro: alterar dados posteriores à origem não muda nada até a origem."""

import pandas as pd
import pytest

import sintetico
from arbo_core.contracts import TemporalContext, WeatherPolicy
from arbo_core.etl.cases import normalizar_casos
from arbo_core.etl.climate import carregar_cache_climatico, carregar_clima_por_ra
from arbo_core.features import ClimaPontoUnico, ClimaRA, Vizinhanca
from arbo_core.panel import montar_painel
from arbo_core.ras import busca_ra_canonica, carregar_historico_populacao

ORIGEM = sintetico.INICIO + pd.Timedelta(weeks=30)
IGNORAR = {"processing_time"}


def _painel(casos, clima, clima_ra):
    contexto = TemporalContext.retrospective()
    return montar_painel(
        normalizar_casos(casos),
        carregar_historico_populacao(),
        contexto,
        lookup=busca_ra_canonica(),
        enriquecedores=[ClimaPontoUnico(clima=clima), Vizinhanca(), ClimaRA(clima_ra=clima_ra)],
    )


def _ate_origem(painel):
    colunas = [c for c in painel.columns if c not in IGNORAR]
    return (painel[painel["epi_sunday"].le(ORIGEM)][colunas]
            .sort_values(["RA", "epi_sunday"]).reset_index(drop=True))


def test_futuro_nao_altera_features_da_origem(fontes_completas):
    casos, _ = sintetico.gerar_casos()
    clima = carregar_cache_climatico(weather_policy=WeatherPolicy.OPERATIONAL_OBSERVED_ONLY)
    clima_ra = carregar_clima_por_ra()
    base = _painel(casos, clima, clima_ra)

    futuro_casos = casos[pd.to_datetime(casos["i_data_prim_sintomas"]) > ORIGEM + pd.Timedelta(days=6)]
    casos_alt = pd.concat([casos, futuro_casos, futuro_casos], ignore_index=True)
    clima_alt = clima.copy()
    clima_alt.loc[clima_alt["epi_sunday"] > ORIGEM, ["temp_mean", "precip_sum", "umidmed"]] *= 3
    clima_ra_alt = clima_ra.copy()
    clima_ra_alt.loc[clima_ra_alt["epi_sunday"] > ORIGEM, ["temp_mean_ra", "precip_sum_ra", "umid_med_ra"]] += 10
    alterado = _painel(casos_alt, clima_alt, clima_ra_alt)

    # Sanidade: a alteração existe depois da origem.
    depois = alterado["epi_sunday"] > ORIGEM
    assert not alterado.loc[depois, "cases"].equals(base.loc[base["epi_sunday"] > ORIGEM, "cases"])
    assert "viz_cases_lag_1" in base.columns
    pd.testing.assert_frame_equal(_ate_origem(base), _ate_origem(alterado))


def _umidade_na_lacuna(politica, tmp_path, multiplicar_futuro):
    info_path = tmp_path / "dados" / "InfoDengue" / "InfoDengue_2016-2026.csv"
    info = pd.read_csv(info_path)
    lacuna = info["umidmed"].isna()
    primeira_lacuna = info.index[lacuna][0]
    if multiplicar_futuro:
        info.loc[primeira_lacuna + 3:, "umidmed"] *= 2
        info.to_csv(info_path, index=False)
    clima = carregar_cache_climatico(weather_policy=politica, as_of_date=pd.Timestamp("2021-06-01"))
    semana = pd.to_datetime(info.loc[primeira_lacuna, "data_iniSE"])
    return clima.set_index("epi_sunday").loc[semana, "umidmed"]


def test_politica_operacional_nao_usa_umidade_futura(fontes_completas, tmp_path):
    politica = WeatherPolicy.OPERATIONAL_OBSERVED_ONLY
    antes = _umidade_na_lacuna(politica, tmp_path, multiplicar_futuro=False)
    depois = _umidade_na_lacuna(politica, tmp_path, multiplicar_futuro=True)
    assert antes == depois


def test_limitacao_documentada_interpolacao_retrospectiva_usa_futuro(fontes_completas, tmp_path):
    """Herdado do repo antigo: RETROSPECTIVE_INTERPOLATION preenche lacunas com o valor seguinte."""
    politica = WeatherPolicy.RETROSPECTIVE_INTERPOLATION
    antes = _umidade_na_lacuna(politica, tmp_path, multiplicar_futuro=False)
    depois = _umidade_na_lacuna(politica, tmp_path, multiplicar_futuro=True)
    assert antes != pytest.approx(depois)
