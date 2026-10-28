"""Contratos do painel de casos: início 2020, alvo oficial, RA ausente excluída, lags por RA."""

import json

import pandas as pd
import pytest

import sintetico
from arbo_core import contracts
from arbo_core.contracts import ALVO_PADRAO, ALVO_SENSIBILIDADE, TemporalContext
from arbo_core.manifest import validar_periodo
from arbo_core.panel import construir_painel
from arbo_core.prepare import preparar


@pytest.fixture
def painel(fontes_sinteticas):
    return construir_painel(TemporalContext.retrospective())


def test_painel_comeca_em_2020(painel):
    assert painel["epi_sunday"].min() == sintetico.INICIO
    assert painel["epi_sunday"].min() >= contracts.INICIO_SERIE_ESPACIAL


def test_universo_de_ras_consolidado(painel):
    assert sorted(painel["RA"].unique()) == sorted(sintetico.RAS.values())


def test_alvo_padrao_conta_casos_provaveis(painel, fontes_sinteticas):
    assert set(painel["target_name"]) == {ALVO_PADRAO}
    obtido = painel.set_index(["epi_sunday", "RA"])["cases"]
    for chave, n in fontes_sinteticas.items():
        assert obtido.loc[chave] == n, chave


def test_linha_sem_ra_excluida_nunca_zero(painel, fontes_sinteticas):
    # O registro "Não Informado" e o de outra UF não aparecem em RA nenhuma.
    assert painel["RA"].notna().all()
    assert painel["cases"].sum() == sum(fontes_sinteticas.values())


def test_sensibilidade_familia_dengue_exclui_inconclusivos(fontes_sinteticas):
    familia = construir_painel(TemporalContext.retrospective(target_name=ALVO_SENSIBILIDADE),
                               target_name=ALVO_SENSIBILIDADE)
    simples = construir_painel(TemporalContext.retrospective())
    assert 0 < familia["cases"].sum() < simples["cases"].sum()


def test_lags_por_ra_sem_cruzar_ras(painel):
    for _, serie in painel.sort_values("epi_sunday").groupby("RA"):
        for k in (1, 2, 3, 4):
            esperado = serie["cases"].shift(k)
            pd.testing.assert_series_equal(serie[f"cases_lag_{k}"], esperado, check_names=False)
            assert serie[f"cases_lag_{k}"].iloc[:k].isna().all()


def test_manifesto_rejeita_inicio_antes_de_2020():
    painel = pd.DataFrame({"epi_sunday": [pd.Timestamp("2019-12-29")], "RA": ["GAMA"]})
    with pytest.raises(ValueError, match="INICIO_SERIE_ESPACIAL"):
        validar_periodo(painel)


def test_prepare_gera_painel_e_manifesto(fontes_sinteticas, tmp_path):
    caminho_painel, caminho_manifesto = preparar(saida=tmp_path / "saida", enriquecedores=())
    painel = pd.read_parquet(caminho_painel)
    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))

    assert len(painel) == manifesto["n_linhas"] == sintetico.N_SEMANAS * len(sintetico.RAS)
    assert manifesto["alvo"] == ALVO_PADRAO
    assert manifesto["periodo"]["inicio"] == "2020-01-05"
    assert manifesto["ras"] == sorted(sintetico.RAS.values())
    assert set(manifesto["fontes"]) == {"info-saude", "populacao"}
    for fonte in manifesto["fontes"].values():
        assert fonte["arquivos"] and all(len(h) == 64 for h in fonte["arquivos"].values())
    assert set(manifesto["versoes_contrato"]) == {
        "artifact_schema_version", "forecast_problem_version",
        "temporal_context_version", "quantile_schema_version",
    }
