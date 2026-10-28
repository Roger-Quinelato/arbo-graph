"""Equivalência com o pipeline antigo: comparadores e execução ponta a ponta em dados sintéticos."""

import json

import numpy as np
import pandas as pd
import pytest

from arbo_core.equivalence import REPO_ANTIGO_PADRAO, comparar_metricas, comparar_paineis


def _painel():
    return pd.DataFrame({
        "epi_sunday": pd.to_datetime(["2020-01-05", "2020-01-05"]), "RA": ["A", "B"],
        "cases": [1.0, 2.0], "cases_lag_1": [np.nan, 3.0], "processing_time": ["x", "y"],
    })


def test_paineis_iguais_ignoram_carimbo_de_execucao():
    a, b = _painel(), _painel()
    b["processing_time"] = ["outro", "outro"]
    assert comparar_paineis(a, b)["equivalente"]


def test_paineis_detectam_valor_divergente_e_coluna_faltante():
    a, b = _painel(), _painel()
    b.loc[1, "cases_lag_1"] = 4.0
    rel = comparar_paineis(a, b.drop(columns="cases"))
    assert not rel["equivalente"]
    assert "cases_lag_1" in rel["colunas_divergentes"]
    assert rel["colunas_so_antigo"] == ["cases"]


def test_colunas_do_censo_sao_diferenca_esperada():
    a = _painel().assign(pct_rede_agua=0.9, saneamento_cobertura=1.0)
    rel = comparar_paineis(a, _painel())
    assert rel["equivalente"]
    assert rel["colunas_fora_por_decisao"] == ["pct_rede_agua", "saneamento_cobertura"]


def test_metricas_detectam_divergencia_e_combinacao_faltante():
    antigo = pd.DataFrame([{"modelo": "RF", "modo": "direto", "horizonte": 1, "mae_df": 2.0},
                           {"modelo": "RF", "modo": "direto", "horizonte": 2, "mae_df": 3.0}])
    novo = pd.DataFrame([{"modelo": "RF", "modo": "direto", "horizonte": 1, "mae_df": 2.5}])
    rel = comparar_metricas(antigo, novo)
    assert not rel["equivalente"]
    assert rel["divergentes"][0]["metrica"] == "mae_df"
    assert rel["combinacoes_faltantes"] == [{"modelo": "RF", "modo": "direto", "horizonte": 2}]


def test_equivalencia_ponta_a_ponta_com_pipeline_antigo(fontes_conectadas, tmp_path):
    pytest.importorskip("hierarchicalforecast")
    pytest.importorskip("xgboost")
    if not (REPO_ANTIGO_PADRAO / "src" / "dengue_pipeline").exists():
        pytest.skip("Repo antigo não encontrado (defina ARBO_REPO_ANTIGO ou clone-o como pasta irmã).")
    from arbo_core.equivalence import verificar

    relatorio = verificar(tmp_path / "saida", ano_teste=2021)
    salvo = json.loads((tmp_path / "saida" / "equivalencia.json").read_text(encoding="utf-8"))
    assert relatorio.equivalente, salvo
    assert relatorio.painel["mesmas_linhas"] and not relatorio.painel["colunas_divergentes"]
    assert not relatorio.metricas["divergentes"]
