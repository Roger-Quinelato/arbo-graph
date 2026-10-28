"""Baseline e0: horizontes 1–4, quantis no XGB, cobertura explícita, reprodutível, sem futuro."""

import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

import sintetico
from arbo_core.contracts import TemporalContext
from arbo_core.experiments import BASELINE, carregar_experimento, executar_experimento
from arbo_core.features import enriquecedores_padrao
from arbo_core.panel import construir_painel

# Painel sintético: 2020 inteiro para treino e as semanas de 2021 como teste.
CONFIG_TESTE = {BASELINE: {"ano_teste": 2021}}
PERMITIDAS = re.compile(r"(_lag_\d+$|_cobertura$|^cases_delta_\d$|^cases_growth_rate$|^(sin|cos)_(week|month)$|^RA$)")


@pytest.fixture
def painel(fontes_completas):
    contexto = TemporalContext.retrospective()
    return construir_painel(contexto, enriquecedores=enriquecedores_padrao(contexto))


def _rodar(painel, tmp_path, **kwargs):
    destino = executar_experimento(BASELINE, painel, saida=tmp_path / "res", sobrescrever_config=CONFIG_TESTE, **kwargs)
    return destino, pd.read_csv(destino / "metricas.csv")


def test_e0_roda_modelos_modos_e_horizontes(painel, tmp_path):
    _, metricas = _rodar(painel, tmp_path)
    combinacoes = set(zip(metricas["modelo"], metricas["modo"], metricas["horizonte"]))
    esperado = {(m, "nowcast", 1) for m in ("RF", "XGB")} | {(m, "direto", k) for m in ("RF", "XGB") for k in (1, 2, 3, 4)}
    assert combinacoes == esperado
    direto_xgb = metricas[(metricas["modelo"] == "XGB") & (metricas["modo"] == "direto")]
    assert np.isfinite(direto_xgb["wis"]).all() and np.isfinite(direto_xgb["crps"]).all()
    assert metricas.loc[metricas["modelo"] == "RF", "wis"].isna().all()


def test_ra_sem_vizinho_fica_sem_previsao_e_reduz_cobertura(painel, tmp_path):
    destino, metricas = _rodar(painel, tmp_path)
    # Plano Piloto não tem vizinho: viz_cases_lag é NaN e a linha não tem previsão.
    assert metricas["cobertura_previsao"].eq(0.75).all()
    previsoes = pd.read_parquet(destino / "previsoes.parquet")
    assert previsoes.loc[previsoes["RA"].eq("PLANO PILOTO"), "prediction"].isna().all()


def test_e0_reprodutivel(painel, tmp_path):
    _, a = _rodar(painel, tmp_path / "a")
    _, b = _rodar(painel, tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_features_do_e0_sao_so_defasadas():
    modulo, cfg = carregar_experimento(BASELINE)
    colunas = modulo.colunas_entrada(modulo.especificacao_features(cfg["features"]))
    proibidas = [c for c in colunas if not PERMITIDAS.search(c)]
    assert proibidas == []
    assert not {"cases", "temp_mean", "precip_sum", "umidmed", "temp_mean_ra"} & set(colunas)


def test_resultado_md_gerado_e_publicacao_opcional(painel, tmp_path):
    destino, _ = _rodar(painel, tmp_path)
    texto = (destino / "RESULTADO.md").read_text(encoding="utf-8")
    assert "Gerado automaticamente" in texto and "Não editar à mão" in texto
    assert "| experimento | modelo | modo | horizonte |" in texto
    assert "não é resultado nosso" in texto


def test_outro_experimento_roda_baseline_junto(painel, tmp_path):
    pasta = tmp_path / "exps" / "e_naive"
    pasta.mkdir(parents=True)
    (pasta / "config.toml").write_text('seed = 42\n', encoding="utf-8")
    (pasta / "model.py").write_text(
        "import pandas as pd\n"
        "from arbo_core.evaluation.metrics import consolidar_metricas_performance\n"
        "def executar(painel, config):\n"
        "    teste = painel[painel['epi_sunday'].dt.year.eq(2021)].copy()\n"
        "    teste['prediction'] = teste['cases_lag_1']\n"
        "    m, _ = consolidar_metricas_performance(teste, historico_total=painel[['epi_sunday','RA','cases']])\n"
        "    return pd.DataFrame([{'modelo': 'naive', 'modo': 'nowcast', 'horizonte': 1, **m}]), teste\n",
        encoding="utf-8",
    )
    destino = executar_experimento("e_naive", painel, saida=tmp_path / "res", diretorio=tmp_path / "exps",
                                   sobrescrever_config=CONFIG_TESTE, publicar=True)
    metricas = pd.read_csv(destino / "metricas.csv")
    assert set(metricas["experimento"]) == {"e_naive", BASELINE}
    assert (pasta / "RESULTADO.md").exists()
