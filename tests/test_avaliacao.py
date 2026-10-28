"""Protocolo de comparação: divisão temporal, identidade, WIS/CRPS e previsão ausente."""

import numpy as np
import pandas as pd
import pytest

from arbo_core.evaluation.identity import TARGET_WEEK_COLUMN, preparar_amostras_horizonte
from arbo_core.evaluation.metrics import (
    CDC_QUANTILES,
    calcular_crps_quantis,
    calcular_wis_completo,
    consolidar_metricas_performance,
    quantile_col,
    validar_monotonicidade_quantis,
)
from arbo_core.evaluation.split import (
    amostras_ate_origem,
    dividir_cv_temporal,
    dividir_treino_teste_temporal,
    origens_expansivas,
)

SEMANAS = pd.date_range("2023-01-01", "2025-12-28", freq="W-SUN")


def _painel(ras=("A", "B")):
    linhas = [{"epi_sunday": d, "RA": ra, "cases": float((i * (j + 2)) % 11)}
              for i, d in enumerate(SEMANAS) for j, ra in enumerate(ras)]
    return pd.DataFrame(linhas)


def test_divisao_treino_teste_sem_sobreposicao():
    treino, teste = dividir_treino_teste_temporal(_painel(), 2025)
    assert treino["epi_sunday"].max() < pd.Timestamp("2025-01-01") <= teste["epi_sunday"].min()
    assert teste["epi_sunday"].dt.year.eq(2025).all()
    assert set(treino.index).isdisjoint(teste.index)


def test_cv_temporal_respeita_gap_de_4_semanas():
    dobras = list(dividir_cv_temporal(len(SEMANAS), n_splits=5, gap=4))
    assert len(dobras) == 5
    inicios = [val[0] for _, val in dobras]
    assert inicios == sorted(inicios)
    for treino, val in dobras:
        assert treino.max() == val.min() - 5  # 4 posições de folga entre treino e validação
        assert set(treino).isdisjoint(val)


def test_cv_temporal_igual_ao_timeseriessplit():
    sklearn = pytest.importorskip("sklearn.model_selection")
    esperado = list(sklearn.TimeSeriesSplit(n_splits=5, gap=4).split(np.arange(len(SEMANAS))))
    obtido = list(dividir_cv_temporal(len(SEMANAS), n_splits=5, gap=4))
    for (tr_e, va_e), (tr_o, va_o) in zip(esperado, obtido):
        assert np.array_equal(tr_e, tr_o) and np.array_equal(va_e, va_o)


def test_origens_crescentes_e_treino_so_com_alvo_observado():
    origens = origens_expansivas(SEMANAS, pd.Timestamp("2024-01-07"), passo_semanas=4)
    assert all(a < b for a, b in zip(origens, origens[1:]))
    amostras = preparar_amostras_horizonte(_painel(), "cases", horizonte=2)
    treino = amostras_ate_origem(amostras, origens[3], gap=4)
    assert pd.to_datetime(treino[TARGET_WEEK_COLUMN]).max() <= origens[3] - pd.Timedelta(weeks=4)


def test_identidade_liga_origem_ao_alvo_da_mesma_ra():
    painel = _painel()
    amostras = preparar_amostras_horizonte(painel, "cases", horizonte=3)
    linha = amostras[amostras["RA"].eq("B")].iloc[0]
    alvo = painel[(painel["RA"] == "B") & (painel["epi_sunday"] == linha[TARGET_WEEK_COLUMN])]
    assert linha[TARGET_WEEK_COLUMN] - linha["forecast_origin"] == pd.Timedelta(weeks=3)
    assert linha["target_k"] == alvo["cases"].iloc[0]


def _previsao_quantis(y, mediana, meia_largura):
    df = pd.DataFrame({"cases": y})
    for tau in CDC_QUANTILES:
        z = (tau - 0.5) * 2  # -1..1
        df[quantile_col(tau)] = mediana + z * meia_largura
    return df


def test_wis_completo_confere_com_calculo_a_mao():
    df = _previsao_quantis([10.0], mediana=10.0, meia_largura=4.0)
    # y na mediana: cada intervalo contribui (alpha/2) × largura; soma / (K+1).
    lower = [t for t in CDC_QUANTILES if t < 0.5]
    esperado = sum((2 * t / 2) * (2 * (1 - 2 * t) * 4.0) for t in lower) / (len(lower) + 1)
    assert calcular_wis_completo(df) == pytest.approx(esperado)


def test_crps_confere_com_pinball_a_mao():
    df = pd.DataFrame({"cases": [3.0], "q_0.250": [2.0], "q_0.500": [4.0], "q_0.750": [5.0]})
    pinball = [0.25 * 1.0, 0.5 * 1.0, 0.25 * 2.0]  # y acima, abaixo, abaixo
    assert calcular_crps_quantis(df) == pytest.approx(2 * np.mean(pinball))


def test_crps_zero_para_previsao_perfeita_degenerada():
    df = _previsao_quantis([7.0, 2.0], mediana=np.array([7.0, 2.0]), meia_largura=0.0)
    assert calcular_crps_quantis(df) == pytest.approx(0.0)


def test_detecta_quantis_fora_de_ordem():
    df = _previsao_quantis([5.0, 5.0], mediana=5.0, meia_largura=2.0)
    df.loc[1, quantile_col(0.9)] = 0.0
    assert validar_monotonicidade_quantis(df) == 1


def _pred(ras=("A", "B")):
    painel = _painel(ras)
    teste = painel[painel["epi_sunday"].dt.year.eq(2025)].copy()
    teste["prediction"] = teste["cases"] + 1.0
    return painel, teste


def test_previsao_ausente_fica_fora_e_reduz_cobertura():
    painel, teste = _pred()
    completo, _ = consolidar_metricas_performance(teste, historico_total=painel)
    ausente = teste.copy()
    ausente.loc[ausente["RA"].eq("B"), "prediction"] = np.nan
    parcial, por_ra = consolidar_metricas_performance(ausente, historico_total=painel)

    assert completo["cobertura_previsao"] == 1.0
    assert parcial["cobertura_previsao"] == pytest.approx(0.5)
    assert parcial["n_previsoes_ausentes"] == int(teste["RA"].eq("B").sum())
    assert set(por_ra["RA"]) == {"A"}
    # O erro do DF não pode ser o de uma previsão zerada para a RA B.
    so_a, _ = consolidar_metricas_performance(teste[teste["RA"].eq("A")], historico_total=painel)
    assert parcial["mae_df"] == pytest.approx(so_a["mae_df"])


def test_metricas_por_ra_e_crps_com_quantis():
    painel, teste = _pred()
    for tau in CDC_QUANTILES:
        teste[quantile_col(tau)] = teste["prediction"] + (tau - 0.5) * 4
    metricas, por_ra = consolidar_metricas_performance(teste, historico_total=painel)
    assert {"mae_ra", "wis_ra", "coverage_ra", "mase_ra"} <= set(por_ra.columns)
    assert np.isfinite(metricas["wis"]) and np.isfinite(metricas["crps"])
    assert metricas["mase_vs_naive"] > 0


def test_previsao_totalmente_ausente_falha():
    _, teste = _pred()
    teste["prediction"] = np.nan
    with pytest.raises(ValueError, match="Nenhuma previsão"):
        consolidar_metricas_performance(teste)
