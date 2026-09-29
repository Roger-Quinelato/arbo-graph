# -*- coding: utf-8 -*-
"""
Métricas de avaliação pontuais e probabilísticas.

Origem: ``dengue_pipeline/modeling/evaluation.py`` (commit 402f845). Mudanças:
- sem dependência de scikit-learn (MAE, RMSE e R² em NumPy, mesmas fórmulas);
- ``calcular_crps_quantis``: CRPS aproximado pela média do pinball loss nos quantis;
- ``consolidar_metricas_performance`` não soma previsão ausente como zero: linhas sem
  previsão saem das métricas e entram em ``cobertura_previsao``.

Cálculo puro a partir de DataFrames prontos; não treina nem grava arquivos. ``calcular_wis``
usa um intervalo (legado); com quantis, ``calcular_wis_completo`` segue a fórmula do CDC
Forecast Hub com 23 quantis.
"""

import logging

import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)

logger = logging.getLogger(__name__)


CDC_QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.95, 0.975, 0.99,
]


def quantile_col(tau: float) -> str:
    """Nome canônico de coluna de quantil do TDD-11."""
    return f"q_{float(tau):.3f}"


def resolver_coluna_quantil(df: pd.DataFrame, tau: float) -> str | None:
    """Resolve nomes legados e canônicos para um quantil."""
    candidates = [
        quantile_col(tau),
        f"q_{float(tau)}",
        f"q_{float(tau):.2f}",
        f"q_{float(tau):.1f}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def colunas_quantis(df: pd.DataFrame) -> list[tuple[float, str]]:
    pairs: list[tuple[float, str]] = []
    for col in df.columns:
        if not col.startswith("q_"):
            continue
        try:
            pairs.append((float(col[2:]), col))
        except ValueError:
            continue
    return sorted(pairs)


def normalizar_colunas_quantis(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona aliases canônicos `q_0.100` sem remover colunas legadas."""
    out = df.copy()
    for tau, col in colunas_quantis(out):
        canonical = quantile_col(tau)
        if canonical not in out.columns:
            out[canonical] = out[col]
    return out


def validar_monotonicidade_quantis(pred_df: pd.DataFrame) -> int:
    """Retorna o número de linhas com quantis fora de ordem."""
    q_pairs = colunas_quantis(pred_df)
    if len(q_pairs) < 2:
        return 0
    cols = [col for _, col in q_pairs]
    values = pred_df[cols].to_numpy(dtype=float)
    valid_rows = ~np.isnan(values).all(axis=1)
    if not valid_rows.any():
        return 0
    diffs = np.diff(values[valid_rows], axis=1)
    return int((diffs < -1e-9).any(axis=1).sum())


def calcular_cobertura_quantil(pred_df: pd.DataFrame, tau: float) -> float:
    col = resolver_coluna_quantil(pred_df, tau)
    if col is None or "cases" not in pred_df.columns:
        return float("nan")
    df = pred_df.dropna(subset=["cases", col])
    if df.empty:
        return float("nan")
    return float((df["cases"].astype(float) <= df[col].astype(float)).mean())


def calcular_cobertura_intervalo_quantil(
    pred_df: pd.DataFrame,
    alpha: float = 0.10,
) -> tuple[float, float]:
    lower_tau = round(alpha / 2.0, 3)
    upper_tau = round(1.0 - alpha / 2.0, 3)
    lower_col = resolver_coluna_quantil(pred_df, lower_tau)
    upper_col = resolver_coluna_quantil(pred_df, upper_tau)
    alpha_eff = alpha

    if lower_col is None or upper_col is None:
        available = [tau for tau, _ in colunas_quantis(pred_df) if tau < 0.50 and resolver_coluna_quantil(pred_df, 1.0 - tau)]
        if not available:
            return float("nan"), alpha
        lower_tau = min(available)
        upper_tau = round(1.0 - lower_tau, 3)
        lower_col = resolver_coluna_quantil(pred_df, lower_tau)
        upper_col = resolver_coluna_quantil(pred_df, upper_tau)
        alpha_eff = round(2.0 * lower_tau, 3)

    df = pred_df.dropna(subset=["cases", lower_col, upper_col])
    if df.empty:
        return float("nan"), alpha_eff
    dentro = (df["cases"] >= df[lower_col]) & (df["cases"] <= df[upper_col])
    return float(dentro.mean()), alpha_eff


def calcular_r2_robusto(y_true, y_pred) -> float:
    """
    Calcula o coeficiente de determinação (R²) de forma segura, tratando casos com variância
    nula ou número insuficiente de amostras retornando NaN ao invés de erro.
    
    Parâmetros:
        y_true: Valores reais.
        y_pred: Valores previstos.
        
    Retorna:
        float: Valor de R² ou float("nan").
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def calcular_erro_quadratico_medio(y_true, y_pred) -> float:
    """
    Calcula a raiz do erro quadrático médio (RMSE).
    
    Parâmetros:
        y_true: Valores reais.
        y_pred: Valores previstos.
        
    Retorna:
        float: RMSE.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def avaliar_cobertura_intervalo(df_avaliado: pd.DataFrame) -> float:
    """
    Calcula a cobertura empírica real (coverage_score).
    
    Verifica a proporção de amostras cujo valor real de 'cases' caiu dentro
    dos limites [lower_ci, upper_ci] das bandas conformalizadas.
    
    Parâmetros:
        df_avaliado (pd.DataFrame): DataFrame contendo 'cases', 'lower_ci' e 'upper_ci'.
        
    Retorna:
        float: Proporção de amostras cobertas (entre 0.0 e 1.0). Retorna NaN se
               não houver amostras válidas.
    """
    required = {"cases", "lower_ci", "upper_ci"}
    if not required.issubset(df_avaliado.columns):
        raise ValueError(f"DataFrame deve conter as colunas {required}. Encontradas: {set(df_avaliado.columns)}")
    
    df = df_avaliado.dropna(subset=["cases", "lower_ci", "upper_ci"])
    if df.empty:
        return float("nan")
    
    dentro = (df["cases"] >= df["lower_ci"]) & (df["cases"] <= df["upper_ci"])
    return float(dentro.mean())


# ---------------------------------------------------------------------------
# Métricas Probabilísticas — TDD-08
# ---------------------------------------------------------------------------

def calcular_cobertura_intervalo(pred_df: pd.DataFrame, alpha: float = 0.10) -> float:
    """
    Calcula a cobertura empírica do intervalo de confiança.

    Proporção de observações reais (``cases``) que caem dentro do intervalo
    ``[lower_ci, upper_ci]``.  Para um intervalo declarado de (1 - alpha) × 100 %,
    o valor ideal de cobertura é ``1 - alpha``.

    Parâmetros:
        pred_df (pd.DataFrame): DataFrame com colunas ``cases``, ``lower_ci``,
            ``upper_ci``.
        alpha (float): Nível de significância do intervalo (default 0.10 → 90 % CI).

    Retorna:
        float: Proporção em [0, 1].  ``NaN`` quando as colunas de CI estão
            ausentes ou contêm apenas ``NaN``.
    """
    ci_cols = {"lower_ci", "upper_ci", "cases"}
    if not ci_cols.issubset(pred_df.columns):
        logger.warning(
            "Colunas de CI ausentes (%s); cobertura não calculada.",
            ci_cols - set(pred_df.columns),
        )
        return float("nan")

    df = pred_df.dropna(subset=["cases", "lower_ci", "upper_ci"])
    if df.empty:
        logger.warning("Todas as linhas de CI são NaN; cobertura retornando NaN.")
        return float("nan")

    dentro = (df["cases"] >= df["lower_ci"]) & (df["cases"] <= df["upper_ci"])
    return float(dentro.mean())


def calcular_wis(pred_df: pd.DataFrame, alpha: float = 0.10) -> float:
    """
    Calcula o Weighted Interval Score (WIS) simplificado com 1 intervalo.

    Decomposição (para cada observação):
        - ``spread``     = ``upper_ci - lower_ci``    (penalidade de amplitude)
        - ``undershoot`` = ``(2/α) × max(0, lower_ci - cases)``  (saída inferior)
        - ``overshoot``  = ``(2/α) × max(0, cases - upper_ci)``  (saída superior)
        - ``WIS``        = ``mean(spread + undershoot + overshoot)``

    **Limitação documentada:** esta é a versão de 1 intervalo.  O WIS completo
    do CDC Forecast Hub utiliza 23 quantis; a versão aqui é uma aproximação
    válida para validação interna.

    Parâmetros:
        pred_df (pd.DataFrame): DataFrame com colunas ``cases``, ``lower_ci``,
            ``upper_ci``.
        alpha (float): Nível de significância (default 0.10 → 90 % CI).

    Retorna:
        float: Score não-negativo (menor é melhor).  ``NaN`` quando CI ausente.
    """
    ci_cols = {"lower_ci", "upper_ci", "cases"}
    if not ci_cols.issubset(pred_df.columns):
        logger.warning(
            "Colunas de CI ausentes (%s); WIS não calculado.",
            ci_cols - set(pred_df.columns),
        )
        return float("nan")

    df = pred_df.dropna(subset=["cases", "lower_ci", "upper_ci"])
    if df.empty:
        logger.warning("Todas as linhas de CI são NaN; WIS retornando NaN.")
        return float("nan")

    cases = df["cases"].values.astype(float)
    lower = df["lower_ci"].values.astype(float)
    upper = df["upper_ci"].values.astype(float)

    spread = upper - lower
    undershoot = (2.0 / alpha) * np.maximum(0.0, lower - cases)
    overshoot = (2.0 / alpha) * np.maximum(0.0, cases - upper)

    wis = float(np.mean(spread + undershoot + overshoot))
    return wis


def calcular_wis_completo(pred_df: pd.DataFrame, alpha_grid: list[float] | None = None) -> float:
    """
    Calcula o Weighted Interval Score (WIS) Completo segundo a fórmula do CDC Forecast Hub.

    Parâmetros:
        pred_df (pd.DataFrame): DataFrame com colunas 'cases' e colunas para os quantis 
            (ex: 'q_0.01', 'q_0.025', ..., 'q_0.99').
        alpha_grid (list[float]): Lista de taus.

    Retorna:
        float: WIS Score médio, ou NaN se faltarem colunas de quantis.
    """
    if alpha_grid is None:
        alpha_grid = CDC_QUANTILES

    pred_df = normalizar_colunas_quantis(pred_df)

    df = pred_df.dropna(subset=["cases"])
    if df.empty:
        return float("nan")

    cases = df["cases"].values.astype(float)

    # 1. Componente da Mediana
    col_mediana = resolver_coluna_quantil(df, 0.50)
            
    if not col_mediana:
        logger.warning("Coluna da mediana ausente; WIS Completo retornando NaN.")
        return float("nan")

    m = df[col_mediana].values.astype(float)
    wis = 0.5 * np.abs(cases - m)

    # 2. Componentes dos K intervalos
    lower_taus = [t for t in alpha_grid if t < 0.50]
    K = len(lower_taus)

    for tau in lower_taus:
        tau_upper = round(1.0 - tau, 3)
        alpha_j = round(2.0 * tau, 3)

        col_lower = resolver_coluna_quantil(df, tau)
        col_upper = resolver_coluna_quantil(df, tau_upper)
        if col_lower is None or col_upper is None:
            logger.warning("Colunas de quantil %.3f ou %.3f ausentes; WIS Completo retornando NaN.", tau, tau_upper)
            return float("nan")

        l_j = df[col_lower].values.astype(float)
        u_j = df[col_upper].values.astype(float)

        dispersion = u_j - l_j
        undershoot = (2.0 / alpha_j) * np.maximum(0.0, l_j - cases)
        overshoot = (2.0 / alpha_j) * np.maximum(0.0, cases - u_j)

        interval_score = (alpha_j / 2.0) * (dispersion + undershoot + overshoot)
        wis += interval_score

    wis = wis / (K + 1)
    return float(np.mean(wis))



def calcular_crps_quantis(pred_df: pd.DataFrame, taus: list[float] | None = None) -> float:
    """
    CRPS aproximado a partir de quantis: 2 × média, nos quantis, do pinball loss.

    Com a grade densa do CDC (23 quantis) a aproximação é próxima do CRPS contínuo. Linhas
    sem ``cases`` saem da média. Devolve NaN se faltar alguma coluna de quantil pedida.
    """
    pred_df = normalizar_colunas_quantis(pred_df)
    taus = taus or [tau for tau, _ in colunas_quantis(pred_df)]
    if not taus:
        return float("nan")
    df = pred_df.dropna(subset=["cases"])
    if df.empty:
        return float("nan")
    y = df["cases"].to_numpy(dtype=float)
    perdas = []
    for tau in taus:
        col = resolver_coluna_quantil(df, tau)
        if col is None:
            return float("nan")
        q = df[col].to_numpy(dtype=float)
        erro = y - q
        perdas.append(np.maximum(tau * erro, (tau - 1.0) * erro))
    return float(2.0 * np.mean(np.vstack(perdas)))


def calcular_calibration_error(coverage_real: float, alpha: float = 0.10) -> float:
    """
    Calcula o erro de calibração: diferença absoluta entre cobertura
    declarada ``(1 - alpha)`` e cobertura real observada.

    Parâmetros:
        coverage_real (float): Cobertura empírica calculada via
            ``calcular_cobertura_intervalo``.
        alpha (float): Nível de significância (default 0.10 → meta 90 %).

    Retorna:
        float: Erro em [0, 1]; zero é ideal.  ``NaN`` se ``coverage_real``
            for ``NaN``.
    """
    if np.isnan(coverage_real):
        return float("nan")
    return float(abs(coverage_real - (1.0 - alpha)))


def _mae_seguro(y_true: pd.Series, y_pred: pd.Series) -> float:
    frame = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if frame.empty:
        return float("nan")
    return float(mean_absolute_error(frame["y"], frame["pred"]))


def calcular_mase(y_true: pd.Series, y_pred: pd.Series, baseline_pred: pd.Series) -> float:
    mae_modelo = _mae_seguro(y_true, y_pred)
    mae_baseline = _mae_seguro(y_true, baseline_pred)
    if np.isnan(mae_modelo) or np.isnan(mae_baseline) or np.isclose(mae_baseline, 0.0):
        return float("nan")
    return float(mae_modelo / mae_baseline)


def calcular_baselines_df(
    df_total: pd.DataFrame,
    historico_total: pd.DataFrame | None = None,
    horizonte: int = 1,
) -> dict[str, float]:
    """
    Calcula MAE de baselines e MASE do modelo sobre a série agregada do DF.

    Parâmetros:
        df_total (pd.DataFrame): Série de teste agregada no nível DF, com
            ``epi_sunday``, ``cases`` e ``prediction``.
        horizonte (int): Passos à frente que o modelo prevê. Governa a
            defasagem dos baselines de persistência — ver nota abaixo.
        historico_total (pd.DataFrame, opcional): Série agregada no nível DF
            cobrindo treino+teste (``epi_sunday``, ``cases``). Quando fornecida,
            os baselines sazonais (``seasonal_naive``, ``historical_seasonal_mean``)
            são calculados sobre o histórico completo e fatiados para as datas
            de teste — sem ela, ``shift(52)`` sobre ~52 semanas de teste é
            integralmente NaN. Ambos os baselines usam apenas valores
            estritamente anteriores a cada data (sem leakage).
    """
    frame = df_total.sort_values("epi_sunday").copy()
    if frame.empty:
        return {}

    y = frame["cases"].astype(float)
    pred = frame["prediction"].astype(float)
    week = pd.to_datetime(frame["epi_sunday"]).dt.isocalendar().week.astype(int)

    mae_seasonal_insample = float("nan")
    if historico_total is not None and not historico_total.empty:
        base = historico_total.copy()
        base["epi_sunday"] = pd.to_datetime(base["epi_sunday"])
        base = (
            base.groupby("epi_sunday", as_index=False)["cases"]
            .sum()
            .sort_values("epi_sunday")
        )
        y_hist = base["cases"].astype(float)
        week_hist = base["epi_sunday"].dt.isocalendar().week.astype(int)
        sn_full = y_hist.shift(52)
        hsm_full = y_hist.groupby(week_hist).transform(lambda s: s.shift(1).expanding().mean())

        datas_teste = pd.to_datetime(frame["epi_sunday"])
        seasonal_naive = pd.Series(
            pd.Series(sn_full.values, index=base["epi_sunday"]).reindex(datas_teste).values,
            index=y.index,
        )
        historical_seasonal_mean = pd.Series(
            pd.Series(hsm_full.values, index=base["epi_sunday"]).reindex(datas_teste).values,
            index=y.index,
        )

        # Denominador MASE textbook (Hyndman & Koehler 2006): erro do seasonal naive
        # IN-SAMPLE (período de TREINO), estável e independente da patologia do ano de
        # teste. Sem isto, o surto de 2024 faz o seasonal naive do teste explodir
        # (MAE ~3000) e o MASE vira ~0,004 — verdadeiro, porém ININFORMATIVO. As datas
        # de teste são excluídas do conjunto in-sample (estritamente retrospectivo).
        datas_teste_set = set(datas_teste)
        mask_insample = ~base["epi_sunday"].isin(datas_teste_set)
        erro_sn_insample = (y_hist - sn_full).abs()[mask_insample]
        if erro_sn_insample.notna().any():
            mae_seasonal_insample = float(erro_sn_insample.dropna().mean())
    else:
        if historico_total is None:
            logger.warning(
                "calcular_baselines_df sem historico_total: baselines sazonais "
                "limitados à janela de teste (provável NaN para ~52 semanas)."
            )
        seasonal_naive = y.shift(52)
        historical_seasonal_mean = y.groupby(week).transform(lambda s: s.shift(1).expanding().mean())

    # Baselines de persistência alinhados ao HORIZONTE (correção TDD-13).
    #
    # A série ``y`` está indexada pela semana-alvo. Uma previsão de horizonte k
    # feita na semana ``w - k`` só pode usar o que era observável naquele
    # momento: o último valor conhecido é ``y_{w-k}``, ou seja ``y.shift(k)``.
    # Usar ``y.shift(1)`` para todo k dava ao baseline acesso a k-1 semanas de
    # dados indisponíveis ao modelo, penalizando artificialmente os horizontes
    # longos — era o que fazia ``mase_vs_naive`` subir de 1,9 (k=1) para 4,1
    # (k=4) sem que o modelo tivesse piorado nessa proporção.
    #
    # O drift generaliza do mesmo modo: extrapola k passos a partir da última
    # inclinação observável. Os baselines sazonais (``seasonal_naive`` com
    # defasagem de 52 semanas e ``historical_seasonal_mean``) são independentes
    # do horizonte para k ≤ 4 e ficam inalterados.
    h = max(int(horizonte), 1)
    persistencia = y.shift(h)
    baselines = {
        "naive": persistencia,
        "seasonal_naive": seasonal_naive,
        "drift": persistencia + h * (persistencia - y.shift(h + 1)),
        "historical_seasonal_mean": historical_seasonal_mean,
    }

    metrics: dict[str, float] = {}
    for name, baseline in baselines.items():
        metrics[f"baseline_mae_{name}"] = _mae_seguro(y, baseline)
        metrics[f"mase_vs_{name}"] = calcular_mase(y, pred, baseline)

    model_mae = _mae_seguro(y, pred)

    # `baseline_mae_seasonal_naive` (acima) é o erro OUT-OF-SAMPLE no período de teste,
    # mantido como DIAGNÓSTICO. O MASE sazonal (`mase_vs_seasonal_naive`) usa o
    # denominador IN-SAMPLE (Hyndman & Koehler).
    metrics["baseline_horizonte"] = h
    metrics["baseline_mae_seasonal_naive_insample"] = mae_seasonal_insample
    if not np.isnan(mae_seasonal_insample) and not np.isclose(mae_seasonal_insample, 0.0):
        metrics["mase_vs_seasonal_naive"] = (
            float("nan") if np.isnan(model_mae) else float(model_mae / mae_seasonal_insample)
        )
    else:
        metrics["mase_vs_seasonal_naive"] = float("nan")

    # Diagnóstico adicional: denominador sazonal restrito ao período INTEREPIDÊMICO
    # in-sample (semanas < P75 do histórico). O denominador irrestrito é inflado pelo
    # supersurto de 2024 (erro sazonal ~milhares de casos), o que torna o MASE sazonal
    # incapaz de reprovar qualquer modelo.
    metrics["mase_vs_seasonal_naive_interepidemic"] = float("nan")
    if historico_total is not None and not historico_total.empty:
        limiar_epi = y_hist.quantile(0.75)
        mask_inter = mask_insample & (y_hist < limiar_epi)
        erro_sn_inter = (y_hist - sn_full).abs()[mask_inter]
        if erro_sn_inter.notna().any():
            mae_sn_inter = float(erro_sn_inter.dropna().mean())
            if not np.isclose(mae_sn_inter, 0.0) and not np.isnan(model_mae):
                metrics["mase_vs_seasonal_naive_interepidemic"] = float(model_mae / mae_sn_inter)

    # HEADLINE (`mase_df`): MASE vs NAIVE lag-1 no período de teste. O naive de 1
    # passo é o benchmark honesto do nowcasting: bem-comportado, computado no mesmo
    # período das predições e capaz de REPROVAR um modelo (o denominador sazonal
    # in-sample, inflado pelo surto de 2024, aprovava qualquer coisa). Os MASE
    # sazonais seguem expostos como diagnóstico.
    metrics["mase_df"] = metrics.get("mase_vs_naive", float("nan"))
    mae_naive = metrics.get("baseline_mae_naive", float("nan"))
    metrics["model_beats_naive"] = bool(
        not np.isnan(model_mae)
        and not np.isnan(mae_naive)
        and model_mae < mae_naive
    )
    metrics["model_beats_seasonal_naive"] = bool(
        not np.isnan(model_mae)
        and not np.isnan(mae_seasonal_insample)
        and model_mae < mae_seasonal_insample
    )
    return metrics


def _seasonal_naive_alinhado(df_total: pd.DataFrame, historico_total: pd.DataFrame | None) -> pd.Series:
    """Série seasonal naive (shift(52)) alinhada às datas de ``df_total``.

    Usa ``historico_total`` (treino+teste agregado) quando disponível para ter
    lookback além do teste; caso contrário, ``shift(52)`` local (provável NaN em
    janela curta). Estritamente retrospectivo.
    """
    base = df_total.sort_values("epi_sunday")
    y = base["cases"].astype(float)
    if historico_total is not None and not historico_total.empty:
        h = historico_total.copy()
        h["epi_sunday"] = pd.to_datetime(h["epi_sunday"])
        h = h.groupby("epi_sunday", as_index=False)["cases"].sum().sort_values("epi_sunday")
        sn_full = h["cases"].astype(float).shift(52)
        datas = pd.to_datetime(base["epi_sunday"])
        return pd.Series(
            pd.Series(sn_full.values, index=h["epi_sunday"]).reindex(datas).values,
            index=base.index,
        )
    return y.shift(52)


def avaliar_segmentos_epidemicos(
    pred_df: pd.DataFrame,
    df_total: pd.DataFrame,
    peak_threshold: float,
    historico_total: pd.DataFrame | None = None,
    min_n: int = 4,
) -> dict[str, float]:
    """
    Avalia MASE/WIS separados para semanas epidêmicas e interepidêmicas no DF-Total.

    O segmento epidêmico usa a MESMA máscara do ``hit_rate_picos``
    (``cases >= quantile(peak_threshold)``). Para cada segmento computa o MASE
    vs seasonal naive e, quando há quantis, o WIS completo sobre o subconjunto
    de linhas. Segmentos com menos de ``min_n`` semanas retornam ``NaN``
    (robustez), sem quebrar.

    Retorna chaves ``mase_epidemic``, ``mase_interepidemic``, ``wis_epidemic``,
    ``wis_interepidemic``, ``n_semanas_epidemic`` (apenas adições).
    """
    base = df_total.sort_values("epi_sunday").reset_index(drop=True)
    peak_val = base["cases"].quantile(peak_threshold)
    mask_epi = (base["cases"] >= peak_val).to_numpy()

    sn = _seasonal_naive_alinhado(base, historico_total).to_numpy(dtype=float)
    y = base["cases"].to_numpy(dtype=float)
    yhat = base["prediction"].to_numpy(dtype=float)

    multi_ci = len(colunas_quantis(pred_df)) >= 2
    datas = pd.to_datetime(base["epi_sunday"])

    def _mase_seg(mask) -> float:
        if mask.sum() < min_n:
            return float("nan")
        ys = pd.Series(y[mask]); ps = pd.Series(yhat[mask]); ss = pd.Series(sn[mask])
        return calcular_mase(ys, ps, ss)

    def _wis_seg(mask) -> float:
        if mask.sum() < min_n or not multi_ci:
            return float("nan")
        datas_seg = set(datas[mask])
        sub = pred_df[pd.to_datetime(pred_df["epi_sunday"]).isin(datas_seg)]
        if sub.empty:
            return float("nan")
        return calcular_wis_completo(sub)

    mask_inter = ~mask_epi
    n_epi = int(mask_epi.sum())
    if n_epi < min_n:
        logger.warning("Segmento epidêmico raso (%d < %d semanas); MASE/WIS = NaN.", n_epi, min_n)

    return {
        "mase_epidemic": _mase_seg(mask_epi),
        "mase_interepidemic": _mase_seg(mask_inter),
        "wis_epidemic": _wis_seg(mask_epi),
        "wis_interepidemic": _wis_seg(mask_inter),
        "n_semanas_epidemic": n_epi,
    }


def diagnosticar_contagens(pred_df: pd.DataFrame) -> dict[str, float]:
    df = pred_df.dropna(subset=["cases"]).copy()
    if df.empty:
        return {
            "count_zero_ratio": float("nan"),
            "count_overdispersion_ratio": float("nan"),
            "count_low_ratio_le_5": float("nan"),
        }
    cases = df["cases"].astype(float)
    mean_cases = float(cases.mean())
    variance = float(cases.var(ddof=1)) if len(cases) > 1 else 0.0
    return {
        "count_zero_ratio": float((cases == 0).mean()),
        "count_overdispersion_ratio": float(variance / mean_cases) if mean_cases > 0 else float("nan"),
        "count_low_ratio_le_5": float((cases <= 5).mean()),
    }


# ---------------------------------------------------------------------------
# Consolidação de Métricas
# ---------------------------------------------------------------------------

def consolidar_metricas_performance(
    pred_df: pd.DataFrame,
    peak_threshold: float = 0.75,
    alpha: float = 0.10,
    historico_total: pd.DataFrame | None = None,
    horizonte: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """
    Consolida métricas globais para o DF e individuais por Região Administrativa (RA).

    Calcula métricas pontuais (R², MAE, RMSE, MAPE, sMAPE, hit_rate_picos) e,
    quando colunas de intervalo de confiança (``lower_ci``, ``upper_ci``) estão
    presentes, métricas probabilísticas (coverage, WIS, calibration_error).

    Parâmetros:
        pred_df (pd.DataFrame): DataFrame contendo ``epi_sunday``, ``RA``,
            ``cases``, ``prediction``.  Opcionalmente ``lower_ci`` e ``upper_ci``.
        peak_threshold (float): Percentil para detecção de pico.
            Default ``0.75`` preserva comportamento original (P75).
        alpha (float): Nível de significância do CI para métricas probabilísticas.
            Default ``0.10`` → intervalo de 90 %.
        historico_total (pd.DataFrame, opcional): Série treino+teste
            (``epi_sunday``, ``cases``) para os baselines sazonais de
            ``calcular_baselines_df``. Se incluir a coluna ``RA``, habilita
            também o seasonal naive por RA (``mase_ra``). Sem ela,
            ``mase_vs_seasonal_naive``/``mase_ra`` tendem a NaN (aviso, não erro).
        horizonte (int): Passos à frente previstos. Alinha os baselines de
            persistência ao que era observável no instante da previsão; ver
            ``calcular_baselines_df``. Default ``1`` (nowcasting).

    Retorna:
        tuple: ``(dict_metricas_globais, DataFrame_metricas_por_ra)``
    """
    pred_df = normalizar_colunas_quantis(pred_df)
    # Previsão ausente nunca vira zero: a soma por semana contaria NaN como 0. As linhas
    # sem previsão saem das métricas e a fração coberta é reportada.
    tem_previsao = pred_df["prediction"].notna()
    n_ausentes = int((~tem_previsao).sum())
    if not tem_previsao.any():
        raise ValueError("Nenhuma previsão disponível para avaliar.")
    if n_ausentes:
        logger.warning("Avaliação: %d linha(s) sem previsão excluídas das métricas.", n_ausentes)
    cobertura_previsao = float(tem_previsao.mean())
    pred_df = pred_df[tem_previsao].copy()
    monotonicity_violations = validar_monotonicidade_quantis(pred_df)
    df_total = pred_df.groupby("epi_sunday", as_index=False)[["cases", "prediction"]].sum()
    
    # Métricas base
    r2 = calcular_r2_robusto(df_total["cases"], df_total["prediction"])
    mae = float(mean_absolute_error(df_total["cases"], df_total["prediction"]))
    rmse_val = calcular_erro_quadratico_medio(df_total["cases"], df_total["prediction"])
    
    # MAPE — apenas para semanas com casos > 0 (evita divisão por zero)
    nonzero = df_total[df_total["cases"] > 0].copy()
    mape_val = float(
        ((nonzero["cases"] - nonzero["prediction"]).abs() / nonzero["cases"] * 100).mean()
    ) if len(nonzero) > 0 else float("nan")
    
    # sMAPE — simétrico, robusto a casos próximos de zero
    denom = df_total["cases"].abs() + df_total["prediction"].abs() + 1e-8
    smape_val = float(
        (2 * (df_total["cases"] - df_total["prediction"]).abs() / denom * 100).mean()
    )
    
    # Hit Rate de Picos — % de semanas de pico real corretamente sinalizadas
    # Pico definido como acima do percentil configurável (default P75)
    peak_val = df_total["cases"].quantile(peak_threshold)
    is_peak_real = df_total["cases"] >= peak_val
    is_peak_pred = df_total["prediction"] >= peak_val
    hit_rate = float(
        (is_peak_real & is_peak_pred).sum() / is_peak_real.sum()
    ) if is_peak_real.sum() > 0 else float("nan")

    logger.debug("peak_threshold=%.2f (quantile=%.1f)", peak_threshold, peak_val)
    
    metrics = {
        "r2_df": r2,
        "mae_df": mae,
        "rmse_df": rmse_val,
        "mape_df": mape_val,
        "smape_df": smape_val,
        "hit_rate_picos": hit_rate,
        "peak_threshold": peak_threshold,
        "quantile_monotonicity_violations": monotonicity_violations,
        "cobertura_previsao": cobertura_previsao,
        "n_previsoes_ausentes": n_ausentes,
    }
    metrics.update(
        calcular_baselines_df(df_total, historico_total=historico_total, horizonte=horizonte)
    )
    metrics.update(diagnosticar_contagens(pred_df))
    metrics.update(
        avaliar_segmentos_epidemicos(
            pred_df, df_total, peak_threshold, historico_total=historico_total
        )
    )
    
    # -----------------------------------------------------------------------
    # Métricas probabilísticas — TDD-08 e TDD-10
    # Calculadas condicionalmente quando CI está presente no DataFrame.
    # -----------------------------------------------------------------------
    ci_cols_present = {"lower_ci", "upper_ci"}.issubset(pred_df.columns)
    multi_ci_cols_present = len(colunas_quantis(pred_df)) >= 2

    if ci_cols_present or multi_ci_cols_present:
        if multi_ci_cols_present:
            coverage, alpha_eff = calcular_cobertura_intervalo_quantil(pred_df, alpha=alpha)
        else:
            coverage, alpha_eff = calcular_cobertura_intervalo(pred_df, alpha=alpha), alpha
        cal_error = calcular_calibration_error(coverage, alpha=alpha_eff)
        
        if multi_ci_cols_present:
            wis = calcular_wis_completo(pred_df)
        else:
            wis = calcular_wis(pred_df, alpha=alpha)

        metrics["coverage_score"] = coverage
        metrics["coverage_alpha"] = alpha_eff
        metrics["wis"] = wis
        metrics["crps"] = calcular_crps_quantis(pred_df) if multi_ci_cols_present else float("nan")
        metrics["calibration_error"] = cal_error
        # Cobertura por quantil reportada apenas para τ ≥ 0,25: em contagens
        # zero-infladas os quantis de cauda baixa (τ < 0,25) são clipados em 0 e a
        # "cobertura" observada neles é artefato do clipping, não calibração real.
        for tau, _ in colunas_quantis(pred_df):
            if tau >= 0.25:
                metrics[f"coverage_q_{tau:.3f}"] = calcular_cobertura_quantil(pred_df, tau)

        logger.info(
            "Métricas probabilísticas: coverage=%.4f, WIS=%.4f, cal_error=%.4f (multi_ci=%s)",
            coverage, wis, cal_error, multi_ci_cols_present
        )
    else:
        metrics["coverage_score"] = float("nan")
        metrics["coverage_alpha"] = float("nan")
        metrics["wis"] = float("nan")
        metrics["crps"] = float("nan")
        metrics["calibration_error"] = float("nan")
        logger.info(
            "Colunas de CI ausentes; métricas probabilísticas preenchidas com NaN."
        )
    
    # Histórico por RA (quando historico_total traz a coluna RA) habilita o
    # seasonal naive por RA com lookback além do teste; sem ele, o shift(52)
    # local sobre ~52 semanas de teste degrada para NaN.
    hist_por_ra = None
    if historico_total is not None and "RA" in historico_total.columns:
        hist_por_ra = historico_total.copy()
        hist_por_ra["epi_sunday"] = pd.to_datetime(hist_por_ra["epi_sunday"])

    rows = []
    for ra, group in pred_df.groupby("RA"):
        group = group.sort_values("epi_sunday")
        y_ra = group["cases"].astype(float)
        if hist_por_ra is not None:
            base_ra = (
                hist_por_ra[hist_por_ra["RA"].eq(ra)]
                .groupby("epi_sunday", as_index=False)["cases"]
                .sum()
                .sort_values("epi_sunday")
            )
            sn_full = base_ra["cases"].astype(float).shift(52)
            sn_ra = pd.Series(
                pd.Series(sn_full.values, index=base_ra["epi_sunday"])
                .reindex(pd.to_datetime(group["epi_sunday"]))
                .values,
                index=y_ra.index,
            )
        else:
            sn_ra = y_ra.shift(52)

        ra_row = {
            "RA": ra,
            "r2_ra": calcular_r2_robusto(group["cases"], group["prediction"]),
            "mae_ra": float(mean_absolute_error(group["cases"], group["prediction"])),
            "rmse_ra": calcular_erro_quadratico_medio(group["cases"], group["prediction"]),
            "mase_ra": calcular_mase(y_ra, group["prediction"].astype(float), sn_ra),
        }
        # Cobertura e WIS por RA quando disponível
        if multi_ci_cols_present:
            ra_row["coverage_ra"], _ = calcular_cobertura_intervalo_quantil(group, alpha=alpha)
            ra_row["wis_ra"] = calcular_wis_completo(group)
        elif ci_cols_present:
            ra_row["coverage_ra"] = calcular_cobertura_intervalo(group, alpha=alpha)
            ra_row["wis_ra"] = calcular_wis(group, alpha=alpha)
        rows.append(ra_row)

    ra_df = pd.DataFrame(rows)
    metrics["r2_media_ras"] = float(ra_df["r2_ra"].mean(skipna=True))
    metrics["mae_media_ras"] = float(ra_df["mae_ra"].mean(skipna=True))
    metrics["rmse_media_ras"] = float(ra_df["rmse_ra"].mean(skipna=True))

    # Agregados de MASE por RA — NaN (histórico <52 semanas ou baseline
    # degenerado) é excluído, não contado como "não bateu o baseline".
    mase_validos = ra_df["mase_ra"].dropna() if "mase_ra" in ra_df.columns else pd.Series(dtype=float)
    metrics["frac_ras_beats_snaive"] = (
        float((mase_validos < 1).mean()) if not mase_validos.empty else float("nan")
    )
    metrics["mediana_mase_ras"] = (
        float(mase_validos.median()) if not mase_validos.empty else float("nan")
    )
    return metrics, ra_df
