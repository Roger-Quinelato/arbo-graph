# -*- coding: utf-8 -*-
"""
e0 — baseline RF/XGBoost (Hurdle) migrado do Arbovirose-Dengue (commit 402f845).

Origem:
- ``modeling/train_tuning.py``: ``HurdleRegressor``, ``fabrica_modelos`` (RF, XGB),
  ``executar_ajuste_previsao`` (modo ``nowcast``: alvo na semana t, lags até t-1);
- ``modeling/feature_engineering.py``: ``obter_configuracao_features`` e o pipeline
  (passthrough numérico + one-hot da RA + ``TransformedTargetRegressor`` log1p/expm1);
- ``modeling/orchestration.executar_forecast_direto_multihorizonte`` (modo ``direto``: um
  modelo por horizonte k, alvo ``shift(-k)`` por RA).

Diferenças deliberadas (decisão D-T1-2): sem reconciliação hierárquica e sem conformal. No
repo antigo a reconciliação preenchia previsão ausente com 0; aqui a linha fica NaN e entra
em ``cobertura_previsao``. XGB no modo direto produz os 23 quantis do CDC (como no antigo);
RF não produz quantis, então WIS e CRPS ficam NaN para ele.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier, XGBRegressor

from arbo_core import features as F
from arbo_core.evaluation.identity import (
    FORECAST_ORIGIN_COLUMN,
    materializar_resultado_forecast,
    preparar_amostras_horizonte,
)
from arbo_core.evaluation.metrics import CDC_QUANTILES, consolidar_metricas_performance, quantile_col
from arbo_core.evaluation.split import dividir_treino_teste_temporal

CASE_LAGS = [f"cases_lag_{lag}" for lag in (1, 2, 3, 4)]
INCID_LAGS = [f"incidencia_100k_lag_{lag}" for lag in (1, 2, 3, 4)]
TENDENCIA = ["cases_delta_1", "cases_delta_2", "cases_growth_rate"]
SAZONAIS = ["sin_week", "cos_week", "sin_month", "cos_month"]
SANEAMENTO = ["pct_rede_agua", "pct_rede_esgoto", "pct_lixo_coletado", "saneamento_cobertura"]
META = ["epi_sunday", "RA", "cases", "populacao"]


def especificacao_features(nome: str) -> dict:
    """Colunas de entrada por config. Nomes do repo antigo e os aliases fixados (e0-geo, e0-trends)."""
    clima = F.colunas_clima_ponto_unico()
    geo_ovi = F.colunas_vizinhanca() + F.colunas_ovitrampas()
    base = CASE_LAGS + TENDENCIA
    ambos = base + clima + F.colunas_clima_ra() + SAZONAIS + geo_ovi
    specs = {
        "lag-only": {"target": "cases", "base": base, "ra": False, "population": False},
        "lag+clima": {"target": "cases", "base": base + clima + SAZONAIS, "ra": False, "population": False},
        "lag+clima+RA": {"target": "cases", "base": base + clima + SAZONAIS, "ra": True, "population": False},
        "lag+clima+RA+incid-target": {"target": "incidencia_100k", "base": INCID_LAGS + TENDENCIA + clima + SAZONAIS,
                                      "ra": True, "population": True},
        "lag+clima+RA+geo-ovi": {"target": "cases", "base": base + clima + SAZONAIS + geo_ovi, "ra": True, "population": False},
        "lag+clima-ra+RA+geo-ovi": {"target": "cases", "base": base + F.colunas_clima_ra() + SAZONAIS + geo_ovi,
                                    "ra": True, "population": False},
        "lag+clima-ambos+RA+geo-ovi": {"target": "cases", "base": ambos, "ra": True, "population": False},
        "lag+clima-ambos+RA+geo-ovi+saneamento": {"target": "cases", "base": ambos + SANEAMENTO,
                                                  "ra": True, "population": False},
        "lag+clima-ambos+RA+geo-ovi+saneamento+trends": {
            "target": "cases", "base": ambos + SANEAMENTO + F.colunas_google_trends(), "ra": True, "population": False},
    }
    specs["e0-geo"] = specs["lag+clima-ambos+RA+geo-ovi"]
    specs["e0-trends"] = {"target": "cases", "base": ambos + F.colunas_google_trends(), "ra": True, "population": False}
    if nome not in specs:
        raise ValueError(f"Configuração de features desconhecida: {nome}")
    return specs[nome]


def colunas_entrada(spec: dict) -> list[str]:
    colunas = list(spec["base"])
    if spec["population"]:
        colunas.append("populacao")
    if spec["ra"]:
        colunas.append("RA")
    return colunas


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """Classificador XGB de y>0 (limiar 0,35) seguido do regressor; zero onde prob < limiar."""

    def __init__(self, regressor, classifier_threshold=0.35, classifier_params=None):
        self.regressor = regressor
        self.classifier_threshold = classifier_threshold
        self.classifier_params = classifier_params

    def fit(self, X, y):
        y_clf = np.asarray((y > 0)).astype(int)
        classes = np.unique(y_clf)
        if len(classes) < 2:
            self.classifier_ = None
            self.constant_prob_ = float(classes[0])
        else:
            self.classifier_ = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", **(self.classifier_params or {})
            )
            self.classifier_.fit(X, y_clf)
            self.constant_prob_ = None
        self.regressor_ = self.regressor
        self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        if self.classifier_ is None:
            prob = np.full(len(X), self.constant_prob_)
        else:
            prob = self.classifier_.predict_proba(X)[:, 1]
        mask = prob >= self.classifier_threshold
        amostra = X[mask] if mask.sum() > 0 else X[:1]
        pred_amostra = self.regressor_.predict(amostra)
        forma = (len(X), pred_amostra.shape[1]) if pred_amostra.ndim == 2 else (len(X),)
        y_pred = np.zeros(forma)
        if mask.sum() > 0:
            y_pred[mask] = pred_amostra
        return y_pred


def fabrica_modelos(nome_modelo: str, parametros: dict | None = None):
    parametros = dict(parametros or {})
    clf_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42, "n_jobs": -1}
    if nome_modelo == "RF":
        defaults = {"n_estimators": 150, "max_depth": 15, "random_state": 42, "n_jobs": -1}
        defaults.update(parametros)
        return HurdleRegressor(RandomForestRegressor(**defaults), 0.35, clf_params)
    if nome_modelo == "XGB":
        is_quantile = parametros.pop("multi_quantile", False)
        quantiles = parametros.pop("quantiles", None)
        defaults = {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "random_state": 42, "n_jobs": -1,
            "objective": "reg:quantileerror" if is_quantile else "reg:squarederror", "tree_method": "hist",
        }
        if is_quantile and quantiles is not None:
            defaults["quantile_alpha"] = quantiles
        defaults.update(parametros)
        return HurdleRegressor(XGBRegressor(**defaults), 0.35, clf_params)
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")


def construir_pipeline(spec: dict, modelo) -> Pipeline:
    numericas = list(spec["base"]) + (["populacao"] if spec["population"] else [])
    transformers = [("num", "passthrough", numericas)]
    if spec["ra"]:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float), ["RA"]))
    regressor = TransformedTargetRegressor(regressor=modelo, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocessor", ColumnTransformer(transformers, remainder="drop")), ("regressor", regressor)])


def _limpar(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    colunas = set(colunas_entrada(spec) + META + (["incidencia_100k"] if spec["target"] == "incidencia_100k" else []))
    return df.dropna(subset=[c for c in colunas if c in df.columns]).copy()


def _prever(pipeline, frame: pd.DataFrame, spec: dict) -> np.ndarray:
    pred = np.clip(pipeline.predict(frame[colunas_entrada(spec)]), 0, None)
    if spec["target"] == "incidencia_100k":
        fator = frame["populacao"].to_numpy(dtype=float) / 100000
        return pred * (fator[:, None] if pred.ndim == 2 else fator)
    return pred


def _com_quantis(pred_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    if preds.ndim == 2:
        pred_df["prediction"] = preds[:, preds.shape[1] // 2]
        for i, tau in enumerate(CDC_QUANTILES):
            pred_df[quantile_col(tau)] = preds[:, i]
    else:
        pred_df["prediction"] = preds
    return pred_df


def _verificar_colunas(painel: pd.DataFrame, spec: dict) -> None:
    faltantes = [c for c in colunas_entrada(spec) if c not in painel.columns]
    if faltantes:
        raise ValueError(f"Painel sem colunas exigidas pela config: {faltantes[:5]}")


def rodar_nowcast(painel: pd.DataFrame, spec: dict, nome_modelo: str, ano_teste: int):
    treino, teste = dividir_treino_teste_temporal(painel, ano_teste)
    treino_l, teste_l = _limpar(treino, spec), _limpar(teste, spec)
    pipeline = construir_pipeline(spec, fabrica_modelos(nome_modelo))
    pipeline.fit(treino_l[colunas_entrada(spec)], treino_l[spec["target"]].astype(float))

    pred = _com_quantis(teste_l[["epi_sunday", "RA"]].copy(), _prever(pipeline, teste_l, spec))
    esperado = teste[["epi_sunday", "RA", "cases", "incidencia_100k", "populacao"]]
    pred_df = esperado.merge(pred, on=["epi_sunday", "RA"], how="left")
    historico = pd.concat([treino, teste], ignore_index=True)[["epi_sunday", "RA", "cases"]]
    return pred_df, historico


def rodar_direto(painel: pd.DataFrame, spec: dict, nome_modelo: str, ano_teste: int, k: int):
    treino, teste = dividir_treino_teste_temporal(painel, ano_teste)
    treino_k = preparar_amostras_horizonte(treino, spec["target"], k)
    teste_k = preparar_amostras_horizonte(teste, spec["target"], k)
    treino_l, teste_l = _limpar(treino_k, spec), _limpar(teste_k, spec)

    parametros = {"multi_quantile": True, "quantiles": CDC_QUANTILES} if nome_modelo == "XGB" else None
    pipeline = construir_pipeline(spec, fabrica_modelos(nome_modelo, parametros))
    pipeline.fit(treino_l[colunas_entrada(spec)], treino_l["target_k"].astype(float))

    preds = _prever(pipeline, teste_l, spec)
    pontual = preds[:, preds.shape[1] // 2] if preds.ndim == 2 else preds
    pred = _com_quantis(materializar_resultado_forecast(teste_l, pontual), preds)
    chave = [FORECAST_ORIGIN_COLUMN, "epi_sunday", "RA"]
    esperado = materializar_resultado_forecast(teste_k, np.full(len(teste_k), np.nan)).drop(columns="prediction")
    pred_df = esperado.merge(pred.drop(columns=[c for c in ("cases", "incidencia_100k", "populacao") if c in pred]),
                             on=chave, how="left")
    historico = pd.concat([treino, teste], ignore_index=True)[["epi_sunday", "RA", "cases"]]
    return pred_df, historico


def executar(painel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = especificacao_features(config["features"])
    _verificar_colunas(painel, spec)
    linhas, previsoes = [], []
    for nome_modelo in config["modelos"]:
        tarefas = []
        if "nowcast" in config["modos"]:
            tarefas.append(("nowcast", 1, lambda m=nome_modelo: rodar_nowcast(painel, spec, m, config["ano_teste"])))
        if "direto" in config["modos"]:
            for k in config["horizontes"]:
                tarefas.append(("direto", k, lambda m=nome_modelo, k=k: rodar_direto(painel, spec, m, config["ano_teste"], k)))
        for modo, k, rodar in tarefas:
            pred_df, historico = rodar()
            metricas, _ = consolidar_metricas_performance(pred_df, historico_total=historico, horizonte=k)
            linhas.append({"modelo": nome_modelo, "modo": modo, "horizonte": k, "n_features": len(colunas_entrada(spec)), **metricas})
            previsoes.append(pred_df.assign(modelo=nome_modelo, modo=modo, horizonte=k))
    return pd.DataFrame(linhas), pd.concat(previsoes, ignore_index=True)
