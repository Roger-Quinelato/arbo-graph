# -*- coding: utf-8 -*-
"""
Executor de experimentos: roda o experimento pedido e o baseline e0 no mesmo painel.

Cada experimento vive em ``experiments/<nome>/`` com ``config.toml`` e ``model.py`` expondo
``executar(painel, config) -> (metricas, previsoes)``. O executor fixa as sementes, grava
métricas, previsões e um ``RESULTADO.md`` gerado (nunca escrito à mão).
"""

from __future__ import annotations

import importlib.util
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from arbo_core import config
from arbo_core.manifest import commit_atual

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
BASELINE = "e0_baseline_rf_xgb"

COLUNAS_RESUMO = [
    "experimento", "modelo", "modo", "horizonte", "mae_df", "rmse_df", "mase_vs_naive",
    "wis", "crps", "coverage_score", "cobertura_previsao", "model_beats_naive", "n_features",
]


def carregar_experimento(nome: str, diretorio: Path | None = None) -> tuple[ModuleType, dict]:
    pasta = (diretorio or EXPERIMENTS_DIR) / nome
    if not (pasta / "model.py").exists():
        raise FileNotFoundError(f"Experimento sem model.py: {pasta}")
    with (pasta / "config.toml").open("rb") as handle:
        cfg = tomllib.load(handle)
    spec = importlib.util.spec_from_file_location(f"experimento_{nome}", pasta / "model.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo, cfg


def _tabela_markdown(df: pd.DataFrame) -> str:
    def fmt(valor) -> str:
        if isinstance(valor, (float, np.floating)):
            return "NaN" if np.isnan(valor) else f"{valor:.3f}"
        return str(valor)

    linhas = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(linhas)


def gerar_resultado_md(nome: str, metricas: pd.DataFrame, painel: pd.DataFrame, configs: dict[str, dict]) -> str:
    resumo = metricas[[c for c in COLUNAS_RESUMO if c in metricas.columns]]
    partes = [
        f"# RESULTADO — {nome}",
        "",
        f"Gerado automaticamente por `run_experiment` em {datetime.now().isoformat(timespec='seconds')}. "
        "Não editar à mão.",
        "",
        f"- Commit: `{commit_atual(config.BASE_DIR)}`",
        f"- Painel: {len(painel)} linhas, {painel['RA'].nunique()} RAs, "
        f"{pd.Timestamp(painel['epi_sunday'].min()).date()} a {pd.Timestamp(painel['epi_sunday'].max()).date()}",
        f"- Alvo: `{painel['target_name'].iloc[0]}`" if "target_name" in painel.columns else "- Alvo: não registrado",
        f"- Baseline `{BASELINE}` rodado na mesma execução: {'sim' if BASELINE in configs else 'não'}",
        "",
        "## Configuração",
        "",
    ]
    for exp, cfg in configs.items():
        partes.append(f"- `{exp}`: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    partes += [
        "",
        "## Métricas (DF agregado; horizontes em semanas)",
        "",
        _tabela_markdown(resumo),
        "",
        "`cobertura_previsao` < 1 indica linhas RA×semana sem previsão (excluídas das métricas, nunca zeradas).",
        "",
        "Desempenho de artigo ou de modelo do Hugging Face não é resultado nosso e não entra nesta tabela.",
        "",
    ]
    return "\n".join(partes)


def executar_experimento(
    nome: str,
    painel: pd.DataFrame,
    saida: Path | None = None,
    diretorio: Path | None = None,
    com_baseline: bool = True,
    sobrescrever_config: dict | None = None,
    publicar: bool = False,
) -> Path:
    """
    Roda ``nome`` e, se for outro experimento, o baseline e0 no mesmo painel.

    Grava ``metricas.csv``, ``previsoes.parquet`` e ``RESULTADO.md`` em
    ``saida/<nome>/<carimbo>/`` e devolve esse diretório. Com ``publicar``, copia o
    ``RESULTADO.md`` para a pasta do experimento.
    """
    execucoes = [(nome, diretorio)]
    if com_baseline and nome != BASELINE:
        execucoes.append((BASELINE, None))

    metricas, previsoes, configs = [], [], {}
    for exp, pasta in execucoes:
        modulo, cfg = carregar_experimento(exp, pasta)
        cfg = {**cfg, **(sobrescrever_config or {}).get(exp, {})}
        config.seed_everything(int(cfg.get("seed", 42)))
        logger.info("Experimento %s: %s", exp, cfg)
        met, prev = modulo.executar(painel, cfg)
        metricas.append(met.assign(experimento=exp))
        previsoes.append(prev.assign(experimento=exp))
        configs[exp] = cfg

    metricas_df = pd.concat(metricas, ignore_index=True)
    previsoes_df = pd.concat(previsoes, ignore_index=True)
    carimbo = datetime.now().strftime("%Y%m%dT%H%M%S")
    destino = (saida or config.RESULTADOS_DIR) / nome / carimbo
    destino.mkdir(parents=True, exist_ok=True)
    metricas_df.to_csv(destino / "metricas.csv", index=False)
    previsoes_df.to_parquet(destino / "previsoes.parquet", index=False)
    texto = gerar_resultado_md(nome, metricas_df, painel, configs)
    (destino / "RESULTADO.md").write_text(texto, encoding="utf-8")
    if publicar:
        ((diretorio or EXPERIMENTS_DIR) / nome / "RESULTADO.md").write_text(texto, encoding="utf-8")
    return destino
