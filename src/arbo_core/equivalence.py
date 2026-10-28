# -*- coding: utf-8 -*-
"""
Validação de equivalência (T9): o repo novo reproduz o painel e as métricas do antigo?

Portão da decisão 8: sem equivalência, os experimentos E1–E6 não começam até a diferença
ser explicada. O pipeline antigo roda em subprocesso isolado sobre as MESMAS fontes
(``ARBO_DADOS_DIR``); ``populacao_historica.csv`` e ``dados_clima_cache.csv``, que o antigo lê
de ``dados_processados/``, são copiados das fontes nomeadas ``populacao`` e ``clima_cache``.

Nenhum arquivo do ``dados_processados/`` real é lido: o antigo roda numa raiz temporária.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from arbo_core import config
from arbo_core.contracts import TemporalContext
from arbo_core.features import enriquecedores_padrao
from arbo_core.panel import construir_painel

COMMIT_ORIGEM = "402f845"
# Checkout do Arbovirose-Dengue: ``ARBO_REPO_ANTIGO`` ou, por padrão, a pasta irmã deste repositório.
REPO_ANTIGO_PADRAO = Path(
    os.environ.get("ARBO_REPO_ANTIGO") or Path(__file__).resolve().parents[3] / "Arbovirose-Dengue"
)

# Diferenças esperadas por decisão, não por erro.
COLUNAS_IGNORADAS = {"processing_time"}  # carimbo de execução
COLUNAS_FORA_POR_DECISAO = {  # Censo fora por enquanto (decisão 4)
    "pct_rede_agua", "pct_rede_esgoto", "pct_lixo_coletado", "saneamento_cobertura",
}
METRICAS_COMPARADAS = [
    "mae_df", "rmse_df", "r2_df", "smape_df", "hit_rate_picos", "mase_vs_naive",
    "mase_vs_seasonal_naive", "wis", "coverage_score", "mae_media_ras",
]


@dataclass
class Relatorio:
    equivalente: bool
    painel: dict = field(default_factory=dict)
    metricas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"equivalente": self.equivalente, "painel": self.painel, "metricas": self.metricas}


def comparar_paineis(antigo: pd.DataFrame, novo: pd.DataFrame, rtol: float = 1e-9, atol: float = 1e-9) -> dict:
    chave = ["epi_sunday", "RA"]
    so_antigo = sorted(set(antigo.columns) - set(novo.columns) - COLUNAS_IGNORADAS)
    so_novo = sorted(set(novo.columns) - set(antigo.columns) - COLUNAS_IGNORADAS)
    comuns = sorted((set(antigo.columns) & set(novo.columns)) - COLUNAS_IGNORADAS - set(chave))

    a = antigo.sort_values(chave).reset_index(drop=True)
    n = novo.sort_values(chave).reset_index(drop=True)
    mesmas_linhas = len(a) == len(n) and a[chave].equals(n[chave])
    divergentes: dict[str, str] = {}
    if mesmas_linhas:
        for coluna in comuns:
            va, vn = a[coluna], n[coluna]
            if pd.api.types.is_numeric_dtype(va) and pd.api.types.is_numeric_dtype(vn):
                iguais = np.isclose(va.to_numpy(float), vn.to_numpy(float), rtol=rtol, atol=atol, equal_nan=True)
                if not iguais.all():
                    diff = np.nanmax(np.abs(va.to_numpy(float) - vn.to_numpy(float)))
                    divergentes[coluna] = f"{int((~iguais).sum())} linhas, máx |dif| = {diff:.3g}"
            elif not va.astype(str).equals(vn.astype(str)):
                divergentes[coluna] = f"{int((va.astype(str) != vn.astype(str)).sum())} linhas diferentes"

    inesperadas_antigo = [c for c in so_antigo if c not in COLUNAS_FORA_POR_DECISAO]
    return {
        "linhas_antigo": len(a),
        "linhas_novo": len(n),
        "mesmas_linhas": bool(mesmas_linhas),
        "ras_so_antigo": sorted(set(a["RA"]) - set(n["RA"])),
        "ras_so_novo": sorted(set(n["RA"]) - set(a["RA"])),
        "colunas_so_antigo": so_antigo,
        "colunas_fora_por_decisao": [c for c in so_antigo if c in COLUNAS_FORA_POR_DECISAO],
        "colunas_so_novo": so_novo,
        "colunas_divergentes": divergentes,
        "equivalente": bool(mesmas_linhas and not divergentes and not inesperadas_antigo),
    }


def comparar_metricas(antigo: pd.DataFrame, novo: pd.DataFrame, rtol: float = 1e-6, atol: float = 1e-9) -> dict:
    chave = ["modelo", "modo", "horizonte"]
    juntas = antigo.merge(novo, on=chave, how="outer", suffixes=("_antigo", "_novo"), indicator=True)
    faltantes = juntas.loc[juntas["_merge"] != "both", chave].to_dict("records")
    divergentes = []
    for _, linha in juntas[juntas["_merge"] == "both"].iterrows():
        for metrica in METRICAS_COMPARADAS:
            va, vn = linha.get(f"{metrica}_antigo"), linha.get(f"{metrica}_novo")
            if va is None or vn is None:
                continue
            if not np.isclose(float(va), float(vn), rtol=rtol, atol=atol, equal_nan=True):
                divergentes.append({**{c: linha[c] for c in chave}, "metrica": metrica,
                                    "antigo": float(va), "novo": float(vn)})
    return {"combinacoes_faltantes": faltantes, "divergentes": divergentes,
            "equivalente": not faltantes and not divergentes}


def rodar_antigo(
    saida: Path, features: str, ano_teste: int, modelos: list[str], horizontes: list[int],
    repo_antigo: Path = REPO_ANTIGO_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda o pipeline antigo numa raiz temporária, com as fontes de ``ARBO_DADOS_DIR``."""
    saida.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="legado_") as raiz:
        raiz = Path(raiz)
        processados = raiz / "dados_processados"
        processados.mkdir()
        shutil.copy(config.diretorio_fonte("populacao") / "populacao_historica.csv", processados)
        shutil.copy(config.diretorio_fonte("clima_cache") / "dados_clima_cache.csv", processados)
        (raiz / "resultados_modelagem").mkdir()
        env = {
            **os.environ,
            "PIPELINE_ROOT": str(raiz),
            "ARBO_DADOS_DIR": str(config._resolve_dados_dir()),
            "PYTHONPATH": str(repo_antigo / "src"),
        }
        comando = [sys.executable, "-P", str(Path(__file__).with_name("_legado_runner.py")), str(saida), features,
                   str(ano_teste), ",".join(modelos), ",".join(str(k) for k in horizontes)]
        subprocess.run(comando, env=env, check=True, cwd=raiz)
    return pd.read_parquet(saida / "painel_antigo.parquet"), pd.read_csv(saida / "metricas_antigo.csv")


def rodar_novo(features: str, ano_teste: int, modelos: list[str], horizontes: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from arbo_core.experiments import BASELINE, carregar_experimento

    contexto = TemporalContext.retrospective()
    painel = construir_painel(contexto, enriquecedores=enriquecedores_padrao(contexto))
    modulo, cfg = carregar_experimento(BASELINE)
    cfg = {**cfg, "features": features, "ano_teste": ano_teste, "modelos": modelos, "horizontes": horizontes}
    config.seed_everything(42)
    metricas, _ = modulo.executar(painel, cfg)
    return painel, metricas


def verificar(
    saida: Path, features: str = "lag+clima-ambos+RA+geo-ovi", ano_teste: int = 2025,
    modelos: tuple[str, ...] = ("RF", "XGB"), horizontes: tuple[int, ...] = (1, 2, 3, 4),
    repo_antigo: Path = REPO_ANTIGO_PADRAO,
) -> Relatorio:
    """Roda antigo e novo, compara e grava ``equivalencia.json`` em ``saida``."""
    painel_a, metricas_a = rodar_antigo(saida, features, ano_teste, list(modelos), list(horizontes), repo_antigo)
    painel_n, metricas_n = rodar_novo(features, ano_teste, list(modelos), list(horizontes))
    rel_painel = comparar_paineis(painel_a, painel_n)
    rel_metricas = comparar_metricas(metricas_a, metricas_n)
    relatorio = Relatorio(rel_painel["equivalente"] and rel_metricas["equivalente"], rel_painel, rel_metricas)
    relatorio.painel["commit_origem"] = COMMIT_ORIGEM
    relatorio.painel["src_antigo_alterado_desde_origem"] = _src_alterado(repo_antigo)
    (saida / "equivalencia.json").write_text(
        json.dumps(relatorio.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return relatorio


def _src_alterado(repo_antigo: Path) -> bool | None:
    """True se ``src/`` do repo antigo difere do commit de origem (resultado não comparável)."""
    try:
        r = subprocess.run(["git", "diff", "--quiet", COMMIT_ORIGEM, "--", "src"], cwd=repo_antigo)
    except OSError:
        return None
    return {0: False, 1: True}.get(r.returncode)
