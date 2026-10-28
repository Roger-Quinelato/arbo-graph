# -*- coding: utf-8 -*-
"""
Manifesto de proveniência por execução de ``prepare``.

Substitui a trilha documental do repo antigo: cada painel gerado carrega o hash das fontes
lidas, o commit do código, as versões de contrato, o período e as RAs. Origem do hash:
``dengue_pipeline/etl/provenance.calcular_hash`` (commit 402f845).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from arbo_core import contracts

_TAMANHO_BLOCO = 1024 * 1024


def calcular_hash(caminho: Path) -> str:
    """SHA-256 do arquivo, lido em blocos."""
    digest = hashlib.sha256()
    with caminho.open("rb") as handle:
        for bloco in iter(lambda: handle.read(_TAMANHO_BLOCO), b""):
            digest.update(bloco)
    return digest.hexdigest()


def hash_fonte(diretorio: Path) -> dict[str, str]:
    """Hash de cada arquivo da fonte, por caminho relativo, em ordem estável."""
    if not diretorio.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {diretorio}")
    return {
        arquivo.relative_to(diretorio).as_posix(): calcular_hash(arquivo)
        for arquivo in sorted(diretorio.rglob("*"))
        if arquivo.is_file()
    }


def commit_atual(raiz: Path) -> str:
    """Commit do código que gerou o painel, ou ``"desconhecido"`` fora de um repositório git."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"
    return resultado.stdout.strip()


def validar_periodo(painel: pd.DataFrame) -> None:
    """Falha se o painel começar antes de ``INICIO_SERIE_ESPACIAL``."""
    inicio = pd.Timestamp(painel["epi_sunday"].min())
    if inicio < contracts.INICIO_SERIE_ESPACIAL:
        raise ValueError(
            f"Painel começa em {inicio.date()}, antes de "
            f"{contracts.INICIO_SERIE_ESPACIAL.date()} (INICIO_SERIE_ESPACIAL)."
        )


def construir_manifesto(
    painel: pd.DataFrame,
    fontes: dict[str, Path],
    contexto: contracts.TemporalContext,
    raiz_codigo: Path,
    caveats: dict | None = None,
) -> dict:
    """Reúne proveniência, período, universo de RAs e avisos de cobertura do painel."""
    validar_periodo(painel)
    metadata = contexto.metadata()
    return {
        "commit": commit_atual(raiz_codigo),
        "alvo": contexto.target_name,
        "inicio_serie_espacial": contracts.INICIO_SERIE_ESPACIAL.date().isoformat(),
        "versoes_contrato": {
            chave: metadata[chave]
            for chave in (
                "artifact_schema_version",
                "forecast_problem_version",
                "temporal_context_version",
                "quantile_schema_version",
            )
        },
        "contexto_temporal": metadata,
        "periodo": {
            "inicio": pd.Timestamp(painel["epi_sunday"].min()).date().isoformat(),
            "fim": pd.Timestamp(painel["epi_sunday"].max()).date().isoformat(),
            "semanas": int(painel["epi_sunday"].nunique()),
        },
        "ras": sorted(painel["RA"].unique().tolist()),
        "n_linhas": int(len(painel)),
        "colunas": list(painel.columns),
        "fontes": {
            nome: {"caminho": str(caminho), "arquivos": hash_fonte(caminho)}
            for nome, caminho in fontes.items()
        },
        "caveats": caveats or {},
    }


def salvar_manifesto(manifesto: dict, caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return caminho
