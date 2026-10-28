# -*- coding: utf-8 -*-
"""
CLI ``prepare``: gera o painel RA×semana e o manifesto de proveniência.

Uso: ``python -m arbo_core.prepare [--saida DIR]``. Fontes via ``ARBO_DADOS_DIR``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from arbo_core import config
from arbo_core.contracts import ALVO_PADRAO, TemporalContext
from arbo_core.features import enriquecedores_padrao
from arbo_core.manifest import construir_manifesto, salvar_manifesto
from arbo_core.panel import Enriquecedor, construir_painel

logger = logging.getLogger(__name__)

# Fontes lidas pelo painel de casos. Cada enriquecedor acrescenta as suas em ``fontes``.
FONTES_CASOS: tuple[str, ...] = ("info-saude", "populacao")

NOME_PAINEL = "painel.parquet"
NOME_MANIFESTO = "manifesto.json"


def preparar(
    saida: Path | None = None,
    contexto: TemporalContext | None = None,
    target_name: str = ALVO_PADRAO,
    enriquecedores: Sequence[Enriquecedor] | None = None,
) -> tuple[Path, Path]:
    """
    Monta o painel, grava Parquet e manifesto em ``saida`` e devolve os dois caminhos.

    Sem ``enriquecedores``, usa o painel completo de ``enriquecedores_padrao``.
    """
    saida = Path(saida) if saida is not None else config.DADOS_PROCESSADOS_DIR
    contexto = contexto or TemporalContext.retrospective(target_name=target_name)
    if enriquecedores is None:
        enriquecedores = enriquecedores_padrao(contexto)

    painel = construir_painel(contexto, target_name, enriquecedores)

    nomes_fontes = list(FONTES_CASOS)
    caveats: dict = {}
    for enriquecedor in enriquecedores:
        nomes_fontes += [f for f in getattr(enriquecedor, "fontes", ()) if f not in nomes_fontes]
        caveats.update(getattr(enriquecedor, "caveats", {}))
    diretorios = {}
    for nome in nomes_fontes:
        diretorio = config.diretorio_fonte(nome)
        if diretorio.exists():
            diretorios[nome] = diretorio
        else:
            caveats.setdefault("fontes_ausentes", []).append(nome)
    manifesto = construir_manifesto(painel, diretorios, contexto, config.BASE_DIR, caveats)

    saida.mkdir(parents=True, exist_ok=True)
    caminho_painel = saida / NOME_PAINEL
    painel.to_parquet(caminho_painel, index=False)
    caminho_manifesto = salvar_manifesto(manifesto, saida / NOME_MANIFESTO)
    logger.info("Painel: %d linhas, %d RAs, %s a %s",
                manifesto["n_linhas"], len(manifesto["ras"]),
                manifesto["periodo"]["inicio"], manifesto["periodo"]["fim"])
    return caminho_painel, caminho_manifesto


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gera o painel RA×semana e o manifesto.")
    parser.add_argument("--saida", type=Path, default=None, help="Diretório de saída (padrão: dados_processados/)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    preparar(saida=args.saida)


if __name__ == "__main__":
    main()
