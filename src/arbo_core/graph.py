# -*- coding: utf-8 -*-
"""
Grafo de adjacência entre RAs a partir de ``dados_geo/geo.json``.

Origem: ``dengue_pipeline/etl/geo_ingestion.py`` (commit 402f845). A regra de adjacência é a
mesma: duas RAs são vizinhas quando compartilham pelo menos 2 vértices (arco de fronteira),
com vértices arredondados a 6 casas decimais. O cache passou a depender do caminho e da data
de modificação do arquivo, para não vazar entre fontes diferentes.

Acrescentado para modelos de grafo: matriz de adjacência, lista de arestas e um mapeamento
explícito entre RAs do painel e nós do grafo.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from arbo_core.ras import consolidar_ra, sanitizar_texto

logger = logging.getLogger(__name__)

# 1 vértice = toque de esquina (não é fronteira); ≥ 2 = arco compartilhado.
_MIN_VERTICES_COMPARTILHADOS = 2
_PRECISAO_VERTICE = 6


def _caminho_geojson() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("dados_geo") / "geo.json"


def _vertices_da_geometria(geom: dict) -> set[tuple[float, float]]:
    """Conjunto de vértices arredondados de um Polygon/MultiPolygon."""
    tipo = geom.get("type")
    if tipo == "Polygon":
        rings = geom["coordinates"]
    elif tipo == "MultiPolygon":
        rings = [ring for poly in geom["coordinates"] for ring in poly]
    else:
        return set()
    return {(round(x, _PRECISAO_VERTICE), round(y, _PRECISAO_VERTICE)) for ring in rings for x, y in ring}


@lru_cache(maxsize=8)
def _adjacencia_do_arquivo(caminho: str, _mtime: float) -> dict[str, frozenset[str]]:
    colecao = json.loads(Path(caminho).read_text(encoding="utf-8"))
    vertices: dict[str, set[tuple[float, float]]] = defaultdict(set)
    for feature in colecao.get("features", []):
        nome = consolidar_ra(sanitizar_texto(feature.get("properties", {}).get("ra_nome")))
        if nome is None:
            continue
        vertices[nome] |= _vertices_da_geometria(feature.get("geometry", {}))

    nomes = sorted(vertices)
    adjacencia: dict[str, set[str]] = {nome: set() for nome in nomes}
    for i, ra_a in enumerate(nomes):
        for ra_b in nomes[i + 1:]:
            if len(vertices[ra_a] & vertices[ra_b]) >= _MIN_VERTICES_COMPARTILHADOS:
                adjacencia[ra_a].add(ra_b)
                adjacencia[ra_b].add(ra_a)

    isoladas = [nome for nome in nomes if not adjacencia[nome]]
    if isoladas:
        logger.warning("Geo: RAs sem vizinhos no grafo de adjacência: %s", isoladas)
    logger.info(
        "Geo: grafo de adjacência com %d unidades, grau médio %.1f",
        len(nomes), sum(len(v) for v in adjacencia.values()) / max(len(nomes), 1),
    )
    return {nome: frozenset(vizinhas) for nome, vizinhas in adjacencia.items()}


def carregar_adjacencia_ras(caminho: Path | None = None) -> dict[str, frozenset[str]]:
    """
    RA canônica → conjunto de RAs vizinhas.

    Levanta ``FileNotFoundError`` se o geo.json não existir.
    """
    caminho = caminho or _caminho_geojson()
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de polígonos das RAs não encontrado: {caminho}")
    return _adjacencia_do_arquivo(str(caminho.resolve()), caminho.stat().st_mtime)


@dataclass(frozen=True)
class MapaNos:
    """Correspondência entre RAs do painel e nós do grafo."""

    nos: tuple[str, ...]                  # RAs do painel com polígono, em ordem estável
    ras_sem_poligono: tuple[str, ...]     # RAs do painel ausentes do geo.json
    poligonos_sem_painel: tuple[str, ...] # polígonos que não viram série no painel

    def indice(self) -> dict[str, int]:
        return {ra: i for i, ra in enumerate(self.nos)}


def mapear_nos(ras_painel, adjacencia: dict[str, frozenset[str]]) -> MapaNos:
    ras_painel = set(ras_painel)
    poligonos = set(adjacencia)
    return MapaNos(
        nos=tuple(sorted(ras_painel & poligonos)),
        ras_sem_poligono=tuple(sorted(ras_painel - poligonos)),
        poligonos_sem_painel=tuple(sorted(poligonos - ras_painel)),
    )


def matriz_adjacencia(adjacencia: dict[str, frozenset[str]], mapa: MapaNos) -> np.ndarray:
    """Matriz binária simétrica N×N, sem laços, na ordem de ``mapa.nos``."""
    idx = mapa.indice()
    matriz = np.zeros((len(mapa.nos), len(mapa.nos)), dtype=float)
    for ra, vizinhas in adjacencia.items():
        if ra not in idx:
            continue
        for vizinha in vizinhas:
            if vizinha in idx:
                matriz[idx[ra], idx[vizinha]] = 1.0
    return matriz


def lista_arestas(adjacencia: dict[str, frozenset[str]], mapa: MapaNos) -> np.ndarray:
    """Arestas dirigidas nos dois sentidos, formato 2×E (``edge_index`` do PyG)."""
    origem, destino = np.nonzero(matriz_adjacencia(adjacencia, mapa))
    return np.vstack([origem, destino]).astype(np.int64)
