"""Grafo do geo.json e feature de vizinhança por média."""

import json

import numpy as np
import pandas as pd

import sintetico
from arbo_core.contracts import TemporalContext
from arbo_core.graph import carregar_adjacencia_ras, lista_arestas, mapear_nos, matriz_adjacencia
from arbo_core.features import Vizinhanca, enriquecedores_padrao
from arbo_core.panel import construir_painel
from arbo_core.prepare import preparar


def test_adjacencia_por_fronteira_compartilhada(fontes_completas):
    adjacencia = carregar_adjacencia_ras()
    assert {ra: set(v) for ra, v in adjacencia.items()} == sintetico.VIZINHOS_ESPERADOS


def test_matriz_simetrica_sem_lacos(fontes_completas):
    adjacencia = carregar_adjacencia_ras()
    mapa = mapear_nos(sintetico.RAS.values(), adjacencia)
    matriz = matriz_adjacencia(adjacencia, mapa)
    assert np.array_equal(matriz, matriz.T)
    assert not np.diag(matriz).any()
    assert lista_arestas(adjacencia, mapa).shape == (2, int(matriz.sum()))


def test_mapeamento_explicito_de_nos():
    adjacencia = {"A": frozenset({"B"}), "B": frozenset({"A"}), "SO_POLIGONO": frozenset()}
    mapa = mapear_nos(["A", "B", "SO_PAINEL"], adjacencia)
    assert mapa.nos == ("A", "B")
    assert mapa.ras_sem_poligono == ("SO_PAINEL",)
    assert mapa.poligonos_sem_painel == ("SO_POLIGONO",)


def test_vizinhanca_media_e_ra_isolada_nan(fontes_completas):
    painel = construir_painel(TemporalContext.retrospective(), enriquecedores=[Vizinhanca()])
    larga = painel.pivot(index="epi_sunday", columns="RA", values="cases_lag_1")
    viz = painel.pivot(index="epi_sunday", columns="RA", values="viz_cases_lag_1")
    semana = sintetico.semana(20)
    assert viz.loc[semana, "AGUAS CLARAS"] == (larga.loc[semana, "CEILANDIA"] + larga.loc[semana, "GAMA"]) / 2
    assert viz.loc[semana, "CEILANDIA"] == larga.loc[semana, "AGUAS CLARAS"]
    assert viz["PLANO PILOTO"].isna().all()


def test_vizinhanca_ausente_degrada(fontes_sinteticas):
    enriquecedor = Vizinhanca()
    painel = construir_painel(TemporalContext.retrospective(), enriquecedores=[enriquecedor])
    assert "viz_cases_lag_1" not in painel.columns
    assert "vizinhanca_pulado" in enriquecedor.caveats


def test_painel_completo_e_manifesto(fontes_completas, tmp_path):
    caminho_painel, caminho_manifesto = preparar(saida=tmp_path / "saida")
    painel = pd.read_parquet(caminho_painel)
    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    assert {"viz_cases_lag_1", "ovi_cobertura", "clima_ra_cobertura", "gt_cobertura"} <= set(painel.columns)
    assert manifesto["caveats"]["grafo_ras_sem_poligono"] == []
    assert len(painel) == sintetico.N_SEMANAS * len(sintetico.RAS)
