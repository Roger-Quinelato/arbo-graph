# -*- coding: utf-8 -*-
"""
Origem: ``dengue_pipeline/etl/climate_station_ingestion.py`` e ``weather_ingestion.py``
(commit 402f845). O cache de ponto único passou a ser a fonte nomeada ``clima_cache``.

Ingestão do clima por estação automática do INMET, espacializado por Região
Administrativa (RA) do DF — corrige o defeito D2 (``carregar_cache_climatico``
devolve uma única série de ponto único para as 31 RAs).

Fonte: ``inmet/*.zip`` — um zip por ano do INMET (lote atual: 2000–2014 e
2016–2026; **2015 não veio no lote** e fica como buraco real na série, nunca
preenchido por interpolação). Cada zip contém um CSV por estação automática do
Brasil inteiro; filtramos pelas estações do Distrito Federal exigindo ``_DF_``
no nome do arquivo (convenção INMET: ``INMET_CO_DF_A001_BRASILIA_...CSV``).

O código da estação (``A001``, ``A042``, ...) é extraído do nome do arquivo por
regex, sem fixar a lista a priori — isso cobre tanto o caso de uma estação sair
do ar num ano quanto o INMET adicionar uma nova estação no DF no futuro. No
lote atual aparecem 5 estações, nem todas presentes na série inteira:
    A001 Brasília           — única desde 2000
    A045 Águas Emendadas    — entra em 2008
    A042 Brazlândia         — entra em 2017/2019 (varia por remessa)
    A046 Gama (Ponte Alta)  — entra em 2014/2016
    A047 Paranoá (COOPA-DF) — entra em 2017/2019
(ver estatística exata no log de ``carregar_clima_por_ra`` — chave "Estações
por ano").

FORMATO DO CSV (estável em toda a série 2000–2026, confirmado por inspeção):
8 linhas de metadados de estação (região, UF, nome, código WMO, LATITUDE,
LONGITUDE, altitude, data de fundação) — usamos as linhas 5 e 6 (LATITUDE e
LONGITUDE) para posicionar a estação —, seguidas de 1 linha de cabeçalho de
colunas e depois a série horária. Separador ``;``, decimal ``,``, sentinela de
ausência ``-9999``. As colunas são lidas por **posição**, não por nome: o nome
de algumas colunas muda de grafia/acentuação entre remessas do INMET, mas a
posição é estável nas 20 colunas (índice 0 = data, 2 = precipitação horária
(mm), 7 = temperatura bulbo seco (°C), 15 = umidade relativa (%)).

MÉTODO DE ESPACIALIZAÇÃO — escolhido: ponderação pelo inverso do quadrado da
distância (IDW, p=2) entre o centroide de cada RA e todas as estações do DF
com leitura válida naquela semana epidemiológica — e **não** "estação mais
próxima" (nearest-neighbor). Justificativa:

  1. Com apenas 4-5 estações espalhadas por ~5.800 km², nearest-neighbor
     particiona o DF em 4-5 células de Voronoi: todas as RAs dentro da mesma
     célula receberiam, de novo, séries IDÊNTICAS entre si — exatamente o
     defeito D2 que este módulo existe para corrigir, só que em escala menor
     (poucos clusters em vez de 1 série única para o DF inteiro).
  2. IDW produz um valor contínuo por RA — nenhuma RA fica idêntica a outra a
     menos que os centroides coincidam —, preservando gradiente espacial em
     vez de um mapa de degraus.
  3. IDW é resiliente a falha pontual de uma estação numa semana: se a
     estação mais próxima de uma RA não reportou naquela semana, ela
     simplesmente some da ponderação (peso 0) e as demais assumem o peso
     relativo, em vez de forçar um salto abrupto para a segunda estação mais
     próxima, como aconteceria num esquema de nearest-neighbor com fallback.
  4. Efeito colateral desejável: sob IDW, ``n_estacoes`` (nº de estações com
     leitura válida na semana) é uma coluna informativa de fato — sob
     nearest-neighbor ela seria trivialmente sempre 1.

Contrapartida documentada (não escondida): IDW dilui o sinal de RAs com
estações "opostas" — uma estação a 3 km puxando a maior parte do peso e outra
a 40 km ainda contaminando o valor com o clima de 40 km de distância. Para
mitigar sem esconder o problema, expomos ``dist_km_estacao`` (distância à
estação mais próxima com leitura válida naquela semana) para que o consumidor
possa, por exemplo, descontar a confiança de RAs com distância grande. RAs a
mais de ``_RAIO_CAVEAT_KM`` km de qualquer estação do DF são contadas e
reportadas via log (ver ``_registrar_ras_distantes``) — isso NÃO impede a
estimativa (fail loud é para dados ausentes, não para dados de baixa
confiança), apenas documenta o caveat.

CENTROIDE DE RA: aproximado pela média aritmética simples dos vértices dos
polígonos da RA em ``dados_geo/geo.json`` (sem geopandas/shapely, seguindo o
padrão de ``etl/geo_ingestion.py``). É uma aproximação grosseira para RAs de
geometria muito alongada ou côncava, mas suficiente dado que o objetivo é
localizar a RA em relação a poucas estações dezenas de km distantes, não
delimitar sua fronteira com precisão.

RAs DO geo.json QUE NÃO EXISTEM NO CONJUNTO CANÔNICO DE 31 RAs: o geo.json tem
2 polígonos ("26 DE SETEMBRO", "PONTE ALTA") que não são mapeados por
``consolidar_ra`` e não aparecem no dataset de população/casos. Seguindo o
mesmo padrão documentado em ``geo_ingestion.py``, este módulo NÃO os filtra —
eles aparecem no resultado com clima próprio, e cabe ao consumidor (join com
``["epi_sunday", "RA"]``) ignorá-los, já que não terão par do lado de casos.

SEM VAZAMENTO TEMPORAL: cada linha (epi_sunday × RA) é calculada apenas a
partir das leituras horárias daquela própria semana civil de estações que
reportaram naquela semana. Não há médias globais, climatologia ou anomalia
sobre a série inteira — a agregação semana a semana é local por construção
(ver ``_agregar_por_ra``), então adicionar semanas futuras ao lote de entrada
nunca muda o valor calculado para uma semana passada.

AUSÊNCIA ≠ ZERO: ``-9999`` é convertido para NaN na leitura (``na_values``).
Uma semana em que uma estação não tem nenhuma leitura horária válida não entra
na ponderação daquela semana (não vira 0 nem puxa a média para baixo). Uma
semana em que NENHUMA estação do DF tem leitura válida gera ``NaN`` nas três
colunas de clima (nunca ``0``), e ``n_estacoes = 0`` (contagem verdadeira,
não sentinela). O grid semanal é preenchido de domingo em domingo do primeiro
ao último epi_sunday observado em qualquer estação — isso garante que buracos
totais (ex.: o ano de 2015, sem zip no lote) aparecem como linhas com NaN em
vez de simplesmente sumirem do dataset.
"""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import json
import logging
import re
import zipfile

import numpy as np
import pandas as pd

from arbo_core.epi_calendar import calcular_semana_epidemiologica
from arbo_core.contracts import WeatherPolicy
from arbo_core.ras import consolidar_ra, sanitizar_texto

logger = logging.getLogger(__name__)

# --- Constantes do parser posicional do CSV do INMET (ver docstring do módulo) ---
_LINHAS_METADADO = 8          # REGIAO..DATA DE FUNDACAO, antes da linha de cabeçalho
_LINHAS_A_PULAR = _LINHAS_METADADO + 1  # + a própria linha de cabeçalho de colunas
_IDX_DATA = 0
_IDX_PRECIP = 2
_IDX_TEMP = 7
_IDX_UMID = 15
_SENTINELA_AUSENTE = "-9999"
_REGEX_CODIGO_ESTACAO = re.compile(r"_([A-Z]\d{3})_")

# --- Constantes do método IDW (ver docstring do módulo) ---
_POTENCIA_IDW = 2
_EPS_DISTANCIA_KM = 1e-6  # evita divisão por zero se o centroide cair sobre a estação
_RAIO_CAVEAT_KM = 15.0    # limiar documentado para o caveat de cobertura espacial
_RAIO_TERRA_KM = 6371.0088


def _diretorio_inmet() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("inmet")


def _caminho_geojson() -> Path:
    from arbo_core.config import diretorio_fonte

    return diretorio_fonte("dados_geo") / "geo.json"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância em km entre dois pontos (graus decimais), fórmula de haversine
    escrita à mão — sem geopandas/shapely, seguindo o padrão de ``geo_ingestion.py``."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * _RAIO_TERRA_KM * np.arcsin(np.sqrt(a)))


def _vertices_da_geometria(geom: dict) -> list[tuple[float, float]]:
    """Extrai (lon, lat) de todos os vértices de um Polygon/MultiPolygon do geo.json."""
    tipo = geom.get("type")
    if tipo == "Polygon":
        rings = geom["coordinates"]
    elif tipo == "MultiPolygon":
        rings = [ring for poly in geom["coordinates"] for ring in poly]
    else:
        return []
    return [(float(x), float(y)) for ring in rings for x, y in ring]


def _centroides_ras() -> dict[str, tuple[float, float]]:
    """
    Lê ``dados_geo/geo.json`` e devolve, por RA canônica consolidada, o
    centroide aproximado (média aritmética dos vértices do polígono) como
    ``(lat, lon)``.

    Levanta:
        FileNotFoundError: se ``dados_geo/geo.json`` não existir.
    """
    caminho = _caminho_geojson()
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de polígonos das RAs não encontrado: {caminho}. "
            "Necessário para localizar o centroide de cada RA em relação às "
            "estações do INMET."
        )

    colecao = json.loads(caminho.read_text(encoding="utf-8"))
    vertices_por_ra: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for feature in colecao.get("features", []):
        nome_bruto = feature.get("properties", {}).get("ra_nome")
        nome = consolidar_ra(sanitizar_texto(nome_bruto))
        if nome is None:
            continue
        vertices_por_ra[nome].extend(_vertices_da_geometria(feature.get("geometry", {})))

    centroides: dict[str, tuple[float, float]] = {}
    for nome, vertices in vertices_por_ra.items():
        if not vertices:
            continue
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        centroides[nome] = (sum(lats) / len(lats), sum(lons) / len(lons))
    return centroides


def _ler_metadados_estacao(linhas: list[str]) -> tuple[float, float]:
    """Extrai (lat, lon) das 8 linhas de metadados do cabeçalho do CSV do INMET."""
    lat_txt = linhas[4].split(";")[1].strip().replace(",", ".")
    lon_txt = linhas[5].split(";")[1].strip().replace(",", ".")
    return float(lat_txt), float(lon_txt)


def _ler_horario_estacao(conteudo: bytes) -> tuple[float, float, pd.DataFrame]:
    """
    Faz o parse de um CSV de estação do INMET (bytes brutos do zip).

    Retorna:
        (lat, lon, DataFrame horário com colunas ``data``, ``precip``, ``temp``, ``umid``)
    """
    texto = conteudo.decode("latin-1")
    linhas = texto.splitlines()
    lat, lon = _ler_metadados_estacao(linhas)

    tabela = pd.read_csv(
        BytesIO(conteudo),
        sep=";",
        skiprows=_LINHAS_A_PULAR,
        header=None,
        usecols=[_IDX_DATA, _IDX_PRECIP, _IDX_TEMP, _IDX_UMID],
        decimal=",",
        na_values=[_SENTINELA_AUSENTE],
        encoding="latin-1",
        engine="c",
    )
    tabela = tabela.rename(
        columns={_IDX_DATA: "data", _IDX_PRECIP: "precip", _IDX_TEMP: "temp", _IDX_UMID: "umid"}
    )
    tabela["data"] = pd.to_datetime(tabela["data"], errors="coerce")
    tabela = tabela.dropna(subset=["data"])
    return lat, lon, tabela[["data", "precip", "temp", "umid"]]


def _coletar_dados_estacoes(
    diretorio: Path,
) -> tuple[dict[str, tuple[float, float]], dict[str, pd.DataFrame], dict[int, set[str]]]:
    """
    Varre todos os ``inmet/*.zip`` e consolida, por estação do DF, as
    coordenadas e a série horária bruta (concatenada entre todos os anos —
    importante para não fatiar ao meio a semana epidemiológica que cruza a
    virada do ano civil entre dois zips).

    Retorna:
        (coordenadas_por_estacao, horario_bruto_por_estacao, estacoes_por_ano)
    """
    zips = sorted(diretorio.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"Nenhum arquivo .zip do INMET encontrado em {diretorio}")

    coordenadas: dict[str, tuple[float, float]] = {}
    partes_horarias: dict[str, list[pd.DataFrame]] = defaultdict(list)
    estacoes_por_ano: dict[int, set[str]] = defaultdict(set)

    for caminho_zip in zips:
        try:
            ano = int(caminho_zip.stem)
        except ValueError:
            ano = -1
        with zipfile.ZipFile(caminho_zip) as zf:
            arquivos_df = [n for n in zf.namelist() if "_DF_" in n.split("/")[-1]]
            for nome in arquivos_df:
                base = nome.split("/")[-1]
                m = _REGEX_CODIGO_ESTACAO.search(base)
                if not m:
                    logger.warning("INMET: não foi possível extrair código de estação de %s", base)
                    continue
                codigo = m.group(1)
                try:
                    conteudo = zf.read(nome)
                    lat, lon, horario = _ler_horario_estacao(conteudo)
                except Exception as exc:  # noqa: BLE001 - arquivo isolado não derruba o lote inteiro
                    logger.warning("INMET: falha ao ler %s (%s): %s", base, caminho_zip.name, exc)
                    continue

                coordenadas[codigo] = (lat, lon)
                partes_horarias[codigo].append(horario)
                if ano > 0:
                    estacoes_por_ano[ano].add(codigo)

    if not partes_horarias:
        raise FileNotFoundError(
            f"Nenhum CSV de estação do DF ('_DF_' no nome) encontrado dentro dos zips em {diretorio}"
        )

    horario_por_estacao = {
        codigo: pd.concat(partes, ignore_index=True) for codigo, partes in partes_horarias.items()
    }
    return coordenadas, horario_por_estacao, estacoes_por_ano


def _agregar_semanal_por_estacao(horario: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega leituras horárias de UMA estação em semanas epidemiológicas
    (domingo-sábado). Agregação é local por semana — nenhuma estatística
    usa dados de outras semanas (sem vazamento temporal).

    ``precip_sum`` usa ``min_count=1``: se todas as leituras horárias de
    precipitação da semana forem NaN, o resultado é NaN (não 0 — "sem chuva
    medida" é diferente de "não choveu").
    """
    horario = horario.copy()
    horario["epi_sunday"] = calcular_semana_epidemiologica(horario["data"])
    horario = horario.dropna(subset=["epi_sunday"])

    agregado = horario.groupby("epi_sunday").agg(
        temp_mean=("temp", "mean"),
        umid_med=("umid", "mean"),
    )
    precip = horario.groupby("epi_sunday")[["precip"]].sum(min_count=1)
    agregado["precip_sum"] = precip["precip"]
    return agregado[["temp_mean", "precip_sum", "umid_med"]]


def ras_distantes(
    centroides: dict[str, tuple[float, float]], coordenadas: dict[str, tuple[float, float]]
) -> list[tuple[str, float]]:
    """RAs a mais de ``_RAIO_CAVEAT_KM`` da estação mais próxima da rede completa, com a distância."""
    distantes = []
    for ra, (lat_ra, lon_ra) in centroides.items():
        d_min = min(_haversine_km(lat_ra, lon_ra, lat_e, lon_e) for lat_e, lon_e in coordenadas.values())
        if d_min > _RAIO_CAVEAT_KM:
            distantes.append((ra, round(d_min, 1)))
    return sorted(distantes)


def _registrar_ras_distantes(
    centroides: dict[str, tuple[float, float]], coordenadas: dict[str, tuple[float, float]]
) -> list[tuple[str, float]]:
    """Loga e devolve as RAs distantes (caveat estrutural da rede, distinto de
    ``dist_km_estacao``, que é dinâmico por semana)."""
    if not coordenadas:
        return []
    distantes = ras_distantes(centroides, coordenadas)
    if distantes:
        logger.warning(
            "INMET: %d RA(s) a mais de %.0f km de qualquer estação do DF (rede completa, %d estações): %s",
            len(distantes), _RAIO_CAVEAT_KM, len(coordenadas), distantes,
        )
    else:
        logger.info(
            "INMET: todas as %d RAs estão a até %.0f km de alguma estação do DF.",
            len(centroides), _RAIO_CAVEAT_KM,
        )
    return distantes


def _agregar_por_ra(
    centroides: dict[str, tuple[float, float]],
    semanal_por_estacao: dict[str, pd.DataFrame],
    coordenadas: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Espacializa as séries semanais por estação para as RAs via IDW (ver
    docstring do módulo). Toda a operação é feita semana a semana (eixo 0 dos
    arrays abaixo é a semana) — nenhum agregado usa informação de outra
    semana, então não há vazamento temporal.
    """
    ras = sorted(centroides)
    codigos = sorted(semanal_por_estacao)

    todas_semanas = pd.DatetimeIndex(
        sorted(set().union(*[df.index for df in semanal_por_estacao.values()]))
    )
    if todas_semanas.empty:
        raise FileNotFoundError("INMET: nenhuma semana epidemiológica com leitura horária válida.")
    grade_semanas = pd.date_range(todas_semanas.min(), todas_semanas.max(), freq="7D")

    # Matriz de distância RA×estação (km), fixa — não depende da semana.
    dist = np.array(
        [
            [_haversine_km(centroides[ra][0], centroides[ra][1], coordenadas[c][0], coordenadas[c][1]) for c in codigos]
            for ra in ras
        ]
    )
    peso_base = 1.0 / (dist**_POTENCIA_IDW + _EPS_DISTANCIA_KM)  # (R, S)

    def _tabela_larga(coluna: str) -> pd.DataFrame:
        larga = pd.DataFrame(index=grade_semanas, columns=codigos, dtype=float)
        for c in codigos:
            larga[c] = semanal_por_estacao[c][coluna].reindex(grade_semanas)
        return larga

    temp_larga = _tabela_larga("temp_mean")
    precip_larga = _tabela_larga("precip_sum")
    umid_larga = _tabela_larga("umid_med")

    # Máscara de disponibilidade da TEMPERATURA usada como referência de rede
    # ativa (n_estacoes, dist_km_estacao) — ver caveat no docstring do módulo:
    # as três variáveis são lidas juntas hora a hora pela mesma estação, então
    # na prática a disponibilidade de temp/precip/umid coincide quase sempre.
    mask_temp = temp_larga.notna().to_numpy()  # (W, S)
    n_estacoes = mask_temp.sum(axis=1)  # (W,)

    dist_masked = np.where(mask_temp[:, None, :], dist[None, :, :], np.inf)  # (W, R, S)
    dist_min = dist_masked.min(axis=2)  # (W, R)
    dist_min = np.where(np.isinf(dist_min), np.nan, dist_min)

    def _idw(tabela: pd.DataFrame) -> np.ndarray:
        mask = tabela.notna().to_numpy()  # (W, S) — só entram estações com leitura válida NAQUELA semana
        vals = np.nan_to_num(tabela.to_numpy(), nan=0.0)  # (W, S)
        pesos = peso_base[None, :, :] * mask[:, None, :]  # (W, R, S)
        denom = pesos.sum(axis=2)  # (W, R)
        numer = np.einsum("wrs,ws->wr", pesos, vals)  # (W, R)
        with np.errstate(invalid="ignore", divide="ignore"):
            resultado = numer / denom
        resultado = np.where(denom == 0, np.nan, resultado)
        return resultado

    temp_ra = _idw(temp_larga)
    precip_ra = _idw(precip_larga)
    umid_ra = _idw(umid_larga)

    n_semanas, n_ras = len(grade_semanas), len(ras)
    saida = pd.DataFrame(
        {
            "epi_sunday": np.repeat(grade_semanas.values, n_ras),
            "RA": np.tile(np.array(ras, dtype=object), n_semanas),
            "temp_mean_ra": temp_ra.reshape(-1),
            "precip_sum_ra": precip_ra.reshape(-1),
            "umid_med_ra": umid_ra.reshape(-1),
            "n_estacoes": np.repeat(n_estacoes, n_ras).astype("int64"),
            "dist_km_estacao": dist_min.reshape(-1),
        }
    )
    return saida.sort_values(["epi_sunday", "RA"]).reset_index(drop=True)


def carregar_clima_por_ra() -> pd.DataFrame:
    """
    Carrega o clima do INMET espacializado por Região Administrativa do DF.

    Fonte: ``inmet/*.zip`` (estações do DF, filtradas por ``_DF_`` no nome do
    arquivo) + ``dados_geo/geo.json`` (centroide de cada RA). Ver docstring do
    módulo para o método de espacialização (IDW), o parser posicional do CSV
    do INMET e as garantias de ausência-nunca-vira-zero / sem-vazamento.

    Retorna:
        pd.DataFrame com colunas ``epi_sunday``, ``RA``, ``temp_mean_ra``,
        ``precip_sum_ra``, ``umid_med_ra``, ``n_estacoes``, ``dist_km_estacao``.
        Chave: ``["epi_sunday", "RA"]``. ``attrs["ras_distantes_km"]`` lista as RAs
        a mais de 15 km da rede.

    Levanta:
        FileNotFoundError: se ``inmet/`` não existir, não tiver nenhum .zip,
            nenhum CSV de estação do DF, ou se ``dados_geo/geo.json`` não
            existir — nunca devolve DataFrame vazio em silêncio.
    """
    diretorio = _diretorio_inmet()
    if not diretorio.exists():
        raise FileNotFoundError(
            f"Diretório de dados do INMET não encontrado: {diretorio}. "
            "Esperado um .zip por ano com CSVs de estação (ver docstring do módulo)."
        )

    coordenadas, horario_por_estacao, estacoes_por_ano = _coletar_dados_estacoes(diretorio)
    semanal_por_estacao = {
        codigo: _agregar_semanal_por_estacao(horario) for codigo, horario in horario_por_estacao.items()
    }

    centroides = _centroides_ras()

    distantes = _registrar_ras_distantes(centroides, coordenadas)
    for ano in sorted(estacoes_por_ano):
        logger.info("INMET: %d — estações do DF ativas: %s", ano, sorted(estacoes_por_ano[ano]))

    resultado = _agregar_por_ra(centroides, semanal_por_estacao, coordenadas)
    logger.info(
        "INMET: %d estações × %d RAs → %d linhas (%s a %s), método IDW p=%d",
        len(coordenadas), len(centroides), len(resultado),
        resultado["epi_sunday"].min().date(), resultado["epi_sunday"].max().date(),
        _POTENCIA_IDW,
    )
    resultado.attrs["ras_distantes_km"] = distantes
    return resultado


# --- Clima de ponto único (Open-Meteo + umidade do InfoDengue) ---


def carregar_cache_climatico(
    weather_policy: WeatherPolicy | str | None = None,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Carrega dados meteorológicos consolidados cruzando cache local do Open-Meteo com dados históricos da InfoDengue.
    Realiza interpolação linear para colunas de umidade faltantes.
    
    Retorna:
        pd.DataFrame: DataFrame de clima semanal indexado por epi_sunday.
        
    Premissas críticas:
        Espera ``dados_clima_cache.csv`` na fonte ``clima_cache``
        e InfoDengue_2016-2026.csv na subpasta InfoDengue/.
    """
    from arbo_core.config import diretorio_fonte

    if weather_policy is None:
        weather_policy = WeatherPolicy.RETROSPECTIVE_INTERPOLATION

    # No repo antigo o cache era lido de dados_processados/; aqui é fonte nomeada.
    caminho_clima_cache = diretorio_fonte("clima_cache") / "dados_clima_cache.csv"
    if not caminho_clima_cache.exists():
        raise FileNotFoundError(f"Arquivo de cache climático não encontrado em {caminho_clima_cache}")
        
    weather = pd.read_csv(caminho_clima_cache)
    weather["epi_sunday"] = pd.to_datetime(weather["epi_sunday"]).dt.normalize()

    info_file = diretorio_fonte("InfoDengue") / "InfoDengue_2016-2026.csv"
    if not info_file.exists():
        raise FileNotFoundError(f"Arquivo InfoDengue não encontrado em {info_file}")
        
    infod = pd.read_csv(info_file)
    infod["epi_sunday"] = pd.to_datetime(infod["data_iniSE"]).dt.normalize()
    humidity_cols = ["epi_sunday", "umidmed", "umidmin", "umidmax"]
    infod = infod[humidity_cols].drop_duplicates("epi_sunday").sort_values("epi_sunday")
    
    weather_policy = WeatherPolicy(weather_policy)
    climate = weather.merge(infod, on="epi_sunday", how="left").sort_values("epi_sunday")
    if as_of_date is not None and weather_policy is not WeatherPolicy.RETROSPECTIVE_INTERPOLATION:
        climate = climate[climate["epi_sunday"].le(pd.Timestamp(as_of_date).normalize())].copy()

    for col in ["umidmed", "umidmin", "umidmax"]:
        if weather_policy is WeatherPolicy.RETROSPECTIVE_INTERPOLATION:
            climate[col] = climate[col].interpolate(method="linear", limit_direction="both")
            climate[f"{col}_imputation_policy"] = weather_policy.value
        else:
            climate[col] = climate[col].ffill()
            climate[f"{col}_imputation_policy"] = weather_policy.value
        
    # Validação de cobertura: não deve haver semanas sem temperatura principal
    colunas_obrigatorias = ["temp_mean", "precip_sum"]
    for col in colunas_obrigatorias:
        if col in climate.columns:
            n_nulos = climate[col].isna().sum()
            if n_nulos > 0:
                semanas = climate.loc[climate[col].isna(), "epi_sunday"].tolist()
                raise ValueError(
                    f"ETL: coluna '{col}' tem {n_nulos} semana(s) sem cobertura climática "
                    f"após interpolação. Primeiras afetadas: {semanas[:5]}. "
                    "Execute fetch_nasa_power.py para atualizar o cache climático."
                )
                
    return climate
