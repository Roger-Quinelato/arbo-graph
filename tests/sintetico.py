"""
Fontes sintéticas pequenas: 4 RAs × 60 semanas desde 2020-01-05.

Nenhum dado real ou individual. Os registros imitam o esquema do InfoSaúde e incluem ruído
que o painel deve descartar: caso descartado, outra UF, RA não informada e data anterior a 2020.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INICIO = pd.Timestamp("2020-01-05")
N_SEMANAS = 60
# Nome como vem da fonte -> nome canônico esperado no painel.
RAS = {
    "Ceilândia": "CEILANDIA",
    "Águas Claras": "AGUAS CLARAS",
    "Gama": "GAMA",
    "Plano Piloto": "PLANO PILOTO",
}
POPULACAO = {"Ceilândia": 350_000, "Águas Claras": 120_000, "Gama": 140_000, "Plano Piloto": 220_000}


def casos_esperados(semana: int, indice_ra: int) -> int:
    """Contagem determinística com semanas zeradas e picos."""
    return (semana * (indice_ra + 1)) % 7


def _registro(data: pd.Timestamp, ra: str, classe="CASO PROVAVEL", doenca="DENGUE", uf="DF") -> dict:
    return {
        "i_class_final": classe,
        "i_desc_classificacao": doenca,
        "i_desc_uf_res": uf,
        "i_desc_radf_res": ra,
        "i_data_prim_sintomas": data.date().isoformat(),
        "i_data_notificacao": (data + pd.Timedelta(days=3)).date().isoformat(),
    }


def gerar_casos() -> tuple[pd.DataFrame, dict[tuple[pd.Timestamp, str], int]]:
    """Registros individuais sintéticos e contagem esperada por (domingo, RA canônica)."""
    registros: list[dict] = []
    esperado: dict[tuple[pd.Timestamp, str], int] = {}
    for semana in range(N_SEMANAS):
        domingo = INICIO + pd.Timedelta(weeks=semana)
        for indice, (nome, canonico) in enumerate(RAS.items()):
            n = casos_esperados(semana, indice)
            esperado[(domingo, canonico)] = n
            for i in range(n):
                doenca = "INCONCLUSIVO" if i == 2 else "DENGUE"
                registros.append(_registro(domingo + pd.Timedelta(days=i % 7), nome, doenca=doenca))

    domingo = INICIO + pd.Timedelta(weeks=10)
    # RA desmembrada: conta para a RA de origem.
    registros.append(_registro(domingo, "Arniqueira"))
    esperado[(domingo, "AGUAS CLARAS")] += 1
    # Ruído que deve ser descartado.
    registros.append(_registro(domingo, "Gama", classe="DESCARTADO"))
    registros.append(_registro(domingo, "Gama", uf="GO"))
    registros.append(_registro(domingo, "Não Informado"))
    registros.append(_registro(pd.Timestamp("2019-06-10"), "Gama"))
    return pd.DataFrame(registros), esperado


def gerar_populacao() -> pd.DataFrame:
    linhas = []
    for ano in (2019, 2020, 2021):
        for nome, pop in POPULACAO.items():
            linhas.append({"RA": nome, "ano": ano, "populacao": pop})
        linhas.append({"RA": "Arniqueira", "ano": ano, "populacao": 50_000})
    return pd.DataFrame(linhas)


def escrever_fontes(raiz_dados: Path) -> dict[tuple[pd.Timestamp, str], int]:
    """Grava ``info-saude/`` e ``populacao/`` sob ``raiz_dados`` e devolve a contagem esperada."""
    casos, esperado = gerar_casos()
    (raiz_dados / "info-saude").mkdir(parents=True)
    casos.to_csv(raiz_dados / "info-saude" / "casos_sinteticos.csv", sep=";", index=False, encoding="utf-8")
    (raiz_dados / "populacao").mkdir(parents=True)
    gerar_populacao().to_csv(raiz_dados / "populacao" / "populacao_historica.csv", index=False)
    return esperado


# --- Geometria e clima sintéticos ---

# Três RAs em linha, com lados compartilhados, e uma RA isolada a cerca de 55 km.
_LADO = 0.05
_ORIGEM = (-48.10, -15.85)  # (lon, lat)
POSICOES = {"Ceilândia": (0, 0), "Águas Claras": (1, 0), "Gama": (2, 0), "Plano Piloto": (10, 0)}
VIZINHOS_ESPERADOS = {
    "CEILANDIA": {"AGUAS CLARAS"},
    "AGUAS CLARAS": {"CEILANDIA", "GAMA"},
    "GAMA": {"AGUAS CLARAS"},
    "PLANO PILOTO": set(),
}


def _quadrado(coluna: int, linha: int) -> list[list[float]]:
    x0 = _ORIGEM[0] + coluna * _LADO
    y0 = _ORIGEM[1] + linha * _LADO
    return [[x0, y0], [x0 + _LADO, y0], [x0 + _LADO, y0 + _LADO], [x0, y0 + _LADO], [x0, y0]]


def centro(nome: str) -> tuple[float, float]:
    """(lat, lon) do centro do quadrado da RA."""
    coluna, linha = POSICOES[nome]
    return (_ORIGEM[1] + (linha + 0.5) * _LADO, _ORIGEM[0] + (coluna + 0.5) * _LADO)


# Variante sem RA isolada: o pipeline antigo falha quando alguma RA não tem vizinha
# (``pd.NA`` convertido com ``astype(float)``), então a equivalência usa esta geometria.
POSICOES_CONECTADAS = {**POSICOES, "Plano Piloto": (3, 0)}


def escrever_geojson(raiz_dados: Path, isolada: bool = True) -> Path:
    import json

    posicoes = POSICOES if isolada else POSICOES_CONECTADAS
    features = [
        {"type": "Feature", "properties": {"ra_nome": nome},
         "geometry": {"type": "Polygon", "coordinates": [_quadrado(*pos)]}}
        for nome, pos in posicoes.items()
    ]
    destino = raiz_dados / "dados_geo" / "geo.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return destino


def gerar_clima_cache(inicio="2019-06-02", semanas=100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache de ponto único e umidade do InfoDengue, com um buraco de umidade em 2020."""
    domingos = pd.date_range(inicio, periods=semanas, freq="W-SUN")
    i = pd.Series(range(semanas), dtype=float)
    cache = pd.DataFrame({"epi_sunday": domingos, "temp_mean": 20 + i % 9, "precip_sum": (i * 3) % 40})
    umid = 50 + (i * 7) % 30
    umid[40:43] = float("nan")
    info = pd.DataFrame({"data_iniSE": domingos, "umidmed": umid, "umidmin": umid - 10, "umidmax": umid + 10})
    return cache, info


def escrever_clima_cache(raiz_dados: Path) -> None:
    cache, info = gerar_clima_cache()
    (raiz_dados / "clima_cache").mkdir(parents=True, exist_ok=True)
    cache.to_csv(raiz_dados / "clima_cache" / "dados_clima_cache.csv", index=False)
    (raiz_dados / "InfoDengue").mkdir(parents=True, exist_ok=True)
    info.to_csv(raiz_dados / "InfoDengue" / "InfoDengue_2016-2026.csv", index=False)


# Estação A001 no centro de Ceilândia; A042 a leste de Gama. Plano Piloto fica a >15 km de ambas.
ESTACOES = {"A001": (centro("Ceilândia"), 25.0), "A042": ((centro("Gama")[0], centro("Gama")[1] + 0.05), 15.0)}
SEMANA_SEM_A042 = pd.Timestamp("2020-03-01")


def _csv_inmet(codigo: str, lat: float, lon: float, temp: float, dias: pd.DatetimeIndex) -> bytes:
    meta = [
        "REGIAO:;CO", "UF:;DF", f"ESTACAO:;{codigo}", f"CODIGO (WMO):;{codigo}",
        f"LATITUDE:;{str(lat).replace('.', ',')}", f"LONGITUDE:;{str(lon).replace('.', ',')}",
        "ALTITUDE:;1000", "DATA DE FUNDACAO:;2000-01-01",
    ]
    cabecalho = ";".join(f"c{i}" for i in range(20))
    linhas = []
    for dia in dias:
        valores = ["0"] * 20
        valores[0] = dia.strftime("%Y/%m/%d")
        valores[1] = "1200 UTC"
        ausente = codigo == "A042" and SEMANA_SEM_A042 <= dia < SEMANA_SEM_A042 + pd.Timedelta(days=7)
        valores[2] = "-9999" if ausente else "1,5"
        valores[7] = "-9999" if ausente else str(temp).replace(".", ",")
        valores[15] = "-9999" if ausente else "60"
        linhas.append(";".join(valores))
    return "\n".join(meta + [cabecalho] + linhas).encode("latin-1")


def escrever_inmet(raiz_dados: Path) -> None:
    import zipfile

    destino = raiz_dados / "inmet"
    destino.mkdir(parents=True, exist_ok=True)
    dias = pd.date_range("2020-01-05", periods=N_SEMANAS * 7, freq="D")
    for ano in sorted(set(dias.year)):
        dias_ano = dias[dias.year == ano]
        with zipfile.ZipFile(destino / f"{ano}.zip", "w") as zf:
            for codigo, ((lat, lon), temp) in ESTACOES.items():
                nome = f"INMET_CO_DF_{codigo}_SINTETICA_01-01-{ano}_A_31-12-{ano}.CSV"
                zf.writestr(nome, _csv_inmet(codigo, lat, lon, temp, dias_ano))
            zf.writestr(f"INMET_SE_SP_A701_OUTRA_UF_{ano}.CSV", "ignorar")


def escrever_fontes_completas(raiz_dados: Path, isolada: bool = True) -> dict[tuple[pd.Timestamp, str], int]:
    esperado = escrever_fontes(raiz_dados)
    escrever_geojson(raiz_dados, isolada=isolada)
    escrever_clima_cache(raiz_dados)
    escrever_inmet(raiz_dados)
    escrever_ovitrampas(raiz_dados)
    escrever_trends(raiz_dados)
    return esperado


# --- Ovitrampas e Google Trends sintéticos ---

def semana(i: int) -> pd.Timestamp:
    return INICIO + pd.Timedelta(weeks=i)


# Leituras: (semana, município como na fonte, ovos, resultado)
LEITURAS_OVI = [
    (5, "Ceilândia (RA IX)", 10, "Positiva"),
    (5, "Ceilândia (RA IX)", 0, "Negativa"),
    (8, "Ceilândia (RA IX)", 0, "Negativa"),
    (5, "Brasília (RA I)", 4, "Positiva"),
    (6, "Setor Inexistente", 3, "Positiva"),
]


def escrever_ovitrampas(raiz_dados: Path) -> None:
    linhas = [
        {"Município": mun, "Data": (semana(s) + pd.Timedelta(days=2)).date().isoformat(),
         "Ovos": ovos, "Resultado": resultado}
        for s, mun, ovos, resultado in LEITURAS_OVI
    ]
    (raiz_dados / "ovitrampas").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(raiz_dados / "ovitrampas" / "ovitrampas.csv", sep=";", index=False)


def gerar_trends(geo: str = "BR-DF") -> pd.DataFrame:
    from arbo_core.epi_calendar import calcular_epi_year_week

    domingos = pd.Series(pd.date_range(INICIO, periods=N_SEMANAS, freq="W-SUN"))
    partes = calcular_epi_year_week(domingos)
    return pd.DataFrame({
        "week_start": domingos.dt.date.astype(str),
        "epi_year": partes["epi_year"].astype(int),
        "epi_week": partes["epi_week"].astype(int),
        "geo": geo,
        "google_trends_dengue": [float(10 + (i * 5) % 50) for i in range(N_SEMANAS)],
    })


def escrever_trends(raiz_dados: Path, geo: str = "BR-DF") -> None:
    (raiz_dados / "Google Trends").mkdir(parents=True, exist_ok=True)
    gerar_trends(geo).to_csv(raiz_dados / "Google Trends" / "trends.csv", index=False)
