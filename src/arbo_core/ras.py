# -*- coding: utf-8 -*-
"""Origem: ``dengue_pipeline/shared_kernel/ra_registry.py`` (commit 402f845)."""
import unicodedata
import pandas as pd

def sanitizar_texto(valor) -> str | None:
    """
    Remove acentos e converte para maiúsculo, removendo espaços nas bordas.
    
    Parâmetros:
        valor: Valor de entrada (esperado string).
        
    Retorna:
        String normalizada ou None caso não seja string.
    """
    if not isinstance(valor, str):
        return None
    valor = "".join(
        c for c in unicodedata.normalize("NFD", valor)
        if unicodedata.category(c) != "Mn"
    )
    return valor.upper().strip()


# Mapa de consolidação espacial: RAs desmembradas ao longo da série (2017→) são
# reagrupadas em suas RAs de origem, garantindo unidades espaciais estáveis em todo
# o período. Chaves e valores em forma SANITIZADA (sem acento, maiúscula).
RA_CONSOLIDACAO: dict[str, str] = {
    "SOL NASCENTE E POR DO SOL": "CEILANDIA",       # criada em 2019, desmembrada de Ceilândia
    "ARNIQUEIRA":                "AGUAS CLARAS",     # criada em 2019, desmembrada de Águas Claras
    "ARAPOANGA":                 "PLANALTINA",       # registro SINAN a partir de 2019, parte de Planaltina
    "AGUA QUENTE":               "RECANTO DAS EMAS", # registro SINAN a partir de 2020, parte de Recanto das Emas
}


# Classificações do SINAN que compõem a "família de dengue" — constante de domínio
# epidemiológico usada pelo ETL (mascaras_target) e reporting (análise de target).
FAMILIA_DENGUE: frozenset[str] = frozenset({
    "DENGUE",
    "DENGUE COM SINAIS DE ALARME",
    "DENGUE GRAVE",
})


def consolidar_ra(ra_sanitizada: str | None) -> str | None:
    """
    Reagrupa uma RA desmembrada em sua RA de origem (ver RA_CONSOLIDACAO).

    Idempotente: nomes que já são RA de origem (ou não constam no mapa) retornam
    inalterados. Espera entrada já sanitizada (sem acento, maiúscula).
    """
    if ra_sanitizada is None:
        return None
    return RA_CONSOLIDACAO.get(ra_sanitizada, ra_sanitizada)


def carregar_historico_populacao(caminho_csv=None) -> pd.DataFrame:
    """
    Carrega o histórico populacional do arquivo populacao_historica.csv.

    A chave de RA é normalizada (sem acento, maiúscula) e as RAs desmembradas ao longo
    da série são reagrupadas em suas RAs de origem (ver RA_CONSOLIDACAO), somando-se as
    populações por ano. Isso garante:
      (a) ausência do mismatch de acento que zerava RAs como CEILÂNDIA no merge espacial;
      (b) unidades espaciais estáveis em todo o período (2017→).

    Parâmetros:
        caminho_csv: Path para o arquivo CSV. Se None, resolve via diretorio_fonte("populacao").

    Retorna:
        DataFrame com colunas ['RA', 'ano', 'populacao', 'ra_key'] já consolidado.
    """
    if caminho_csv is None:
        # No repo antigo o arquivo era lido de dados_processados/; aqui é fonte nomeada.
        from arbo_core.config import diretorio_fonte
        caminho_csv = diretorio_fonte("populacao") / "populacao_historica.csv"
    from pathlib import Path
    caminho_csv = Path(caminho_csv)
    if not caminho_csv.exists():
        raise FileNotFoundError(f"Arquivo populacao_historica.csv não encontrado em {caminho_csv}")
    pop = pd.read_csv(caminho_csv)
    pop["RA"] = pop["RA"].astype(str).map(sanitizar_texto).map(consolidar_ra)
    pop = pop.groupby(["RA", "ano"], as_index=False)["populacao"].sum()
    pop["ra_key"] = pop["RA"]
    return pop

def busca_ra_canonica() -> dict[str, str]:
    """
    Monta um mapeamento das chaves normalizadas (sem acentos) para os nomes canônicos das RAs.
    Implementa um fallback estático robusto caso o arquivo CSV esteja indisponível.
    
    Retorna:
        Dicionário com o mapeamento chave_normalizada -> nome_canonico.
    """
    lookup = {}
    try:
        pop = carregar_historico_populacao()
        lookup = dict(zip(pop["ra_key"], pop["RA"]))
    except Exception as e:
        import warnings
        warnings.warn(
            f"Falha ao carregar populacao_historica.csv em busca_ra_canonica: {e}. Usando fallback estatico.",
            UserWarning
        )
        # Fallback estático com nomes normalizados sem acento
        static_ras = [
            "AGUA QUENTE", "ARAPOANGA", "ARNIQUEIRA", "BRAZLANDIA", "CANDANGOLANDIA",
            "CEILANDIA", "CRUZEIRO", "FERCAL", "GAMA", "GUARA", "ITAPOA",
            "JARDIM BOTANICO", "LAGO NORTE", "LAGO SUL", "NUCLEO BANDEIRANTE",
            "PARANOA", "PARK WAY", "PLANALTINA", "PLANO PILOTO", "RECANTO DAS EMAS",
            "RIACHO FUNDO", "RIACHO FUNDO II", "SAMAMBAIA", "SANTA MARIA", "SCIA",
            "SIA", "SOBRADINHO", "SOBRADINHO II", "SOL NASCENTE E POR DO SOL",
            "SUDOESTE/OCTOGONAL", "SAO SEBASTIAO", "TAGUATINGA", "VARJAO",
            "VICENTE PIRES", "AGUAS CLARAS"
        ]
        lookup = {ra: ra for ra in static_ras}
    
    # Aliases comuns de normalização
    aliases = {
        "SCIA (ESTRUTURAL)": "SCIA",
        "SOL NASCENTE/POR DO SOL": "SOL NASCENTE E POR DO SOL",
        "SOL NASCENTE/POR DO SOL RES": "SOL NASCENTE E POR DO SOL",
        "SAO SEBASTIAO": "SAO SEBASTIAO",
        "CEILANDIA": "CEILANDIA",
        "BRAZLANDIA": "BRAZLANDIA",
        "GUARA": "GUARA",
        "PARANOA": "PARANOA",
        "ITAPOA": "ITAPOA",
        "AGUAS CLARAS": "AGUAS CLARAS",
        "JARDIM BOTANICO": "JARDIM BOTANICO",
        "NUCLEO BANDEIRANTE": "NUCLEO BANDEIRANTE",
        "CANDANGOLANDIA": "CANDANGOLANDIA",
    }
    
    # Garantir que todos os aliases apontem para as chaves normalizadas no lookup
    for alias, canonical_key in aliases.items():
        key_alias = sanitizar_texto(alias)
        key_canonical = sanitizar_texto(canonical_key)
        if key_canonical in lookup:
            lookup[key_alias] = lookup[key_canonical]
        else:
            lookup[key_alias] = key_canonical

    # Reagrupa RAs desmembradas (Sol Nascente, Arniqueira, ...) na RA de origem, de modo
    # que o conjunto canônico de RAs (values) seja idêntico ao do dataset consolidado.
    lookup = {chave: consolidar_ra(valor) for chave, valor in lookup.items()}
    return lookup

def padronizar_regioes_administrativas(valor, lookup: dict[str, str] | None = None) -> str | None:
    """
    Padroniza os nomes das Regiões Administrativas para maiúsculo e sem acento.
    Mapeia aliases e resolve variações de digitação ou acentuação.
    
    Parâmetros:
        valor: String contendo o nome da RA.
        lookup: Dicionário opcional pré-carregado de RAs canônicas.
        
    Retorna:
        Nome da RA padronizado ou None caso seja inválido ou "NAO INFORMADO".
    """
    chave = sanitizar_texto(valor)
    if chave is None or chave == "NAO INFORMADO":
        return None
    lookup = lookup or busca_ra_canonica()

    res = lookup.get(chave, chave)
    res = sanitizar_texto(res) if res else None
    return consolidar_ra(res)
