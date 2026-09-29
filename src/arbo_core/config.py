# -*- coding: utf-8 -*-
"""
Resolução de caminhos e regra de fontes.

Origem: ``dengue_pipeline/runtime/config.py`` (commit 402f845). Só a parte de fontes,
sementes e horizontes foi migrada; caminhos de artefatos do pipeline antigo ficaram para trás.
"""

import functools
import logging
import os
import random
from pathlib import Path

_logger = logging.getLogger(__name__)


def _resolve_base_dir() -> Path:
    """Raiz do projeto: ``ARBO_GRAPH_ROOT`` se definida, senão dois níveis acima de ``src/``."""
    env_root = os.getenv("ARBO_GRAPH_ROOT")
    if env_root:
        base = Path(env_root).resolve()
        if not base.exists():
            raise RuntimeError(f"ARBO_GRAPH_ROOT informada não existe: '{env_root}'")
        return base
    # config.py reside em src/arbo_core/config.py
    return Path(__file__).resolve().parents[2]


BASE_DIR = _resolve_base_dir()

# Saídas geradas, sempre dentro do projeto e fora do git.
DADOS_PROCESSADOS_DIR = BASE_DIR / "dados_processados"
RESULTADOS_DIR = BASE_DIR / "resultados"

# Fontes vetadas para qualquer uso: entrada, validação ou número reportado.
FONTES_VETADAS: tuple[str, ...] = ("pendrive", "lai-011436-2026")

HORIZONTE_PRINCIPAL = 1
MAX_HORIZONTE = 4


def _resolve_dados_dir(base: Path | None = None) -> Path:
    """Raiz das fontes externas: ``ARBO_DADOS_DIR`` ou a pasta ``dados`` irmã do projeto."""
    env_dados = os.getenv("ARBO_DADOS_DIR")
    if env_dados:
        return Path(env_dados)
    return (BASE_DIR if base is None else base).parent / "dados"


def _contem_fonte_vetada(texto: str) -> bool:
    texto = texto.lower()
    return any(vetada in texto for vetada in FONTES_VETADAS)


@functools.lru_cache(maxsize=None)
def _logar_origem_fonte(nome: str, origem: str, caminho: str) -> None:
    """Registra uma vez, por combinação, de onde cada fonte foi lida."""
    _logger.info("Fonte '%s' resolvida em %s: %s", nome, origem, caminho)


def diretorio_fonte(nome: str) -> Path:
    """
    Resolve o diretório de uma fonte bruta: o externo, se existir; senão a cópia local.

    Quando nenhuma das duas existe, devolve o caminho local para que o chamador levante o
    ``FileNotFoundError`` com a própria mensagem. Nomes ou raízes que apontem para uma fonte
    vetada levantam ``ValueError`` antes de qualquer acesso ao disco.

    A raiz externa é resolvida a cada chamada a partir do ``BASE_DIR`` corrente, e não no
    import: os testes neutralizam as duas origens trocando ``BASE_DIR``.
    """
    if _contem_fonte_vetada(nome):
        raise ValueError(f"Fonte vetada para qualquer uso: '{nome}'")

    raiz_externa = _resolve_dados_dir()
    if _contem_fonte_vetada(str(raiz_externa)):
        raise ValueError(f"Raiz de fontes aponta para fonte vetada: '{raiz_externa}'")

    externo = raiz_externa / nome
    local = BASE_DIR / nome
    if externo.exists():
        _logar_origem_fonte(nome, "fonte externa", str(externo))
        return externo
    if local.exists():
        _logar_origem_fonte(nome, "cópia local (fallback)", str(local))
        return local
    return local


def seed_everything(seed: int = 42) -> None:
    """Fixa as fontes de aleatoriedade e limita threads para determinismo."""
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
