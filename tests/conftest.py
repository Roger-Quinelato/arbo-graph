import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import sintetico  # noqa: E402
from arbo_core import config  # noqa: E402


@pytest.fixture
def fontes_sinteticas(tmp_path, monkeypatch):
    """Projeto e ``ARBO_DADOS_DIR`` sintéticos; devolve a contagem esperada por (domingo, RA)."""
    projeto = tmp_path / "projeto"
    projeto.mkdir()
    raiz_dados = tmp_path / "dados"
    esperado = sintetico.escrever_fontes(raiz_dados)
    monkeypatch.setattr(config, "BASE_DIR", projeto)
    monkeypatch.setenv("ARBO_DADOS_DIR", str(raiz_dados))
    return esperado


@pytest.fixture
def fontes_completas(tmp_path, monkeypatch):
    """Como ``fontes_sinteticas``, mais geo.json, cache de clima, InfoDengue e INMET."""
    projeto = tmp_path / "projeto"
    projeto.mkdir()
    raiz_dados = tmp_path / "dados"
    esperado = sintetico.escrever_fontes_completas(raiz_dados)
    monkeypatch.setattr(config, "BASE_DIR", projeto)
    monkeypatch.setenv("ARBO_DADOS_DIR", str(raiz_dados))
    return esperado


@pytest.fixture
def fontes_conectadas(tmp_path, monkeypatch):
    """Fontes completas sem RA isolada (o pipeline antigo não suporta RA sem vizinha)."""
    projeto = tmp_path / "projeto"
    projeto.mkdir()
    raiz_dados = tmp_path / "dados"
    esperado = sintetico.escrever_fontes_completas(raiz_dados, isolada=False)
    monkeypatch.setattr(config, "BASE_DIR", projeto)
    monkeypatch.setenv("ARBO_DADOS_DIR", str(raiz_dados))
    return esperado
