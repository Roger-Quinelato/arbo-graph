"""Regra de fontes: externo antes de local, ARBO_DADOS_DIR, fontes vetadas."""

import pandas as pd
import pytest

from arbo_core import config, contracts


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    """Projeto e raiz externa sintéticos; nenhuma fonte real fica visível."""
    projeto = tmp_path / "projeto"
    projeto.mkdir()
    monkeypatch.setattr(config, "BASE_DIR", projeto)
    monkeypatch.delenv("ARBO_DADOS_DIR", raising=False)
    return projeto


def test_externo_tem_prioridade(raiz):
    externo = raiz.parent / "dados" / "info-saude"
    externo.mkdir(parents=True)
    (raiz / "info-saude").mkdir()
    assert config.diretorio_fonte("info-saude") == externo


def test_fallback_local_quando_externo_ausente(raiz, caplog):
    local = raiz / "ovitrampas"
    local.mkdir()
    config._logar_origem_fonte.cache_clear()
    with caplog.at_level("INFO", logger="arbo_core.config"):
        assert config.diretorio_fonte("ovitrampas") == local
    assert "fallback" in caplog.text


def test_nenhuma_origem_devolve_caminho_local(raiz):
    assert config.diretorio_fonte("inmet") == raiz / "inmet"
    assert not (raiz / "inmet").exists()


def test_arbo_dados_dir_respeitada(raiz, tmp_path, monkeypatch):
    outra = tmp_path / "outra_raiz"
    (outra / "inmet").mkdir(parents=True)
    monkeypatch.setenv("ARBO_DADOS_DIR", str(outra))
    assert config.diretorio_fonte("inmet") == outra / "inmet"


@pytest.mark.parametrize("nome", ["pendrive", "PENDRIVE", "LAI-011436-2026", "lai-011436-2026/tabela"])
def test_fonte_vetada_levanta_erro(raiz, nome):
    with pytest.raises(ValueError, match="vetada"):
        config.diretorio_fonte(nome)


def test_raiz_externa_vetada_levanta_erro(raiz, tmp_path, monkeypatch):
    monkeypatch.setenv("ARBO_DADOS_DIR", str(tmp_path / "pendrive"))
    with pytest.raises(ValueError, match="vetada"):
        config.diretorio_fonte("info-saude")


def test_contratos_invioláveis():
    assert contracts.ALVO_PADRAO == "simples_caso_provavel"
    assert contracts.INICIO_SERIE_ESPACIAL == pd.Timestamp("2020-01-01")
    assert contracts.TemporalContext.retrospective().target_name == contracts.ALVO_PADRAO
