# arbo-graph (incubadora)

Core de dados e validação para testar previsão espaço-temporal de dengue por RA do DF (grafos,
Chronos-2), comparável ao baseline RF/XGBoost do Arbovirose-Dengue.

Incubado em `driveHugging/arbo-graph/` do Arbovirose-Dengue até 2026-09-29 e migrado para este
repositório no commit `54c3e65` da origem.
Origem do código: commit `402f845` (ver `docs/origem.md`). Plano: [`PLANO_REPO_NOVO.md`](https://github.com/Roger-Quinelato/Arbovirose-Dengue/blob/main/driveHugging/PLANO_REPO_NOVO.md).

## Regras invioláveis

- Painel a partir de 2020-01-01 (`contracts.INICIO_SERIE_ESPACIAL`).
- Alvo `contracts.ALVO_PADRAO = "simples_caso_provavel"`.
- Fontes brutas só por `config.diretorio_fonte()`; `pendrive/` e LAI-011436-2026 levantam erro.
- Ausência de RA ou de previsão nunca vira zero. Exceção documentada: lags de ovitrampas e
  Trends preenchidos com 0 ao lado de flag de cobertura.
- Nenhum dado versionado. Fontes em `ARBO_DADOS_DIR` (padrão: pasta `dados` irmã do projeto).
- Desempenho de artigo ou de modelo do Hub não é resultado nosso.

## Documentação

- `docs/dados.md`: fontes, cobertura e limitações.
- `docs/decisoes.md`: decisões em uma linha cada.
- `docs/protocolo.md`: regras de comparação com o baseline.
- `docs/equivalencia.md`: portão de equivalência e achados sobre o pipeline antigo.
- `docs/origem.md`, `features_baseline.md`, `validacao_atual.md`: inventário da migração.

## Uso

```bash
pip install -e ".[test]"          # core
pip install -e ".[baseline,test]" # + RF/XGBoost
pytest

ARBO_DADOS_DIR=/caminho/dados python -m arbo_core.prepare        # painel + manifesto
python scripts/run_experiment.py e0_baseline_rf_xgb              # baseline
python scripts/check_equivalence.py                               # portão T9
```

O portão de equivalência e o teste `test_equivalencia_ponta_a_ponta_com_pipeline_antigo` rodam o
pipeline antigo em subprocesso. Eles precisam de um checkout do Arbovirose-Dengue em
`ARBO_REPO_ANTIGO` (padrão: pasta irmã `Arbovirose-Dengue`) e das dependências dele
(`pip install -r ../Arbovirose-Dengue/requirements.txt`, que exige Python ≥ 3.12). Sem isso o
teste é pulado.
