# Validação de equivalência (T9)

Portão da decisão 8: os experimentos E1–E6 só começam depois que o repo novo reproduzir o
painel e as métricas do pipeline antigo (commit `402f845`) sobre as mesmas fontes.

## Como rodar (máquina com as fontes)

```bash
ARBO_DADOS_DIR=D:\arboRoger\dados python scripts/check_equivalence.py
```

O script:
1. roda o pipeline antigo em subprocesso, numa raiz temporária. `populacao_historica.csv` e
   `dados_clima_cache.csv` são copiados das fontes `populacao` e `clima_cache`. Nenhum arquivo
   do `dados_processados/` real é lido;
2. monta o painel novo e roda o e0 com a config `lag+clima-ambos+RA+geo-ovi`, RF e XGB,
   nowcast e direto k = 1..4, teste 2025;
3. compara painel (linhas, RAs, colunas, valores) e métricas, grava
   `resultados/equivalencia/equivalencia.json` e sai com código 1 se houver diferença.

As colunas do Censo que aparecem só no antigo são diferença esperada (decisão 4). O carimbo
`processing_time` é ignorado.

## Estado

| Base | Painel | Métricas | Observação |
|---|---|---|---|
| Sintética (4 RAs × 60 semanas, teste 2021) | igual (240 linhas, todas as colunas) | iguais (RF e XGB, nowcast e k=1..4) | `tests/test_equivalencia.py` |
| Real (`ARBO_DADOS_DIR`) | **pendente** | **pendente** | Fontes ausentes no ambiente de desenvolvimento |

## Achados sobre o pipeline antigo

1. **RA sem vizinha derruba o painel.** `_adicionar_features_geo_ovi` grava `pd.NA` para RA
   sem vizinha e depois chama `astype(float)`, que falha. Sob pandas 3.0.6 a falha foi
   reproduzida; sob pandas 2.x não foi verificada. O repo novo usa `np.nan`. Na base real a
   falha só aparece se alguma das RAs do painel ficar sem vizinha no `geo.json`.
2. **A reconciliação preenchia previsão ausente com 0** (`_prever_hierarquico_reconciliado`,
   `fillna(0.0)`). O e0 não reconcilia (D-T1-2): previsão ausente fica NaN e reduz
   `cobertura_previsao`.
3. **"Direto k" está k+1 semanas à frente da última observação.** Na origem t as features vão
   até `cases_lag_1` = y(t−1), e o alvo é y(t+k). O modo nowcast prevê y(t) com y(t−1). Qualquer
   modelo novo (Chronos-2, GNN) precisa receber o mesmo conjunto de informação por horizonte
   para a comparação valer.
4. **Interpolação retrospectiva da umidade usa valor futuro**
   (`WeatherPolicy.RETROSPECTIVE_INTERPOLATION`, padrão do `TemporalContext.retrospective`).
   Documentado em `tests/test_vazamento.py`. No protocolo de pesquisa use
   `OPERATIONAL_OBSERVED_ONLY`.
