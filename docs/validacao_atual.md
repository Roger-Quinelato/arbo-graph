# Validação atual do baseline

Origem: `modeling/train_tuning.py` e `modeling/orchestration.py`, commit `402f845`.
Subtarefa T1.3 (GitHub #17, Jira PA7GDP-14).

## O que o repo antigo faz

| Etapa | Função | Treino | Teste | Horizonte |
|---|---|---|---|---|
| Ablação / "nowcasting rolling" | `executar_ajuste_previsao` | semanas < 2025-01-01, treino único | 2025 inteiro | 1 semana, lags observados |
| Forecast fechado recursivo | `executar_validacao_temporal` | < 2025, treino único | 2025, lags vindos das próprias previsões | recursivo |
| Tuning | `cv_score_parametros` | `TimeSeriesSplit(n_splits=5, gap=4)` dentro de < 2025 | dobras de validação | 1 |
| Forecast direto | `executar_forecast_direto_multihorizonte` | < 2025, um modelo por k | 2025 | k = 1..4 (`shift(-k)` por RA) |

Nenhuma etapa treina de novo a cada origem. O nome "rolling" no código se refere ao fato de a
origem avançar semana a semana em 2025 com lags observados. O modelo é o mesmo em todo o teste.

O forecast direto aplica ainda reconciliação hierárquica RA↔DF
(`_prever_hierarquico_reconciliado`) e conformal por horizonte.

## Proposta para o repo novo (decisão 1: origem móvel)

1. **Equivalência (T9):** reproduzir o esquema acima tal como está: treino único < 2025,
   teste 2025, k = 1..4, forecast direto. Sem isso não há como provar que a migração preservou
   o baseline.
2. **Protocolo de pesquisa (depois da T9):** origem expansiva com re-treino, igual para todos os
   modelos:
   - origens semanais em 2024–2025;
   - treino com todas as semanas ≤ origem − 4 (atraso de notificação, `gap=4`);
   - previsão k = 1..4.

## Decisões abertas

- **D-T1-1**: confirmar a origem expansiva do item 2 (datas de início e passo). Proposta: origens
  a cada 4 semanas a partir de 2024-01-07, para limitar custo com RF/XGB.
- **D-T1-2**: reconciliação hierárquica e conformal ficam fora da comparação com modelos novos.
  Proposta: comparar o modelo "cru" de todos (RA-level, sem reconciliação). A equivalência
  confere o caminho sem reconciliação, reproduzido chamando `executar_ajuste_previsao` e o
  forecast direto com `pipeline_k` só no nível RA.

Até o usuário responder, as tarefas seguem a proposta, que é reversível.
