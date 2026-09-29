# Protocolo de comparação

Vale para todo experimento em `experiments/`. Origem do baseline: commit `402f845`.

## Divisão temporal

- **Equivalência (T9):** treino com semanas antes de 2025-01-01, teste em 2025 inteiro,
  modelo treinado uma vez (`evaluation.split.dividir_treino_teste_temporal`).
- **Pesquisa (depois da T9, decisão aberta D-T1-1):** origem expansiva com re-treino, igual
  para todos os modelos. Proposta: origens a cada 4 semanas desde 2024-01-07; treino com
  amostras cujo alvo foi observado até a origem menos 4 semanas
  (`origens_expansivas`, `amostras_ate_origem`).
- Tuning, se houver, só dentro do treino, com `dividir_cv_temporal(n_splits=5, gap=4)`.

## Horizontes e informação disponível

- Horizontes de 1 a 4 semanas, forecast direto (um modelo por k), mais o modo nowcast.
- Na origem t as features vão até a semana t−1. "Direto k" prevê y(t+k), ou seja, k+1
  semanas depois da última observação. Todo modelo novo recebe o mesmo conjunto de informação.
- Só colunas defasadas (`*_lag_*`), coberturas, sazonais e a identidade da RA entram como
  features. Colunas contemporâneas ficam no painel apenas para diagnóstico.

## Métricas

- Pontuais no DF agregado: MAE, RMSE, R², sMAPE, hit rate de picos; por RA: MAE, RMSE, MASE.
- MASE contra naive alinhado ao horizonte (headline) e contra seasonal naive (diagnóstico).
- Probabilísticas quando o modelo gera quantis (23 do CDC): WIS completo, CRPS (pinball nos
  quantis), cobertura do intervalo de 90%.
- `cobertura_previsao`: fração de linhas RA×semana com previsão. Linhas sem previsão ficam NaN
  e saem das métricas; nunca são somadas como zero.

## Execução

- `scripts/run_experiment.py eN` roda o experimento e o e0 no mesmo painel, com sementes fixas,
  e gera `RESULTADO.md`. Nenhum número vem de execução anterior.
- Desempenho declarado em artigo ou em cartão de modelo do Hugging Face não é resultado nosso.
  Ele só aparece numa seção "Referência externa (não é resultado nosso)".

## Série curta

Cerca de 300 semanas desde 2020, com 31 RAs. Reportar variação entre origens e sementes e o
número de parâmetros. Desconfiar de ganhos grandes em modelos com muitos parâmetros; exigir
ganho em mais de uma origem.
