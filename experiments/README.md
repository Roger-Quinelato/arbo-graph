# Experimentos

Cada experimento é uma pasta `eN_nome/` com:

- `config.toml`: parâmetros do experimento;
- `model.py`: função `executar(painel, config) -> (metricas, previsoes)`;
- `RESULTADO.md`: gerado por `scripts/run_experiment.py --publicar`, nunca escrito à mão.

## Regras

1. **O baseline roda de novo em toda execução.** `run_experiment.py eN` executa também o
   `e0_baseline_rf_xgb` sobre o mesmo painel, com as mesmas sementes e a mesma divisão.
2. **Desempenho de artigo ou de modelo do Hugging Face não é resultado nosso.** Números de
   literatura ou de cartão de modelo entram só numa seção "Referência externa (não é resultado
   nosso)" e nunca em tabela de comparação.
3. **Sem informação futura.** Só colunas defasadas do painel (`*_lag_*`, coberturas, sazonais e
   identidade da RA) entram como features.
4. **Previsão ausente não vira zero.** Linha sem previsão fica NaN e reduz `cobertura_previsao`.
5. **Série curta** (desde 2020): reportar horizontes 1–4, número de parâmetros e variação entre
   origens; desconfiar de ganhos grandes de modelos com muitos parâmetros.

| Pasta | Pergunta | Estado |
|---|---|---|
| `e0_baseline_rf_xgb` | Baseline RF/XGBoost (Hurdle) migrado | Implementado (T8) |
| `e1_viz_soma` … `e6_seir_gnn_cape` | Plano do [`driveHugging/HANDOFF.md`](https://github.com/Roger-Quinelato/Arbovirose-Dengue/blob/main/driveHugging/HANDOFF.md) no Arbovirose-Dengue | Só depois da equivalência (T9) |
