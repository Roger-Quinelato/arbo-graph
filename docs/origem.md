# Origem do código migrado

Repositório de origem: `Roger-Quinelato/Arbovirose-Dengue`, commit `402f845`.
Incubadora: `driveHugging/arbo-graph/` do Arbovirose-Dengue, migrada para `Roger-Quinelato/arbo-graph` em 2026-09-29
a partir do commit `54c3e65` da origem (ARBO2-35). A referência de equivalência continua `402f845` até a decisão ARBO2-58.
Inventário da issue T1 (GitHub #14, Jira PA7GDP-2).

## Módulos migrados

| Origem (`src/dengue_pipeline/…`) | Destino (`arbo_core/…`) | Motivo | Ticket |
|---|---|---|---|
| `runtime/config.py`: `_resolve_dados_dir`, `_logar_origem_fonte`, `diretorio_fonte`, `seed_everything`, `HORIZONTE_PRINCIPAL`, `MAX_HORIZONTE` | `config.py` | Regra de fontes e reprodutibilidade | T2 |
| `shared_kernel/contracts.py`: `ALVO_PADRAO`, `ALVO_SENSIBILIDADE`, `ForecastMode`, `WeatherPolicy`, `WatermarkPolicy`, `TemporalContext`, `ExogenousCovariateContract`, `GOOGLE_TRENDS_CONTRACT` | `contracts.py` | Alvo oficial e disponibilidade temporal | T2 |
| `modeling/feature_engineering.py`: `INICIO_SERIE_ESPACIAL` | `contracts.py` | Regra inviolável; sai do módulo de features | T2 |
| `shared_kernel/epi_calendar.py` (inteiro) | `epi_calendar.py` | Semana epidemiológica domingo–sábado | T3 |
| `shared_kernel/ra_registry.py`: `sanitizar_texto`, `RA_CONSOLIDACAO`, `consolidar_ra`, `FAMILIA_DENGUE`, `busca_ra_canonica`, `padronizar_regioes_administrativas`, `carregar_historico_populacao` | `ras.py` | Universo e normalização de RAs | T3 |
| `etl/case_ingestion.py` (inteiro) | `etl/cases.py` | Casos, máscaras de alvo, disponibilidade | T3 |
| `modeling/feature_engineering.py`: `construir_dataset_consolidado` (grade, lags de casos, tendência, sazonais) | `panel.py` | Painel RA×semana | T3 |
| `etl/provenance.py`: `calcular_hash` e detecção de período | `manifest.py` | Base do manifesto | T3 |
| `etl/weather_ingestion.py`, `etl/climate_station_ingestion.py`; `LAG_WINDOWS`, `CLIMA_RA_LAGS`, `_adicionar_features_clima_ra` | `etl/climate.py`, `panel.py` | Clima de ponto único e por RA | T4 |
| `etl/vector_ingestion.py`, `etl/trends_ingestion.py`; `OVI_LAGS`, `GT_LAGS`, `_adicionar_features_google_trends`, parte ovitrampas de `_adicionar_features_geo_ovi` | `etl/vector.py`, `etl/trends.py`, `panel.py` | Ovitrampas e Trends | T5 |
| `etl/geo_ingestion.py`; `VIZ_LAGS`, parte vizinhança de `_adicionar_features_geo_ovi` | `graph.py`, `panel.py` | Adjacência e `viz_cases_lag_*` | T6 |
| `modeling/forecast_identity.py`, `modeling/evaluation.py` (quantis, WIS, cobertura, MASE, baselines naive/sazonal), `train_tuning.dividir_treino_teste_temporal`, `TimeSeriesSplit(n_splits=5, gap=4)` | `evaluation/` | Protocolo de comparação | T7 |
| `train_tuning.py`: `HurdleRegressor`, `fabrica_modelos`, `executar_ajuste_previsao`; `feature_engineering.py`: `obter_configuracao_features`, `construir_pipeline_modelo`; `orchestration.executar_forecast_direto_multihorizonte` | `experiments/e0_baseline_rf_xgb/` | Baseline de referência | T8 |

## Deixados para trás

| Origem | Motivo |
|---|---|
| `modeling/conformal_prediction.py` | Calibração pós-hoc; não entra na comparação por padrão (decisão aberta D-T1-2) |
| `modeling/hierarchy.py` e `_prever_hierarquico_reconciliado` | Reconciliação RA↔DF; decisão aberta D-T1-2 |
| `orchestration.executar_estudo_ablacao`, `_selecionar_campeao`, `otimizar_hiperparametros` | Seleção de campeão e tuning: o baseline do repo novo é fixado por config, não reescolhido a cada execução |
| `orchestration.executar_validacao_temporal` (forecast fechado recursivo) | Modo recursivo; o repo novo usa forecast direto |
| `etl/census_ingestion.py`, `etl/national_ingestion.py` | Fora por enquanto (decisão 4) |
| `reporting/`, `api.py`, `utils/scaffold.py`, `runtime/audit.py`, `runtime/pipeline_runner.py` | Relatórios e orquestração do repo antigo |
| `scripts/` exceto a lógica de `preprocessar_dados_demograficos.py` | Notebook, README, PDFs e Kaggle |
| Testes da raiz | Recriados com fixtures sintéticas (T3–T7) |

## Achados do inventário

1. **Dependência escondida de `dados_processados/`.** Três entradas não passam por
   `diretorio_fonte()`:
   - `populacao_historica.csv` (lido por `carregar_historico_populacao`). Ele define o universo de RAs da grade;
   - `dados_clima_cache.csv` (lido por `carregar_cache_climatico`);
   - `populacao.csv`, a entrada de `preprocessar_dados_demograficos.py`.

   No repo novo elas viram fontes nomeadas (`populacao`, `clima_cache`) resolvidas por
   `diretorio_fonte()`. O pipeline não lê nada de `dados_processados/` como entrada.
2. **O baseline RF/XGBoost é um Hurdle.** `fabrica_modelos` embrulha RF, XGB e XGB_TWEEDIE em
   `HurdleRegressor`, com classificador XGB e limiar 0,35. Para haver equivalência, o e0 leva o
   Hurdle junto. Isso corrige o plano, que listava "hurdle" entre os itens deixados para trás.
3. **O baseline não é fixo.** O runner escolhe config e modelo por ablação (`winner`) a cada
   execução. O repo novo fixa o baseline em config explícita (ver `features_baseline.md`).
4. **Zero legítimo na grade.** Linha sem RA é excluída antes da agregação. Célula RA×semana sem
   caso vira 0, que é contagem observada, não ausência. Ovitrampas e Trends preenchem lags com 0
   junto de uma flag de cobertura (exceção documentada, decisão 5).
5. **Nenhuma fonte bruta no container de desenvolvimento.** `../dados`, `dados_processados/` e
   `resultados_modelagem/` não existem aqui. A equivalência (T9) só roda na máquina com as fontes.
