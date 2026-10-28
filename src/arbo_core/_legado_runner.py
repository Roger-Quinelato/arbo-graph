# -*- coding: utf-8 -*-
"""
Executa o pipeline ANTIGO (``dengue_pipeline``) para a validação de equivalência.

Rodado em subprocesso por ``arbo_core.equivalence``, com o ``src/`` do repo antigo no
``PYTHONPATH`` e ``PIPELINE_ROOT`` apontando para uma raiz temporária. Não importa
``arbo_core``. Uso:

    python _legado_runner.py <saida> <config_features> <ano_teste> <modelos,sep,virgula> <horizontes>

Grava ``painel_antigo.parquet`` e ``metricas_antigo.csv`` em ``<saida>``. As métricas do modo
direto usam os blocos do código antigo no nível RA, sem reconciliação nem conformal
(decisão D-T1-2), para comparar com o e0.
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    saida = Path(sys.argv[1])
    config = sys.argv[2]
    ano_teste = int(sys.argv[3])
    modelos = sys.argv[4].split(",")
    horizontes = [int(k) for k in sys.argv[5].split(",")]

    from dengue_pipeline.modeling.contracts import TemporalContext
    from dengue_pipeline.modeling.evaluation import CDC_QUANTILES, consolidar_metricas_performance
    from dengue_pipeline.modeling.feature_engineering import (
        construir_dataset_consolidado,
        construir_pipeline_modelo,
        obter_colunas_entrada_pipeline,
        obter_configuracao_features,
    )
    from dengue_pipeline.modeling.forecast_identity import materializar_resultado_forecast, preparar_amostras_horizonte
    from dengue_pipeline.modeling.train_tuning import (
        _preparar_entrada_pipeline,
        _prever_com_pipeline,
        _usa_loss_contagem,
        dividir_treino_teste_temporal,
        executar_ajuste_previsao,
        fabrica_modelos,
    )
    from dengue_pipeline.runtime.config import seed_everything

    seed_everything(42)
    painel = construir_dataset_consolidado(TemporalContext.retrospective())
    painel.to_parquet(saida / "painel_antigo.parquet", index=False)

    spec = obter_configuracao_features(config)
    colunas = obter_colunas_entrada_pipeline(config)
    linhas = []
    for nome_modelo in modelos:
        agora = executar_ajuste_previsao(painel, config, nome_modelo, ano_teste=ano_teste)
        linhas.append({"modelo": nome_modelo, "modo": "nowcast", "horizonte": 1, **agora.metrics})

        treino, teste = dividir_treino_teste_temporal(painel, ano_teste)
        historico = pd.concat([treino, teste], ignore_index=True)[["epi_sunday", "RA", "cases"]]
        parametros = {"multi_quantile": True, "quantiles": CDC_QUANTILES} if nome_modelo == "XGB" else None
        for k in horizontes:
            treino_k = _preparar_entrada_pipeline(preparar_amostras_horizonte(treino, spec["target"], k), config)
            teste_k = _preparar_entrada_pipeline(preparar_amostras_horizonte(teste, spec["target"], k), config)
            modelo = fabrica_modelos(nome_modelo, parametros=dict(parametros) if parametros else None)
            pipeline = construir_pipeline_modelo(config, modelo, transform_target=not _usa_loss_contagem(nome_modelo))
            pipeline.fit(treino_k[colunas], treino_k["target_k"].astype(float))
            preds = _prever_com_pipeline(pipeline, teste_k, config, spec)
            pontual = preds[:, preds.shape[1] // 2] if preds.ndim == 2 else preds
            pred_df = materializar_resultado_forecast(teste_k, pontual)
            if preds.ndim == 2:
                for i, tau in enumerate(CDC_QUANTILES):
                    pred_df[f"q_{tau}"] = preds[:, i]
            metricas, _ = consolidar_metricas_performance(pred_df, historico_total=historico, horizonte=k)
            linhas.append({"modelo": nome_modelo, "modo": "direto", "horizonte": k, **metricas})

    pd.DataFrame(linhas).to_csv(saida / "metricas_antigo.csv", index=False)


if __name__ == "__main__":
    main()
