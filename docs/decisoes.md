# Decisões

Uma linha por decisão. Origem: código do Arbovirose-Dengue (commit `402f845`) e decisões do
usuário de 2026-09-24 ([`driveHugging/PLANO_REPO_NOVO.md`](https://github.com/Roger-Quinelato/Arbovirose-Dengue/blob/main/driveHugging/PLANO_REPO_NOVO.md) no Arbovirose-Dengue).

| Decisão | Motivo |
|---|---|
| Painel começa em 2020-01-01 | Antes disso a ficha não identifica a RA; a grade com zero fabricaria anos sem casos |
| Alvo `simples_caso_provavel`; `familia_dengue` só como sensibilidade | Definição do Ministério da Saúde; a restrita exclui cerca de 27% dos casos prováveis |
| Registro sem RA é excluído; célula RA×semana sem caso vale 0 | Ausência de identificação não é zero; célula vazia de RA identificada é contagem observada |
| Atraso de notificação de 4 semanas | Gap da validação cruzada e janela operacional |
| Clima só observado no modo operacional | A interpolação retrospectiva da umidade usa valor futuro |
| Precipitação com lags 2–11 | Correlação com casos sobe até o lag 11 e estabiliza |
| Horizonte principal 1, máximo 4 | O erro cresce com k; o forecast direto do repo antigo não batia o naive em nenhum horizonte |
| Ovitrampas e Trends: lag ausente = 0 com flag de cobertura | Exceção documentada (decisão 5): separa "sem vigilância/índice" de "valor baixo" |
| Censo e SINAN nacional fora | Decisão 4; podem entrar depois, só no recorte 2020–2025 |
| Universo de RAs como está (31 no painel) | Decisão 6, adiada até antes da GNN |
| Baseline fixado em `lag+clima-ambos+RA+geo-ovi`, RF e XGB Hurdle | O repo antigo escolhia o campeão a cada execução |
| Sem reconciliação e sem conformal na comparação (D-T1-2) | Comparar todos os modelos no nível RA, sem pós-processamento; a reconciliação antiga zerava previsão ausente |
| Equivalência pelo esquema antigo (treino < 2025, teste 2025); depois, origem expansiva com re-treino (D-T1-1) | Provar a migração antes de mudar o protocolo |
| Só código no repo novo, fontes por `ARBO_DADOS_DIR` | Decisão 2 |
| Chronos-2 como extra opcional; pesos baixados só no E3, versão fixada | Decisão 7; nenhum dado nosso sai da máquina |
| Experimentos param se a equivalência falhar | Decisão 8 |
