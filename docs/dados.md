# Fontes de dados

Condensado do código do Arbovirose-Dengue (commit `402f845`). Todas as fontes são lidas por
`config.diretorio_fonte(nome)`, com a raiz externa `ARBO_DADOS_DIR` antes da cópia local.
Nenhum dado é versionado. `pendrive/` e LAI-011436-2026 estão vetados para qualquer uso.

| Fonte (`nome`) | Conteúdo | Período | Granularidade | Entra como | Lags | Limitações |
|---|---|---|---|---|---|---|
| `info-saude` | Notificações individuais do SINAN/InfoSaúde (CSV `;`) | desde 2017; painel usa 2020+ | registro → semana×RA | `cases`, alvo | 1–4 | Antes de 2020 quase toda ficha vem sem RA ("Não Informado" em 99% dos registros de 2017 e 2019). Registro sem RA é excluído, não zerado |
| `populacao` | `populacao_historica.csv`: população por RA e ano | 2017–2026 | ano×RA | `populacao`, `incidencia_100k`, universo de RAs | — | Projeção a partir da PDAD-A 2024 com taxas da Codeplan (1,20% a.a. desde 2020, 1,39% antes). Era lido de `dados_processados/` no repo antigo |
| `clima_cache` | `dados_clima_cache.csv`: temperatura e precipitação semanais do DF (Open-Meteo) | desde 2016 | semana, série única | `temp_mean_lag_*`, `precip_sum_lag_*` | temp 2–4, precip 2–11 | Um valor para o DF inteiro. Era lido de `dados_processados/` no repo antigo |
| `InfoDengue` | `InfoDengue_2016-2026.csv`: umidade semanal | 2016–2026 | semana, série única | `umidmed_lag_*` | 1–3 | Lacunas preenchidas por interpolação nos dois sentidos no modo retrospectivo (usa futuro); só para frente no operacional |
| `inmet` | Um zip por ano; CSV horário por estação automática | 2000–2026, **sem 2015** | hora×estação → semana×RA (IDW p=2) | `*_ra_lag_*`, `clima_ra_cobertura` | iguais ao ponto único | 5 estações no DF, nem todas na série inteira (A001 desde 2000; A042, A045, A046, A047 entram entre 2008 e 2019). RAs a mais de 15 km de qualquer estação vão para o manifesto |
| `dados_geo` | `geo.json`: polígonos das RAs (IDE-DF) | estático | RA | adjacência, centroides | — | 37 polígonos contra 31 RAs no painel (RAs desmembradas consolidadas; "26 DE SETEMBRO" e "PONTE ALTA" sem série). Centroide = média dos vértices |
| `ovitrampas` | Leituras de armadilha (SES-DF), CSV `;` | a partir de 2023 no repo antigo | leitura → semana×RA | `ovi_ipo_lag_*`, `ovi_ovos_media_lag_*`, `ovi_cobertura` | 2–4 | Digitação com semanas ou meses de atraso: a data usada é a da coleta. Lag ausente vira 0 com flag de cobertura (decisão 5) |
| `Google Trends` | Índice semanal "dengue", recorte BR-DF | 2020+ | semana, série única | `google_trends_dengue_lag_*`, `gt_cobertura` | 1–4 | Só em modo retrospectivo: o fator de calibração anual não existe para o ano corrente. Lag ausente vira 0 com flag de cobertura |
| `censo` | Agregados do Censo 2022 por setor | 2022 | RA, transversal | saneamento | — | **Fora por enquanto** (decisão 4) |
| `dados-gov` | SINAN nacional (DENGBR) | 2017–2023 | UF | validação | — | **Fora por enquanto** (decisão 4); só o recorte 2020–2023 cabe no painel |

## Proveniência e cobertura

Cada `prepare` grava `dados_processados/manifesto.json` com:
- hash SHA-256 de cada arquivo das fontes lidas;
- commit do código e versões dos contratos;
- período, RAs e colunas do painel;
- `caveats`: fontes ausentes, grupos de features pulados, RAs distantes das estações, RAs sem
  polígono e coberturas de ovitrampas, clima por RA e Trends.

Uma fonte ausente não interrompe o painel quando o grupo é opcional (clima por RA, vizinhança,
ovitrampas, Trends). A ausência fica registrada no manifesto, nunca preenchida com zero.
Casos, população e clima de ponto único são obrigatórios.
