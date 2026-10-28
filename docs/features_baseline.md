# Features do baseline de referência

Origem: `modeling/feature_engineering.py`, `obter_configuracao_features`, commit `402f845`.
Subtarefa T1.2 (GitHub #16, Jira PA7GDP-13).

## Grupos de colunas

| Grupo | Colunas | Fonte (`diretorio_fonte`) | Lags |
|---|---|---|---|
| Casos | `cases_lag_1..4` | `info-saude` | 1–4 |
| Tendência | `cases_delta_1`, `cases_delta_2`, `cases_growth_rate` | derivado dos lags de casos | — |
| Incidência | `incidencia_100k_lag_1..4` (só na config `incid-target`) | `info-saude` + `populacao` | 1–4 |
| Clima ponto único | `temp_mean_lag_2..4`, `precip_sum_lag_2..11`, `umidmed_lag_1..3` | `clima_cache` + `InfoDengue` | por variável |
| Clima por RA | `temp_mean_ra_lag_2..4`, `precip_sum_ra_lag_2..11`, `umid_med_ra_lag_1..3`, `clima_ra_cobertura` | `inmet` + `dados_geo` | por variável |
| Sazonais | `sin_week`, `cos_week`, `sin_month`, `cos_month` | calendário | — |
| Identidade | one-hot de `RA` | — | — |
| Vizinhança | `viz_cases_lag_1`, `viz_cases_lag_2` (média dos vizinhos) | `dados_geo` | 1–2 |
| Ovitrampas | `ovi_ipo_lag_2..4`, `ovi_ovos_media_lag_2..4`, `ovi_cobertura` | `ovitrampas` | 2–4 |
| Saneamento | `pct_rede_agua`, `pct_rede_esgoto`, `pct_lixo_coletado`, `saneamento_cobertura` | `censo` | transversal |
| Trends | `google_trends_dengue_lag_1..4`, `gt_cobertura` | `Google Trends` | 1–4 |

## Configs existentes

| Config | Grupos |
|---|---|
| `lag-only` | casos, tendência |
| `lag+clima` | + clima ponto único, sazonais |
| `lag+clima+RA` | + identidade |
| `lag+clima+RA+incid-target` | alvo incidência |
| `lag+clima+RA+geo-ovi` | + vizinhança, ovitrampas |
| `lag+clima-ra+RA+geo-ovi` | troca ponto único por clima por RA |
| `lag+clima-ambos+RA+geo-ovi` | os dois climas |
| `lag+clima-ambos+RA+geo-ovi+saneamento` | + saneamento (Censo) |
| `lag+clima-ambos+RA+geo-ovi+saneamento+trends` | + Trends |

## Baseline fixado no repo novo

O runner antigo escolhe o campeão a cada execução. Não há `campeao_ablacao_nowcasting.json`
disponível neste ambiente para confirmar qual config venceu por último. Por isso o repo novo
fixa duas configs explícitas:

- **`e0-geo`** = `lag+clima-ambos+RA+geo-ovi`: config mais completa **sem** Censo (decisão 4)
  e sem Trends. É o baseline de comparação da vizinhança (E1) e dos modelos de grafo.
- **`e0-trends`** = `e0-geo` + Trends: só em modo `RETROSPECTIVE` (`GOOGLE_TRENDS_CONTRACT`).

Modelos: `RF` e `XGB` (Hurdle, parâmetros padrão de `fabrica_modelos`, `random_state=42`).
`XGB_TWEEDIE` fica fora da primeira migração.

ETLs que entram: `info-saude`, `populacao`, `clima_cache`, `InfoDengue`, `inmet`, `dados_geo`,
`ovitrampas`, `Google Trends`. ETLs que ficam fora: `censo`, `dados-gov` (decisão 4).

Pendente de confirmação na T9: comparar essas configs com o campeão real registrado na máquina
que tem as fontes.
