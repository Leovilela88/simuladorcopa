# Simulador Copa do Mundo 2026

Modelo Dixon-Coles + Monte Carlo para tentar cravar o bolão da Copa.

## Como funciona

1. **Histórico** — 50k+ jogos internacionais (1872→hoje) viram a base.
2. **Dixon-Coles** — cada seleção ganha força ofensiva/defensiva, com decay temporal (jogos recentes pesam mais). Correção para placares baixos.
3. **Monte Carlo** — simula o torneio 10k+ vezes; saída = probabilidade de campeão, top4, passar de grupo, etc.
4. **Atualização ao vivo** — a cada jogo da Copa, você insere o placar real e re-roda a simulação. As probabilidades dos cenários ainda possíveis se ajustam.

## Setup

```bash
cd simuladorcopa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

simcopa init           # cria SQLite
simcopa ingest         # baixa histórico (~5 MB)
simcopa fit --since 2018-01-01    # ajusta modelo
```

## Rodar a UI

```bash
streamlit run app/streamlit_app.py
```

Na barra lateral você preenche os 12 grupos com os 48 times. Clicando em "Rodar simulação", vê as probabilidades de cada seleção.

## CLI

```bash
simcopa simulate -n 20000 --out data/processed/probs.csv
```

## Estrutura

```
src/simcopa/
  db.py                 SQLite (teams, historical_matches, wc_matches, model_probs)
  ingest/historical.py  Dataset Mart Jürisoo
  model/dixon_coles.py  Ajuste Dixon-Coles + score_matrix
  tournament/           Estrutura da Copa 2026 + Monte Carlo
  cli.py                init / ingest / fit / simulate
app/streamlit_app.py    UI
```

## Roadmap

- [ ] Chaveamento oficial do mata-mata (FIFA confirmou em 2023)
- [ ] Elo ratings como prior bayesiano para sair do cold-start
- [ ] Forma recente (últimos 12 meses) com peso maior
- [ ] Ajuste por força do elenco (xG e minutagem dos convocados via FBref)
- [ ] Modo "fechar bolão": dado seu palpite, qual a EV?
- [ ] Backtesting: rodar o modelo nas Copas 2018/2022 e medir Brier score
