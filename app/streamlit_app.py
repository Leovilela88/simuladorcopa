"""Streamlit: inserir resultados e ver probabilidades em tempo real."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from simcopa.model.dixon_coles import match_probs
from simcopa.tournament.simulate import monte_carlo
from simcopa.tournament.structure import GROUPS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "processed" / "dc_params.pkl"
GROUPS_PATH = ROOT / "data" / "processed" / "groups.json"

st.set_page_config(page_title="Simulador Copa 2026", layout="wide")
st.title("⚽ Simulador da Copa do Mundo 2026")

if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Rode: `simcopa init && simcopa ingest && simcopa fit`")
    st.stop()

params = pickle.loads(MODEL_PATH.read_bytes())

with st.sidebar:
    st.header("Configuração dos grupos")
    if GROUPS_PATH.exists():
        groups = json.loads(GROUPS_PATH.read_text())
    else:
        groups = {g: ["", "", "", ""] for g in GROUPS}

    team_options = [""] + params.teams
    edited = {}
    for g in GROUPS:
        st.subheader(f"Grupo {g}")
        edited[g] = [
            st.selectbox(f"{g}{i+1}", team_options,
                         index=team_options.index(groups[g][i]) if groups[g][i] in team_options else 0,
                         key=f"{g}{i}")
            for i in range(4)
        ]
    if st.button("Salvar grupos"):
        GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GROUPS_PATH.write_text(json.dumps(edited, indent=2))
        st.success("Grupos salvos.")

    n_sims = st.number_input("Simulações Monte Carlo", 1000, 100_000, 10_000, step=1000)
    run = st.button("🎲 Rodar simulação")

tabs = st.tabs(["Probabilidades", "Jogo a jogo", "Sobre o modelo"])

with tabs[0]:
    valid = all(all(t for t in v) for v in edited.values())
    if not valid:
        st.info("Preencha todos os 48 times nos 12 grupos para rodar.")
    elif run:
        with st.spinner(f"Rodando {n_sims:,} torneios..."):
            df = monte_carlo(params, edited, n_sims=int(n_sims))
        st.dataframe(
            df.style.format({c: "{:.1%}" for c in df.columns if c.startswith("p_")}),
            use_container_width=True,
        )

with tabs[1]:
    st.subheader("Probabilidade de um confronto isolado")
    c1, c2 = st.columns(2)
    h = c1.selectbox("Mandante", params.teams, key="match_h")
    a = c2.selectbox("Visitante", params.teams, key="match_a")
    neutral = st.checkbox("Campo neutro", value=True)
    if h and a and h != a:
        probs = match_probs(params, h, a, neutral=neutral)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{h} vence", f"{probs['p_home']:.1%}")
        m2.metric("Empate", f"{probs['p_draw']:.1%}")
        m3.metric(f"{a} vence", f"{probs['p_away']:.1%}")
        m4.metric("Placar mais provável", f"{probs['mode_score'][0]} x {probs['mode_score'][1]}")

with tabs[2]:
    st.markdown("""
    **Modelo:** Dixon-Coles (1997) com decay temporal (xi = 0.0019/dia, meia-vida ~365 dias).

    - Força ofensiva/defensiva por seleção
    - Correção para placares baixos (0-0, 1-0, 0-1, 1-1)
    - Vantagem de mando aplicada só em jogos não neutros
    - Mata-mata: pênaltis simulados como 50/50 em caso de empate (MVP)

    **Próximos passos:**
    - Chaveamento oficial do mata-mata (FIFA)
    - Incorporar Elo + forma recente (últimos 12 meses) como prior
    - Ajuste por força do elenco convocado (xG dos jogadores)
    """)
