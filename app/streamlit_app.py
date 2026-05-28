"""Simulador Copa 2026 — UI profissional com bandeiras."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from simcopa.data.countries import BY_ISO3, FIFA_CODES, flag_url, name_pt
from simcopa.model.dixon_coles import match_probs
from simcopa.tournament.simulate import monte_carlo
from simcopa.tournament.structure import GROUPS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "processed" / "dc_params.pkl"
GROUPS_PATH = ROOT / "data" / "processed" / "groups.json"

st.set_page_config(
    page_title="Simulador Copa 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom para visual mais polido
st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; }
    h1 { letter-spacing: -1px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 12px 16px;
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
    .team-row { display: flex; align-items: center; gap: 8px; }
    .team-row img { border-radius: 2px; box-shadow: 0 0 0 1px rgba(0,0,0,0.3); }
    div[data-testid="stDataFrame"] td { vertical-align: middle; }
    .hero { background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #10b981 100%);
            padding: 24px 28px; border-radius: 16px; margin-bottom: 20px; }
    .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
    .hero p { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Rode: `simcopa init && simcopa ingest && simcopa fit`")
    st.stop()

params = pickle.loads(MODEL_PATH.read_bytes())

# só seleções FIFA na UI
fifa_teams = [t for t in params.teams if t in FIFA_CODES]
fifa_teams.sort(key=lambda t: name_pt(t))


@st.cache_data(show_spinner=False)
def cached_monte_carlo(groups_signature: str, n_sims: int, model_mtime: float) -> pd.DataFrame:
    """Roda Monte Carlo com cache pra economizar CPU no Railway."""
    groups = json.loads(groups_signature)
    return monte_carlo(params, groups, n_sims=n_sims)


# Hero
st.markdown(
    '<div class="hero">'
    "<h1>⚽ Simulador Copa do Mundo 2026</h1>"
    "<p>Modelo Dixon-Coles + Monte Carlo · 48 seleções FIFA · "
    f"calibrado com {len(params.teams)} times e pesos por tipo de jogo</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ===== SIDEBAR: GRUPOS =====
with st.sidebar:
    st.markdown("### 🏆 Grupos da Copa")
    st.caption("Selecione os 4 times de cada grupo (A–L).")

    if GROUPS_PATH.exists():
        groups = json.loads(GROUPS_PATH.read_text())
    else:
        groups = {g: ["", "", "", ""] for g in GROUPS}

    team_options = [""] + fifa_teams
    edited = {}
    # 2 grupos por linha (6 linhas × 2)
    for row_start in range(0, len(GROUPS), 2):
        cols = st.columns(2)
        for col_i, gi in enumerate(range(row_start, min(row_start + 2, len(GROUPS)))):
            g = GROUPS[gi]
            with cols[col_i]:
                st.markdown(f"**Grupo {g}**")
                slots = []
                for i in range(4):
                    cur = groups.get(g, ["", "", "", ""])[i]
                    idx = team_options.index(cur) if cur in team_options else 0
                    sel = st.selectbox(
                        f"{g}{i+1}", team_options, index=idx,
                        format_func=lambda t: f"{name_pt(t)}" if t else "—",
                        key=f"sel_{g}_{i}", label_visibility="collapsed",
                    )
                    slots.append(sel)
                edited[g] = slots

    if st.button("💾 Salvar grupos", use_container_width=True):
        GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GROUPS_PATH.write_text(json.dumps(edited, indent=2))
        st.success("Grupos salvos.")

    st.divider()
    n_sims = st.select_slider(
        "Simulações Monte Carlo",
        options=[1000, 2500, 5000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    run = st.button("🎲 Rodar simulação", type="primary", use_container_width=True)


# ===== TABS PRINCIPAIS =====
tab_probs, tab_match, tab_ranking, tab_about = st.tabs(
    ["📊 Probabilidades", "⚔️ Confronto direto", "🏅 Ranking", "ℹ️ Sobre"]
)


def render_team_html(iso3: str, h: int = 24) -> str:
    url = flag_url(iso3, h=h * 2)
    return (
        f'<div class="team-row">'
        f'<img src="{url}" height="{h}" alt="{iso3}"/>'
        f'<span><b>{name_pt(iso3)}</b> <span style="opacity:0.6">({iso3})</span></span>'
        f'</div>'
    )


with tab_probs:
    valid = all(all(t for t in v) for v in edited.values())
    if not valid:
        st.info(
            "👈 Preencha os **48 times** nos 12 grupos para rodar a simulação. "
            "Use o seletor da barra lateral."
        )
    elif run or "last_probs" in st.session_state:
        if run:
            with st.spinner(f"Rodando {n_sims:,} torneios..."):
                df = cached_monte_carlo(
                    json.dumps(edited, sort_keys=True),
                    int(n_sims),
                    MODEL_PATH.stat().st_mtime,
                )
            st.session_state["last_probs"] = df
        df = st.session_state["last_probs"].copy()
        df["flag"] = df["team"].map(lambda t: flag_url(t, h=40))
        df["Seleção"] = df["team"].map(name_pt)
        cols_pct = ["p_group_1", "p_advance", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
        labels = {
            "p_group_1": "1º grupo", "p_advance": "Passar fase",
            "p_r16": "Oitavas", "p_qf": "Quartas", "p_sf": "Semi",
            "p_final": "Final", "p_champion": "🏆 Campeão",
        }

        # Pódio
        top3 = df.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i]
            with cols[i]:
                st.markdown(
                    f'<div style="text-align:center; padding:16px; '
                    f'background:rgba(255,255,255,0.04); border-radius:12px;">'
                    f'<div style="font-size:2rem">{medal}</div>'
                    f'<img src="{flag_url(row["team"], h=60)}" height="48"/>'
                    f'<div style="font-size:1.15rem; font-weight:600; margin-top:8px">'
                    f'{name_pt(row["team"])}</div>'
                    f'<div style="font-size:1.4rem; color:#10b981; font-weight:700; margin-top:4px">'
                    f'{row["p_champion"]:.1%}</div>'
                    f'<div style="opacity:0.6; font-size:0.85rem">campeão</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Probabilidades por seleção")
        show = df[["flag", "Seleção"] + cols_pct].rename(columns=labels)
        st.dataframe(
            show,
            column_config={
                "flag": st.column_config.ImageColumn("", width="small"),
                "Seleção": st.column_config.TextColumn("Seleção", width="medium"),
                **{labels[c]: st.column_config.ProgressColumn(
                    labels[c], format="%.1f%%", min_value=0, max_value=1)
                   for c in cols_pct},
            },
            hide_index=True,
            use_container_width=True,
            height=min(48 * 25 + 38, 720),
        )

with tab_match:
    st.markdown("### Probabilidade de um confronto isolado")
    c1, c2 = st.columns(2)
    h = c1.selectbox(
        "🏠 Time A", fifa_teams, key="match_h",
        format_func=lambda t: name_pt(t),
    )
    a = c2.selectbox(
        "🛫 Time B", fifa_teams, key="match_a",
        format_func=lambda t: name_pt(t),
        index=min(1, len(fifa_teams) - 1),
    )
    neutral = st.checkbox("Campo neutro", value=True)
    if h and a and h != a:
        probs = match_probs(params, h, a, neutral=neutral)

        c_l, c_m, c_r = st.columns([1, 1, 1])
        with c_l:
            st.markdown(
                f'<div style="text-align:center; padding:16px;">'
                f'<img src="{flag_url(h, h=80)}" height="56"/>'
                f'<div style="font-size:1.2rem; font-weight:600; margin-top:8px">{name_pt(h)}</div>'
                f'<div style="font-size:1.6rem; color:#10b981; font-weight:700">'
                f'{probs["p_home"]:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with c_m:
            st.markdown(
                f'<div style="text-align:center; padding:16px;">'
                f'<div style="font-size:2.5rem">⚖️</div>'
                f'<div style="opacity:0.7; margin-top:8px">Empate</div>'
                f'<div style="font-size:1.6rem; color:#9ca3af; font-weight:700">'
                f'{probs["p_draw"]:.1%}</div>'
                f'<div style="opacity:0.6; margin-top:12px">Placar mais provável:<br/>'
                f'<b>{probs["mode_score"][0]} × {probs["mode_score"][1]}</b></div></div>',
                unsafe_allow_html=True,
            )
        with c_r:
            st.markdown(
                f'<div style="text-align:center; padding:16px;">'
                f'<img src="{flag_url(a, h=80)}" height="56"/>'
                f'<div style="font-size:1.2rem; font-weight:600; margin-top:8px">{name_pt(a)}</div>'
                f'<div style="font-size:1.6rem; color:#10b981; font-weight:700">'
                f'{probs["p_away"]:.1%}</div></div>',
                unsafe_allow_html=True,
            )

with tab_ranking:
    st.markdown("### Força das seleções (modelo Dixon-Coles)")
    st.caption("Força líquida = ataque (α) − defesa (β). Quanto maior, melhor.")
    rows = []
    for i, t in enumerate(params.teams):
        if t not in FIFA_CODES:
            continue
        rows.append({
            "team": t,
            "flag": flag_url(t, h=40),
            "Seleção": name_pt(t),
            "Confederação": BY_ISO3[t]["confederation"],
            "Ataque (α)": float(params.alpha[i]),
            "Defesa (β)": float(params.beta[i]),
            "Força líquida": float(params.alpha[i] - params.beta[i]),
        })
    rk = pd.DataFrame(rows).sort_values("Força líquida", ascending=False).reset_index(drop=True)
    rk.index = rk.index + 1
    st.dataframe(
        rk.drop(columns=["team"]),
        column_config={
            "flag": st.column_config.ImageColumn("", width="small"),
            "Ataque (α)": st.column_config.NumberColumn(format="%+.2f"),
            "Defesa (β)": st.column_config.NumberColumn(format="%+.2f"),
            "Força líquida": st.column_config.ProgressColumn(
                "Força líquida", format="%+.2f",
                min_value=float(rk["Força líquida"].min()),
                max_value=float(rk["Força líquida"].max()),
            ),
        },
        use_container_width=True,
        height=720,
    )

with tab_about:
    st.markdown(
        """
        ### Como funciona

        **Modelo:** Dixon-Coles (1997) com decay temporal e pesos por tipo de torneio.

        - **α (ataque)** e **β (defesa)** por seleção, calibrados em ~3 mil jogos
          entre seleções FIFA desde 2018.
        - **Decay** `xi = 0.0019/dia` (meia-vida ~1 ano) — jogos recentes pesam mais.
        - **Pesos por torneio** equilibram CONMEBOL (round-robin denso) vs UEFA
          (qualifying com grupos fracos):
          - Copa do Mundo: ×1.6
          - Euro/Copa América: ×1.5
          - Eliminatórias: ×1.0
          - Nations League: ×1.1
          - Amistosos: ×0.5
        - **Correção Dixon-Coles** ajusta probabilidade de placares baixos (0-0, 1-0, 1-1).
        - **Monte Carlo:** simulamos o torneio N vezes e contamos quantas vezes cada
          time chega em cada fase.

        ### Dados

        - 49.257 jogos internacionais 1872→hoje (Mart Jürisoo, atualizado).
        - Filtro: apenas seleções FIFA (211 membros).
        - Inclui amistosos e Nations League, mas com peso menor que torneios oficiais.

        ### Roadmap

        - Chaveamento oficial do mata-mata (FIFA 2026)
        - Elo ratings como prior bayesiano
        - Forma individual dos jogadores nos clubes (FBref/Transfermarkt)
        - Inserção de placares ao vivo durante a Copa, com recálculo das fases abertas
        """
    )
