"""Simulador Copa 2026 — agenda jogo a jogo + simulação por partida."""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from simcopa.data.countries import BY_ISO3, FIFA_CODES, flag_url, name_pt
from simcopa.model.dixon_coles import match_probs, score_matrix
from simcopa.tournament.fixtures import (
    Match,
    by_date,
    load_fixtures,
    load_results,
    matches_with_results,
    save_results,
    standings_for_group,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "processed" / "dc_params.pkl"

st.set_page_config(page_title="Simulador Copa 2026", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; max-width: 1100px; }
    h1 { letter-spacing: -1px; }
    .hero { background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #10b981 100%);
            padding: 20px 24px; border-radius: 14px; margin-bottom: 18px; }
    .hero h1 { color: white; margin: 0; font-size: 1.9rem; }
    .hero p { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 0.9rem; }
    .match-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                  border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }
    .match-played { border-left: 3px solid #10b981; }
    .match-pending { border-left: 3px solid rgba(255,255,255,0.15); }
    .group-chip { display:inline-block; padding:2px 8px; border-radius:8px;
                  background: rgba(16,185,129,0.15); color:#10b981; font-size:0.75rem;
                  font-weight:600; letter-spacing:0.5px; }
    .day-header { font-size:1.1rem; font-weight:700; margin: 18px 0 10px 0;
                  color:#10b981; }
    .score-big { font-size: 1.6rem; font-weight: 700; padding: 0 14px; }
    .vs-mid { opacity: 0.5; font-weight: 600; }
    div[data-testid="stHorizontalBlock"] { align-items: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Rode: `simcopa init && simcopa ingest && simcopa fit`")
    st.stop()


@st.cache_resource
def load_model():
    return pickle.loads(MODEL_PATH.read_bytes())


params = load_model()
rng = np.random.default_rng()


# ===== HERO =====
all_matches = matches_with_results()
n_played = sum(1 for m in all_matches if m.played())
total = len(all_matches)
st.markdown(
    f'<div class="hero">'
    f"<h1>⚽ Simulador Copa do Mundo 2026</h1>"
    f"<p>Agenda jogo a jogo · {n_played}/{total} partidas registradas · "
    f"modelo Dixon-Coles</p>"
    f"</div>",
    unsafe_allow_html=True,
)


def render_match_card(m: Match):
    """Renderiza 1 partida com bandeiras, placar (ou botão), e ações."""
    container = st.container()
    with container:
        status_class = "match-played" if m.played() else "match-pending"
        st.markdown(
            f'<div class="match-card {status_class}">'
            f'<span class="group-chip">Grupo {m.group} · Rodada {m.md}</span>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns([2.4, 0.7, 0.5, 0.7, 2.4])
        with c1:
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:10px; justify-content:flex-end;">'
                f'<span style="text-align:right; font-weight:600;">{name_pt(m.home)}</span>'
                f'<img src="{flag_url(m.home, h=36)}" height="28"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            if m.played():
                st.markdown(
                    f'<div class="score-big" style="text-align:right">{m.home_score}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="score-big" style="text-align:right">·</div>',
                            unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="vs-mid" style="text-align:center">×</div>',
                        unsafe_allow_html=True)
        with c4:
            if m.played():
                st.markdown(
                    f'<div class="score-big">{m.away_score}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="score-big">·</div>', unsafe_allow_html=True)
        with c5:
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:10px;">'
                f'<img src="{flag_url(m.away, h=36)}" height="28"/>'
                f'<span style="font-weight:600;">{name_pt(m.away)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ações
        if m.played():
            badge = "🎲 Simulado" if m.status == "simulated" else "✅ Resultado real"
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            ac1.caption(badge)
            if ac2.button("✏️ Editar", key=f"edit_{m.id}", use_container_width=True):
                st.session_state[f"editing_{m.id}"] = True
            if ac3.button("🗑️ Limpar", key=f"clear_{m.id}", use_container_width=True):
                results = load_results()
                results.pop(m.id, None)
                save_results(results)
                st.rerun()
        else:
            probs = match_probs(params, m.home, m.away, neutral=True)
            eh, ea = probs["expected_goals"]
            top3 = " · ".join(f"**{r}-{c}** {p:.0%}" for r, c, p in probs["top3_scores"])
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            ac1.caption(
                f"📊 {name_pt(m.home)} {probs['p_home']:.0%} · "
                f"empate {probs['p_draw']:.0%} · "
                f"{name_pt(m.away)} {probs['p_away']:.0%} · "
                f"gols esperados **{eh:.1f}:{ea:.1f}**  ·  {top3}"
            )
            if ac2.button("🎲 Simular", key=f"sim_{m.id}", use_container_width=True,
                          type="primary"):
                mat = score_matrix(params, m.home, m.away, neutral=True)
                flat = mat.ravel() / mat.sum()
                k = rng.choice(flat.size, p=flat)
                hs, as_ = divmod(int(k), mat.shape[1])
                results = load_results()
                results[m.id] = {"home_score": hs, "away_score": as_,
                                  "status": "simulated"}
                save_results(results)
                st.rerun()
            if ac3.button("✏️ Inserir", key=f"input_{m.id}", use_container_width=True):
                st.session_state[f"editing_{m.id}"] = True

        # form de edição manual
        if st.session_state.get(f"editing_{m.id}"):
            with st.form(key=f"form_{m.id}"):
                fc1, fc2, fc3 = st.columns([1, 1, 1])
                hs = fc1.number_input(
                    name_pt(m.home), min_value=0, max_value=15,
                    value=m.home_score if m.home_score is not None else 0,
                    key=f"hs_{m.id}",
                )
                as_ = fc2.number_input(
                    name_pt(m.away), min_value=0, max_value=15,
                    value=m.away_score if m.away_score is not None else 0,
                    key=f"as_{m.id}",
                )
                save = fc3.form_submit_button("💾 Salvar (real)", use_container_width=True)
                if save:
                    results = load_results()
                    results[m.id] = {"home_score": int(hs), "away_score": int(as_),
                                      "status": "actual"}
                    save_results(results)
                    st.session_state[f"editing_{m.id}"] = False
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


tab_agenda, tab_groups, tab_match, tab_about = st.tabs(
    ["📅 Agenda", "🏆 Grupos", "⚔️ Confronto", "ℹ️ Sobre"]
)


# ===== AGENDA =====
with tab_agenda:
    cont_top, cont_bulk = st.columns([3, 1])
    with cont_top:
        st.caption(
            "Cada partida tem probabilidades do modelo. Clique **Simular** para sortear um placar "
            "pela distribuição prevista, ou **Inserir** pra colocar o resultado real."
        )
    with cont_bulk:
        if st.button("🎲 Simular todos os pendentes", use_container_width=True):
            results = load_results()
            for m in all_matches:
                if not m.played():
                    mat = score_matrix(params, m.home, m.away, neutral=True)
                    flat = mat.ravel() / mat.sum()
                    k = rng.choice(flat.size, p=flat)
                    hs, as_ = divmod(int(k), mat.shape[1])
                    results[m.id] = {"home_score": hs, "away_score": as_,
                                      "status": "simulated"}
            save_results(results)
            st.rerun()

    # filtros
    fc1, fc2 = st.columns([1, 1])
    show_filter = fc1.radio("Mostrar", ["Todos", "Pendentes", "Jogados"],
                             horizontal=True, label_visibility="collapsed")
    group_filter = fc2.multiselect(
        "Grupos", options=sorted({m.group for m in all_matches}),
        placeholder="Filtrar por grupo (vazio = todos)",
    )

    grouped = by_date(all_matches)
    for date_str, day_matches in grouped.items():
        # aplicar filtros
        day_matches = [m for m in day_matches
                       if (not group_filter or m.group in group_filter)
                       and (show_filter == "Todos"
                            or (show_filter == "Pendentes" and not m.played())
                            or (show_filter == "Jogados" and m.played()))]
        if not day_matches:
            continue
        dt = datetime.fromisoformat(date_str)
        dia_semana = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][dt.weekday()]
        st.markdown(
            f'<div class="day-header">📅 {dia_semana} · {dt.strftime("%d/%m/%Y")}'
            f' <span style="opacity:0.5; font-weight:400; font-size:0.9rem">'
            f'({len(day_matches)} jogo{"s" if len(day_matches) > 1 else ""})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for m in sorted(day_matches, key=lambda x: (x.group, x.id)):
            render_match_card(m)


# ===== GRUPOS / TABELA =====
with tab_groups:
    st.caption("Classificação dos grupos com base nos jogos já registrados.")
    groups_letters = sorted({m.group for m in all_matches})
    for row_start in range(0, len(groups_letters), 2):
        cols = st.columns(2)
        for ci, gi in enumerate(range(row_start, min(row_start + 2, len(groups_letters)))):
            g = groups_letters[gi]
            with cols[ci]:
                st.markdown(f"#### Grupo {g}")
                rows = standings_for_group(g, all_matches)
                df = pd.DataFrame([
                    {"": flag_url(r["team"], h=30), "Seleção": name_pt(r["team"]),
                     "P": r["P"], "V": r["V"], "E": r["E"], "D": r["D"],
                     "GP": r["GP"], "GC": r["GC"], "SG": r["SG"], "Pts": r["Pts"]}
                    for r in rows
                ])
                df.index = range(1, len(df) + 1)
                st.dataframe(
                    df,
                    column_config={"": st.column_config.ImageColumn("", width="small")},
                    use_container_width=True, height=190,
                )


# ===== CONFRONTO LIVRE =====
with tab_match:
    fifa_teams = sorted([t for t in params.teams if t in FIFA_CODES], key=name_pt)
    st.markdown("### Probabilidade de um confronto isolado")
    c1, c2 = st.columns(2)
    h = c1.selectbox("🏠 Time A", fifa_teams, key="match_h", format_func=name_pt)
    a = c2.selectbox("🛫 Time B", fifa_teams, key="match_a", format_func=name_pt,
                      index=min(1, len(fifa_teams) - 1))
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


with tab_about:
    st.markdown(
        """
        ### Como funciona

        Cada jogo tem uma **matriz de probabilidades de placar** calculada pelo modelo Dixon-Coles
        (ataque/defesa por seleção, decay temporal, correção de placares baixos).

        - **Simular esta partida**: sorteia um placar pela distribuição prevista.
        - **Inserir**: você digita o placar real quando o jogo acontecer.
        - **Simular todos pendentes**: roda uma simulação inteira da fase de grupos.
        - A **classificação** dos grupos atualiza com base nos jogos registrados.

        **Próximos passos:**
        - Mata-mata (R32 → Final) com base nos classificados
        - Persistência em volume Railway (resultados sobrevivem ao redeploy)
        - Ajuste por jogadores convocados e forma recente nos clubes
        """
    )
