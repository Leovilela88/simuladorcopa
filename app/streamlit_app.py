"""Simulador Copa 2026 — agenda completa (grupos + mata-mata) e bracket visual."""
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
    STAGE_LABEL,
    by_date,
    by_stage,
    delete_result,
    load_results,
    matches_with_results,
    save_results,
    standings_for_group,
    upsert_result,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "processed" / "dc_params.pkl"

st.set_page_config(page_title="Simulador Copa 2026", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .main > div { padding-top: 0.5rem; max-width: 1200px; }
    h1 { letter-spacing: -1px; }
    /* HERO */
    .hero { background: linear-gradient(135deg, #052e16 0%, #064e3b 40%, #047857 100%);
            padding: 22px 28px; border-radius: 16px; margin-bottom: 14px;
            border: 1px solid rgba(16,185,129,0.25); }
    .hero h1 { color: white; margin: 0; font-size: 1.9rem; font-weight: 700; }
    .hero p  { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 0.92rem; }
    .stat-bar { display:flex; gap:18px; margin-top:14px; }
    .stat-bar .stat { background:rgba(255,255,255,0.06); padding:10px 14px;
                      border-radius:10px; flex:1; }
    .stat .v { font-size:1.4rem; font-weight:700; color:#10b981; }
    .stat .l { font-size:0.75rem; opacity:0.7; text-transform:uppercase; letter-spacing:0.5px; }

    /* PHASE HEADER */
    .phase-header { font-size:1.25rem; font-weight:700; margin: 22px 0 10px 0;
                    padding-bottom: 8px; border-bottom: 2px solid rgba(16,185,129,0.3); }

    /* MATCH CARD */
    .match { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.08);
             border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
             transition: border-color 0.15s; }
    .match:hover { border-color: rgba(16,185,129,0.4); }
    .match.played { border-left: 3px solid #10b981; }
    .match.pending { border-left: 3px solid rgba(255,255,255,0.12); }

    /* CHIPS */
    .chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.7rem;
            font-weight:600; letter-spacing:0.4px; text-transform:uppercase; }
    .chip-group { background:rgba(16,185,129,0.15); color:#34d399; }
    .chip-knock { background:rgba(244,114,182,0.15); color:#f472b6; }
    .chip-sim   { background:rgba(99,102,241,0.18); color:#a5b4fc; }
    .chip-real  { background:rgba(16,185,129,0.2); color:#10b981; }

    .score { font-size:1.7rem; font-weight:700; line-height:1; }
    .vs    { opacity:0.45; font-weight:600; font-size:1.1rem; }
    .placeholder { opacity:0.5; font-style:italic; }

    /* DAY HEADER */
    .day-header { font-size:1.05rem; font-weight:700; margin: 18px 0 8px 0; color:#10b981; }
    .day-header .meta { opacity:0.5; font-weight:400; font-size:0.85rem; margin-left:6px; }

    /* BRACKET */
    .br-match { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                border-radius:8px; padding:8px 10px; margin-bottom:6px; }
    .br-line { display:flex; justify-content:space-between; align-items:center;
               padding:3px 0; }
    .br-team { display:flex; align-items:center; gap:6px; font-size:0.85rem; }
    .br-score { font-weight:700; font-size:1rem; min-width:18px; text-align:right; }
    .br-winner { color:#10b981; }
    .br-loser  { opacity:0.55; }

    div[data-testid="stHorizontalBlock"] { align-items: center; }
    [data-testid="stCaptionContainer"] { font-size:0.78rem; }
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


# ===== DADOS DA SESSÃO =====
all_matches = matches_with_results()
matches_by_stage = by_stage(all_matches)
n_played = sum(1 for m in all_matches if m.played())
total_matches = len(all_matches)
total_goals = sum((m.home_score or 0) + (m.away_score or 0) for m in all_matches if m.played())
group_progress = sum(1 for m in all_matches if m.stage == "GROUP" and m.played())

# top favorito atual considerando estado da Copa
@st.cache_data(show_spinner=False)
def quick_champion_probs(_n_played: int) -> dict:
    """Heurística rápida: força líquida normalizada dos 16 melhores ainda na chave."""
    return {}


# ===== HERO =====
st.markdown(
    f'<div class="hero">'
    f"<h1>⚽ Simulador Copa do Mundo 2026</h1>"
    f"<p>104 partidas · 48 seleções · 16 cidades-sede em USA, Canadá e México</p>"
    f"<div class=\"stat-bar\">"
    f"<div class=\"stat\"><div class=\"v\">{n_played}/{total_matches}</div>"
    f"<div class=\"l\">Jogos disputados</div></div>"
    f"<div class=\"stat\"><div class=\"v\">{total_goals}</div>"
    f"<div class=\"l\">Gols marcados</div></div>"
    f"<div class=\"stat\"><div class=\"v\">{group_progress}/72</div>"
    f"<div class=\"l\">Fase de grupos</div></div>"
    f"<div class=\"stat\"><div class=\"v\">{(total_goals/n_played) if n_played else 0:.2f}</div>"
    f"<div class=\"l\">Gols/jogo</div></div>"
    f"</div></div>",
    unsafe_allow_html=True,
)


# ===== HELPERS DE RENDER =====
def render_team(iso3: str, side: str = "left", small: bool = False) -> str:
    """HTML de um time (bandeira + nome)."""
    if not iso3 or iso3 not in BY_ISO3:
        return f'<span class="placeholder">{iso3 or "—"}</span>'
    name = name_pt(iso3)
    fl = flag_url(iso3, h=40)
    sz = 24 if small else 28
    if side == "left":
        return (
            f'<div style="display:flex; align-items:center; gap:10px; justify-content:flex-end;">'
            f'<span style="text-align:right; font-weight:600;">{name}</span>'
            f'<img src="{fl}" height="{sz}" alt="{iso3}" '
            f'style="border-radius:2px; box-shadow:0 0 0 1px rgba(0,0,0,0.3);"/></div>'
        )
    return (
        f'<div style="display:flex; align-items:center; gap:10px;">'
        f'<img src="{fl}" height="{sz}" alt="{iso3}" '
        f'style="border-radius:2px; box-shadow:0 0 0 1px rgba(0,0,0,0.3);"/>'
        f'<span style="font-weight:600;">{name}</span></div>'
    )


def stage_chip(m: Match) -> str:
    if m.stage == "GROUP":
        return f'<span class="chip chip-group">Grupo {m.group} · R{m.md}</span>'
    return f'<span class="chip chip-knock">{STAGE_LABEL.get(m.stage, m.stage)}</span>'


def status_chip(m: Match) -> str:
    if not m.played():
        return ""
    if m.status == "actual":
        return '<span class="chip chip-real">✓ Resultado real</span>'
    return '<span class="chip chip-sim">🎲 Simulado</span>'


def render_match_card(m: Match):
    """Card de partida — home esquerda, placar/× central, away direita, ações abaixo."""
    home_id = m.home_team if m.is_resolved() else m.home
    away_id = m.away_team if m.is_resolved() else m.away
    home_resolved = m.home_team in BY_ISO3
    away_resolved = m.away_team in BY_ISO3
    can_simulate = home_resolved and away_resolved

    klass = "match played" if m.played() else "match pending"
    st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)

    # cabeçalho: chips
    chips = f"{stage_chip(m)} {status_chip(m)}"
    if m.venue:
        chips += f' <span style="opacity:0.5; font-size:0.75rem; margin-left:8px;">📍 {m.venue}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([2.6, 0.7, 0.4, 0.7, 2.6])
    with c1:
        st.markdown(render_team(home_id, "left"), unsafe_allow_html=True)
    with c2:
        score = (f'<div class="score" style="text-align:right">{m.home_score}</div>'
                 if m.played() else '<div class="score" style="text-align:right; opacity:0.3">·</div>')
        st.markdown(score, unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="vs" style="text-align:center">×</div>',
                    unsafe_allow_html=True)
    with c4:
        score = (f'<div class="score">{m.away_score}</div>'
                 if m.played() else '<div class="score" style="opacity:0.3">·</div>')
        st.markdown(score, unsafe_allow_html=True)
    with c5:
        st.markdown(render_team(away_id, "right"), unsafe_allow_html=True)

    # ações
    if m.played():
        ac1, ac2 = st.columns([1, 1])
        if ac1.button("✏️ Editar", key=f"edit_{m.id}", use_container_width=True):
            st.session_state[f"editing_{m.id}"] = True
        if ac2.button("🗑️ Limpar", key=f"clear_{m.id}", use_container_width=True):
            delete_result(m.id)
            st.rerun()
    elif can_simulate:
        probs = match_probs(params, home_id, away_id, neutral=True)
        eh, ea = probs["expected_goals"]
        top3 = " · ".join(f"**{r}-{c}** {p:.0%}" for r, c, p in probs["top3_scores"])
        ac1, ac2, ac3 = st.columns([3, 1, 1])
        ac1.caption(
            f"📊 {name_pt(home_id)} {probs['p_home']:.0%} · "
            f"empate {probs['p_draw']:.0%} · "
            f"{name_pt(away_id)} {probs['p_away']:.0%} · "
            f"gols esp **{eh:.1f}:{ea:.1f}**  ·  {top3}"
        )
        if ac2.button("🎲 Simular", key=f"sim_{m.id}", use_container_width=True,
                      type="primary"):
            mat = score_matrix(params, home_id, away_id, neutral=True)
            flat = mat.ravel() / mat.sum()
            k = rng.choice(flat.size, p=flat)
            hs, as_ = divmod(int(k), mat.shape[1])
            # mata-mata: empate vai pros pênaltis (50/50 simples)
            res = {"home_score": hs, "away_score": as_, "status": "simulated"}
            if m.stage != "GROUP" and hs == as_:
                if rng.random() < 0.5:
                    res["home_pen"] = 5
                    res["away_pen"] = 4
                else:
                    res["home_pen"] = 4
                    res["away_pen"] = 5
            upsert_result(m.id, res)
            st.rerun()
        if ac3.button("✏️ Inserir", key=f"input_{m.id}", use_container_width=True):
            st.session_state[f"editing_{m.id}"] = True
    else:
        st.caption("⏳ Aguardando resolução dos confrontos anteriores.")

    if st.session_state.get(f"editing_{m.id}"):
        with st.form(key=f"form_{m.id}"):
            fc1, fc2, fc3 = st.columns([1, 1, 1])
            hs = fc1.number_input(name_pt(home_id), min_value=0, max_value=15,
                                   value=m.home_score if m.home_score is not None else 0,
                                   key=f"hs_{m.id}")
            as_ = fc2.number_input(name_pt(away_id), min_value=0, max_value=15,
                                    value=m.away_score if m.away_score is not None else 0,
                                    key=f"as_{m.id}")
            save = fc3.form_submit_button("💾 Salvar (real)", use_container_width=True)
            if save:
                payload = {"home_score": int(hs), "away_score": int(as_),
                            "status": "actual"}
                if m.stage != "GROUP" and int(hs) == int(as_):
                    payload["home_pen"] = 4
                    payload["away_pen"] = 5
                upsert_result(m.id, payload)
                st.session_state[f"editing_{m.id}"] = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ===== TABS =====
tab_agenda, tab_groups, tab_bracket, tab_match, tab_about = st.tabs(
    ["📅 Agenda", "🏆 Grupos", "🎯 Mata-Mata", "⚔️ Confronto", "ℹ️ Sobre"]
)


# ----- AGENDA -----
with tab_agenda:
    fc1, fc2, fc3 = st.columns([1.5, 1.5, 1])
    show_filter = fc1.radio("Mostrar", ["Todos", "Pendentes", "Jogados"],
                             horizontal=True, label_visibility="collapsed")
    stage_filter = fc2.multiselect(
        "Fases", options=list(STAGE_LABEL.keys()),
        format_func=lambda s: STAGE_LABEL[s], placeholder="Filtrar por fase",
    )
    bulk = fc3.button("🎲 Simular pendentes", use_container_width=True)
    if bulk:
        results = load_results()
        # várias passadas pra cascatear knockouts
        for _ in range(8):
            current = matches_with_results()
            changed = False
            for m in current:
                if m.played() or not m.is_resolved():
                    continue
                mat = score_matrix(params, m.home_team, m.away_team, neutral=True)
                flat = mat.ravel() / mat.sum()
                k = rng.choice(flat.size, p=flat)
                hs, as_ = divmod(int(k), mat.shape[1])
                res = {"home_score": hs, "away_score": as_, "status": "simulated"}
                if m.stage != "GROUP" and hs == as_:
                    if rng.random() < 0.5:
                        res["home_pen"] = 5; res["away_pen"] = 4
                    else:
                        res["home_pen"] = 4; res["away_pen"] = 5
                results[m.id] = res
                changed = True
                save_results(results)
            if not changed:
                break
        st.rerun()

    grouped = by_date(all_matches)
    for date_str, day_matches in grouped.items():
        filtered = [m for m in day_matches
                    if (not stage_filter or m.stage in stage_filter)
                    and (show_filter == "Todos"
                         or (show_filter == "Pendentes" and not m.played())
                         or (show_filter == "Jogados" and m.played()))]
        if not filtered:
            continue
        dt = datetime.fromisoformat(date_str)
        wd = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][dt.weekday()]
        st.markdown(
            f'<div class="day-header">📅 {wd} · {dt.strftime("%d/%m/%Y")}'
            f' <span class="meta">({len(filtered)} jogo{"s" if len(filtered) > 1 else ""})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for m in sorted(filtered, key=lambda x: (x.id)):
            render_match_card(m)


# ----- GRUPOS -----
with tab_groups:
    st.caption("Classificação com base nos jogos da fase de grupos.")
    groups_letters = "ABCDEFGHIJKL"
    for row_start in range(0, len(groups_letters), 2):
        cols = st.columns(2)
        for ci, gi in enumerate(range(row_start, min(row_start + 2, len(groups_letters)))):
            g = groups_letters[gi]
            with cols[ci]:
                st.markdown(f"#### Grupo {g}")
                rows = standings_for_group(g, all_matches)
                df = pd.DataFrame([
                    {"": flag_url(r["team"], h=40), "Seleção": name_pt(r["team"]),
                     "P": r["P"], "V": r["V"], "E": r["E"], "D": r["D"],
                     "GP": r["GP"], "GC": r["GC"], "SG": r["SG"], "Pts": r["Pts"]}
                    for r in rows
                ])
                df.index = range(1, len(df) + 1)
                st.dataframe(
                    df, hide_index=False,
                    column_config={"": st.column_config.ImageColumn("", width="small")},
                    use_container_width=True, height=190,
                )


# ----- MATA-MATA (BRACKET) -----
with tab_bracket:
    st.caption(
        "Os confrontos do mata-mata são preenchidos automaticamente "
        "conforme os jogos anteriores terminam. Use a aba **Agenda** para registrar resultados."
    )
    knock_stages = ["R32", "R16", "QF", "SF", "THIRD", "FINAL"]

    def render_bracket_match(m: Match):
        home = m.home_team if m.is_resolved() else m.home
        away = m.away_team if m.is_resolved() else m.away
        h_in = home in BY_ISO3
        a_in = away in BY_ISO3
        winner = m.winner_code() if m.played() else None
        h_class = "br-winner" if (winner and winner == home) else (
            "br-loser" if winner else ""
        )
        a_class = "br-winner" if (winner and winner == away) else (
            "br-loser" if winner else ""
        )
        h_flag = (f'<img src="{flag_url(home, h=20)}" height="14" '
                  f'style="border-radius:1px;"/>') if h_in else ""
        a_flag = (f'<img src="{flag_url(away, h=20)}" height="14" '
                  f'style="border-radius:1px;"/>') if a_in else ""
        h_name = name_pt(home) if h_in else f'<span class="placeholder">{home}</span>'
        a_name = name_pt(away) if a_in else f'<span class="placeholder">{away}</span>'
        h_score = m.home_score if m.played() else "·"
        a_score = m.away_score if m.played() else "·"
        pen = ""
        if m.played() and m.home_pen is not None:
            pen = f' <span style="font-size:0.7rem; opacity:0.7;">({m.home_pen}-{m.away_pen} pen)</span>'
        return (
            f'<div class="br-match">'
            f'<div class="br-line {h_class}"><div class="br-team">{h_flag} {h_name}</div>'
            f'<div class="br-score">{h_score}</div></div>'
            f'<div class="br-line {a_class}"><div class="br-team">{a_flag} {a_name}</div>'
            f'<div class="br-score">{a_score}{pen}</div></div>'
            f'</div>'
        )

    cols = st.columns(len(knock_stages))
    for col, stage in zip(cols, knock_stages):
        with col:
            st.markdown(
                f'<div style="font-weight:700; text-align:center; margin-bottom:8px; '
                f'color:#10b981; font-size:0.9rem;">{STAGE_LABEL[stage]}</div>',
                unsafe_allow_html=True,
            )
            for m in matches_by_stage.get(stage, []):
                st.markdown(render_bracket_match(m), unsafe_allow_html=True)


# ----- CONFRONTO -----
with tab_match:
    fifa_teams = sorted([t for t in params.teams if t in FIFA_CODES], key=name_pt)
    st.markdown("### Probabilidade de qualquer confronto")
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
            eh, ea = probs["expected_goals"]
            st.markdown(
                f'<div style="text-align:center; padding:16px;">'
                f'<div style="font-size:2.5rem">⚖️</div>'
                f'<div style="opacity:0.7; margin-top:8px">Empate</div>'
                f'<div style="font-size:1.6rem; color:#9ca3af; font-weight:700">'
                f'{probs["p_draw"]:.1%}</div>'
                f'<div style="opacity:0.6; margin-top:12px">Gols esperados:<br/>'
                f'<b>{eh:.2f} : {ea:.2f}</b></div></div>',
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
        # top 5 placares
        st.markdown("##### Placares mais prováveis")
        cols = st.columns(5)
        for col, (r, c, p) in zip(cols, probs["top3_scores"][:5] +
                                    [(0, 0, 0), (0, 0, 0)][:5 - len(probs["top3_scores"])]):
            if p > 0:
                col.metric(f"{r} × {c}", f"{p:.1%}")


with tab_about:
    st.markdown(
        """
        ### Como funciona

        - **Dixon-Coles** com decay temporal e pesos por torneio (Copa do Mundo ×1.6,
          Eurocopa/Copa América ×1.5, eliminatórias ×1.0, amistosos ×0.5).
        - Calibrado em ~3 mil jogos entre seleções **FIFA** desde 2018.
        - Cada jogo tem uma **matriz de probabilidades de placar**.
          O botão *Simular* sorteia um placar dessa distribuição.
        - **Mata-mata:** empate vai aos pênaltis (50/50 simples no MVP).
        - **Bracket auto-resolve** conforme você registra resultados.

        ### Próximos passos
        - Persistência em volume Railway (resultados sobrevivem ao redeploy)
        - Tabela oficial dos 8 melhores 3ºs colocados (regra FIFA)
        - Ajuste por elenco convocado e forma recente dos jogadores nos clubes
        - Modelo de pênaltis com base no histórico do batedor
        """
    )
