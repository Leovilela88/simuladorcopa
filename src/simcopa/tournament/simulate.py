"""Simulação Monte Carlo da Copa 2026.

Dado:
  - params: Dixon-Coles ajustado
  - groups: dict {grupo: [4 códigos]}
  - results: dict de resultados já conhecidos (chave = id do jogo)

Simula N torneios e devolve, por seleção:
  P(passar de grupo), P(R16), P(QF), P(SF), P(final), P(campeão).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from simcopa.model.dixon_coles import DCParams, simulate_match
from simcopa.tournament.structure import GROUPS, group_stage_fixtures


@dataclass
class TournamentState:
    groups: dict[str, list[str]]                    # 4 times por grupo
    fixed_results: dict[tuple[str, int, int], tuple[int, int]] = None  # (grp, md, slot_idx_home) -> (h,a)

    def __post_init__(self):
        if self.fixed_results is None:
            self.fixed_results = {}


def _group_standings(played: list[tuple[str, str, int, int]]) -> list[tuple[str, int, int, int]]:
    """Retorna lista ordenada (team, pts, gd, gf) — critérios FIFA: pts, sg, gp."""
    stats = defaultdict(lambda: [0, 0, 0])  # pts, gd, gf
    for h, a, hs, as_ in played:
        if hs > as_:
            stats[h][0] += 3
        elif hs < as_:
            stats[a][0] += 3
        else:
            stats[h][0] += 1
            stats[a][0] += 1
        stats[h][1] += hs - as_
        stats[a][1] += as_ - hs
        stats[h][2] += hs
        stats[a][2] += as_
    rows = [(t, *vals) for t, vals in stats.items()]
    rows.sort(key=lambda r: (-r[1], -r[2], -r[3], r[0]))
    return rows


def _simulate_groups(
    params: DCParams,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
) -> dict[str, list[tuple[str, int, int, int]]]:
    standings = {}
    for g, teams in groups.items():
        played = []
        for md, (i, j) in [(1, (1, 2)), (1, (3, 4)),
                            (2, (1, 3)), (2, (4, 2)),
                            (3, (4, 1)), (3, (2, 3))]:
            h, a = teams[i - 1], teams[j - 1]
            hs, as_, _ = simulate_match(params, h, a, rng, neutral=True, knockout=False)
            played.append((h, a, hs, as_))
        standings[g] = _group_standings(played)
    return standings


def _pick_best_thirds(standings: dict[str, list]) -> list[str]:
    """Escolhe os 8 melhores 3ºs colocados entre os 12 grupos."""
    thirds = []
    for g, rows in standings.items():
        if len(rows) >= 3:
            t, pts, gd, gf = rows[2]
            thirds.append((t, pts, gd, gf, g))
    thirds.sort(key=lambda r: (-r[1], -r[2], -r[3], r[0]))
    return [t[0] for t in thirds[:8]]


def _knockout_round(params: DCParams, pairs: list[tuple[str, str]],
                    rng: np.random.Generator) -> list[str]:
    winners = []
    for h, a in pairs:
        _, _, w = simulate_match(params, h, a, rng, neutral=True, knockout=True)
        winners.append(w)
    return winners


def simulate_tournament(
    params: DCParams,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Roda 1 simulação completa. Retorna dict de progressão por time."""
    progress: dict[str, dict] = {}
    for teams in groups.values():
        for t in teams:
            progress[t] = {"advance": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0,
                            "group_1": 0, "group_2": 0}

    standings = _simulate_groups(params, groups, rng)
    advancers: list[str] = []
    for g, rows in standings.items():
        if len(rows) >= 1:
            progress[rows[0][0]]["group_1"] = 1
            progress[rows[0][0]]["advance"] = 1
            advancers.append(rows[0][0])
        if len(rows) >= 2:
            progress[rows[1][0]]["group_2"] = 1
            progress[rows[1][0]]["advance"] = 1
            advancers.append(rows[1][0])
    best_thirds = _pick_best_thirds(standings)
    for t in best_thirds:
        progress[t]["advance"] = 1
    advancers.extend(best_thirds)

    # 32 → 16: emparelhamento simples (1º de A vs melhor 3º, etc.) — aproximação.
    # Refinamento do bracket oficial fica para próxima iteração.
    rng.shuffle(advancers)
    pairs_r32 = [(advancers[i], advancers[i + 1]) for i in range(0, 32, 2)]
    r16 = _knockout_round(params, pairs_r32, rng)
    for t in r16:
        progress[t]["r16"] = 1

    pairs_r16 = [(r16[i], r16[i + 1]) for i in range(0, 16, 2)]
    qf = _knockout_round(params, pairs_r16, rng)
    for t in qf:
        progress[t]["qf"] = 1

    pairs_qf = [(qf[i], qf[i + 1]) for i in range(0, 8, 2)]
    sf = _knockout_round(params, pairs_qf, rng)
    for t in sf:
        progress[t]["sf"] = 1

    pairs_sf = [(sf[0], sf[1]), (sf[2], sf[3])]
    finalists = _knockout_round(params, pairs_sf, rng)
    for t in finalists:
        progress[t]["final"] = 1

    champion = _knockout_round(params, [(finalists[0], finalists[1])], rng)[0]
    progress[champion]["champion"] = 1
    return progress


def monte_carlo(
    params: DCParams,
    groups: dict[str, list[str]],
    n_sims: int = 10_000,
    seed: int = 42,
) -> "pd.DataFrame":
    import pandas as pd
    rng = np.random.default_rng(seed)
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        prog = simulate_tournament(params, groups, rng)
        for t, d in prog.items():
            for k, v in d.items():
                agg[t][k] += v
    rows = []
    for t, d in agg.items():
        rows.append({"team": t, **{f"p_{k}": v / n_sims for k, v in d.items()}})
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
