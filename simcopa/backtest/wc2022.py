"""Backtest do Dixon-Coles na Copa do Mundo 2022 (Qatar).

Protocolo:
 1. Treina DC usando APENAS jogos com data < 2022-11-20 (início da Copa).
 2. Monta os 8 grupos reais (A..H, 32 times) — formato antigo.
 3. Roda Monte Carlo (N simulações).
 4. Compara probabilidades estimadas vs resultados reais:
    - Brier score multiclass (campeão entre os 32)
    - Log-loss (campeão)
    - Acurácia top-1 (predição modal == campeão?)
    - Brier do "passou de grupo" (16 acertos possíveis)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simcopa.db import DB_PATH, connect
from simcopa.model.dixon_coles import fit_dixon_coles
from simcopa.tournament.simulate import monte_carlo as monte_carlo_2026

WC2022_START = "2022-11-20"

# Grupos reais Copa 2022 (32 times, 8 grupos)
GROUPS_2022: dict[str, list[str]] = {
    "A": ["QAT", "ECU", "SEN", "NED"],
    "B": ["ENG", "IRN", "USA", "WAL"],
    "C": ["ARG", "KSA", "MEX", "POL"],
    "D": ["FRA", "AUS", "DEN", "TUN"],
    "E": ["ESP", "CRC", "GER", "JPN"],
    "F": ["BEL", "CAN", "MAR", "CRO"],
    "G": ["BRA", "SRB", "SUI", "CMR"],
    "H": ["POR", "GHA", "URU", "KOR"],
}

# Resultados oficiais (quem avançou em cada fase)
WC2022_RESULTS = {
    "champion": "ARG",
    "runner_up": "FRA",
    "third": "CRO",
    "fourth": "MAR",
    "semifinalists": {"ARG", "FRA", "CRO", "MAR"},
    "quarterfinalists": {"ARG", "FRA", "CRO", "MAR", "NED", "ENG", "BRA", "POR"},
    "advanced": {  # 16 que passaram da fase de grupos
        "NED", "SEN", "ENG", "USA", "ARG", "POL", "FRA", "AUS",
        "JPN", "ESP", "MAR", "CRO", "BRA", "SUI", "POR", "KOR",
    },
}


def _simulate_groups_2022(params, groups, rng):
    """Versão 2022: 8 grupos, top 2 avançam direto pros oitavas."""
    from collections import defaultdict
    from simcopa.model.dixon_coles import simulate_match

    progress = defaultdict(lambda: {"advance": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0,
                                     "group_1": 0, "group_2": 0})
    advancers = []
    bracket = {}  # mapping group->[1st, 2nd]
    for g, teams in groups.items():
        played = []
        for md, (i, j) in [(1, (1, 2)), (1, (3, 4)),
                            (2, (1, 3)), (2, (4, 2)),
                            (3, (4, 1)), (3, (2, 3))]:
            h, a = teams[i - 1], teams[j - 1]
            hs, as_, _ = simulate_match(params, h, a, rng, neutral=True, knockout=False)
            played.append((h, a, hs, as_))
        # standings
        stats = defaultdict(lambda: [0, 0, 0])
        for h, a, hs, as_ in played:
            if hs > as_: stats[h][0] += 3
            elif hs < as_: stats[a][0] += 3
            else: stats[h][0] += 1; stats[a][0] += 1
            stats[h][1] += hs - as_; stats[a][1] += as_ - hs
            stats[h][2] += hs; stats[a][2] += as_
        rows = sorted(stats.items(), key=lambda r: (-r[1][0], -r[1][1], -r[1][2]))
        first, second = rows[0][0], rows[1][0]
        bracket[g] = [first, second]
        progress[first]["group_1"] = 1
        progress[first]["advance"] = 1
        progress[second]["group_2"] = 1
        progress[second]["advance"] = 1
        advancers.extend([first, second])
    return progress, bracket


def _simulate_tournament_2022(params, groups, rng):
    """Simula 1 Copa 2022 completa com chaveamento oficial."""
    from simcopa.model.dixon_coles import simulate_match

    progress, bracket = _simulate_groups_2022(params, groups, rng)

    # Chaveamento oficial 2022 (oitavas):
    # 1A vs 2B, 1C vs 2D, 1E vs 2F, 1G vs 2H
    # 1B vs 2A, 1D vs 2C, 1F vs 2E, 1H vs 2G
    pairs_r16 = [
        (bracket["A"][0], bracket["B"][1]),
        (bracket["C"][0], bracket["D"][1]),
        (bracket["E"][0], bracket["F"][1]),
        (bracket["G"][0], bracket["H"][1]),
        (bracket["B"][0], bracket["A"][1]),
        (bracket["D"][0], bracket["C"][1]),
        (bracket["F"][0], bracket["E"][1]),
        (bracket["H"][0], bracket["G"][1]),
    ]
    qf = []
    for h, a in pairs_r16:
        _, _, w = simulate_match(params, h, a, rng, neutral=True, knockout=True)
        qf.append(w)
        progress[w]["qf"] = 1

    pairs_qf = [(qf[0], qf[1]), (qf[2], qf[3]), (qf[4], qf[5]), (qf[6], qf[7])]
    sf = []
    for h, a in pairs_qf:
        _, _, w = simulate_match(params, h, a, rng, neutral=True, knockout=True)
        sf.append(w)
        progress[w]["sf"] = 1

    pairs_sf = [(sf[0], sf[1]), (sf[2], sf[3])]
    finalists = []
    for h, a in pairs_sf:
        _, _, w = simulate_match(params, h, a, rng, neutral=True, knockout=True)
        finalists.append(w)
        progress[w]["final"] = 1

    _, _, champion = simulate_match(params, finalists[0], finalists[1], rng,
                                     neutral=True, knockout=True)
    progress[champion]["champion"] = 1
    return progress


def run_backtest(n_sims: int = 10_000, since: str = "2014-01-01",
                 xi: float = 0.0019, seed: int = 42) -> dict:
    print(f"[backtest 2022] Carregando jogos {since} → {WC2022_START}...")
    with connect() as con:
        df = pd.read_sql(
            "SELECT date, home, away, home_score, away_score, neutral "
            "FROM historical_matches WHERE date >= ? AND date < ?",
            con, params=(since, WC2022_START),
        )
    print(f"  {len(df)} jogos para treino")

    # garantir que todos os 32 times de 2022 estão no dataset
    all_teams_22 = {t for ts in GROUPS_2022.values() for t in ts}
    have = set(df["home"]).union(df["away"])
    missing = all_teams_22 - have
    if missing:
        print(f"  ⚠️  Times sem dados de treino: {missing}")

    print(f"[backtest 2022] Ajustando Dixon-Coles (xi={xi})...")
    params = fit_dixon_coles(df, xi=xi)
    print(f"  gamma={params.gamma:.3f} rho={params.rho:.3f}")

    print(f"[backtest 2022] Rodando {n_sims:,} simulações...")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))
    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        prog = _simulate_tournament_2022(params, GROUPS_2022, rng)
        for t, d in prog.items():
            for k, v in d.items():
                agg[t][k] += v

    probs = []
    for t in all_teams_22:
        d = agg.get(t, {})
        probs.append({
            "team": t,
            "p_advance": d.get("advance", 0) / n_sims,
            "p_qf": d.get("qf", 0) / n_sims,
            "p_sf": d.get("sf", 0) / n_sims,
            "p_final": d.get("final", 0) / n_sims,
            "p_champion": d.get("champion", 0) / n_sims,
        })
    probs_df = pd.DataFrame(probs).sort_values("p_champion", ascending=False).reset_index(drop=True)

    # métricas
    # 1) Brier multiclass — campeão
    p_champ = probs_df.set_index("team")["p_champion"]
    y_champ = (probs_df["team"] == WC2022_RESULTS["champion"]).astype(int).values
    brier_champion = float(np.mean((probs_df["p_champion"].values - y_champ) ** 2))
    # 2) log-loss campeão (clip para evitar -inf)
    p_arg = max(p_champ.get(WC2022_RESULTS["champion"], 0), 1e-6)
    logloss_champion = -np.log(p_arg)
    # 3) top-1 == campeão?
    top1_correct = bool(probs_df.iloc[0]["team"] == WC2022_RESULTS["champion"])
    # 4) Brier "avançou" (16 acertos possíveis)
    y_adv = probs_df["team"].isin(WC2022_RESULTS["advanced"]).astype(int).values
    brier_advance = float(np.mean((probs_df["p_advance"].values - y_adv) ** 2))
    # 5) Brier semifinal
    y_sf = probs_df["team"].isin(WC2022_RESULTS["semifinalists"]).astype(int).values
    brier_sf = float(np.mean((probs_df["p_sf"].values - y_sf) ** 2))

    metrics = {
        "n_sims": n_sims,
        "brier_champion": brier_champion,
        "logloss_champion": float(logloss_champion),
        "top1_predicted": probs_df.iloc[0]["team"],
        "top1_correct": top1_correct,
        "p_argentina_champion": float(p_arg),
        "brier_advance": brier_advance,
        "brier_semifinal": brier_sf,
    }
    return {"probs": probs_df, "metrics": metrics, "params": params}


if __name__ == "__main__":
    import sys, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    res = run_backtest(n_sims=n)
    print("\n=== TOP 10 (probabilidade de campeão) ===")
    print(res["probs"].head(10).to_string(index=False,
        formatters={c: "{:.1%}".format for c in res["probs"].columns if c.startswith("p_")}))
    print("\n=== MÉTRICAS ===")
    print(json.dumps(res["metrics"], indent=2, default=float))
    out = Path("data/processed/backtest_2022.csv")
    res["probs"].to_csv(out, index=False)
    print(f"\nSalvo em {out}")
