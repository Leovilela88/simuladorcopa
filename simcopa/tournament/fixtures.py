"""Carrega fixtures e resultados da Copa 2026."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIXTURES_PATH = DATA_DIR / "fixtures_2026.json"
RESULTS_PATH = DATA_DIR / "results.json"


@dataclass
class Match:
    id: int
    date: str
    group: str
    md: int
    home: str
    away: str
    home_score: int | None = None
    away_score: int | None = None
    status: str = "pending"  # pending | simulated | actual

    def played(self) -> bool:
        return self.home_score is not None and self.away_score is not None


def load_fixtures() -> list[Match]:
    raw = json.loads(FIXTURES_PATH.read_text())
    return [Match(**m) for m in raw["matches"]]


def load_results() -> dict[int, dict]:
    if not RESULTS_PATH.exists():
        return {}
    return {int(k): v for k, v in json.loads(RESULTS_PATH.read_text()).items()}


def save_results(results: dict[int, dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2))


def matches_with_results() -> list[Match]:
    """Lê fixtures e aplica resultados salvos por cima."""
    fix = load_fixtures()
    res = load_results()
    for m in fix:
        if m.id in res:
            r = res[m.id]
            m.home_score = r["home_score"]
            m.away_score = r["away_score"]
            m.status = r.get("status", "simulated")
    return fix


def by_date(matches: list[Match]) -> dict[str, list[Match]]:
    out = defaultdict(list)
    for m in matches:
        out[m.date].append(m)
    return dict(sorted(out.items()))


def standings_for_group(group: str, matches: list[Match]) -> list[dict]:
    """Calcula classificação do grupo com base nos jogos já jogados."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"P": 0, "V": 0, "E": 0, "D": 0, "GP": 0, "GC": 0, "SG": 0, "Pts": 0}
    )
    teams = set()
    for m in matches:
        if m.group != group:
            continue
        teams.update([m.home, m.away])
        if not m.played():
            continue
        h, a, hs, as_ = m.home, m.away, m.home_score, m.away_score
        stats[h]["P"] += 1
        stats[a]["P"] += 1
        stats[h]["GP"] += hs
        stats[h]["GC"] += as_
        stats[a]["GP"] += as_
        stats[a]["GC"] += hs
        if hs > as_:
            stats[h]["V"] += 1
            stats[a]["D"] += 1
            stats[h]["Pts"] += 3
        elif hs < as_:
            stats[a]["V"] += 1
            stats[h]["D"] += 1
            stats[a]["Pts"] += 3
        else:
            stats[h]["E"] += 1
            stats[a]["E"] += 1
            stats[h]["Pts"] += 1
            stats[a]["Pts"] += 1
    for t in teams:
        stats[t]["SG"] = stats[t]["GP"] - stats[t]["GC"]
    rows = [{"team": t, **stats[t]} for t in teams]
    rows.sort(key=lambda r: (-r["Pts"], -r["SG"], -r["GP"], r["team"]))
    return rows
