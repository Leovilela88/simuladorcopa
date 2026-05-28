"""Carrega fixtures (grupos + mata-mata) e resultados da Copa 2026.

Os slots do mata-mata são placeholders (ex.: '1A', '2B', '3CDEF', 'W74', 'L101')
que vão sendo resolvidos para códigos de seleção conforme os jogos terminam.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIXTURES_PATH = DATA_DIR / "fixtures_2026.json"
RESULTS_PATH = DATA_DIR / "results.json"

# Atribuição dos 8 melhores 3ºs colocados → slots do R32 (regra FIFA 2026).
# Ordem dos grupos dos 3ºs que avançam (de A-L) → mapeamento para confrontos.
# Tabela oficial: dependendo de QUAIS grupos contribuem com 3º, o emparelhamento muda.
# Fonte: FIFA 2026 third-place qualification rules.
THIRD_PLACE_ALLOCATION = {
    # frozenset(of_groups_with_third_qualified): {match_id: group_letter}
    # 12 grupos, 8 dos 12 3ºs avançam → C(12,8) = 495 combinações.
    # Por simplicidade no MVP, usamos uma heurística (primeiros 8 em ordem alfabética).
    # Refinaremos depois com a tabela exata da FIFA.
}


@dataclass
class Match:
    id: int
    date: str
    home: str
    away: str
    stage: str = "GROUP"           # GROUP, R32, R16, QF, SF, THIRD, FINAL
    group: str | None = None       # A..L (só para GROUP)
    md: int | None = None          # 1..3 nos grupos
    venue: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    home_pen: int | None = None
    away_pen: int | None = None
    status: str = "pending"        # pending | simulated | actual

    # campos resolvidos em runtime (não persistidos):
    home_resolved: str = field(default="", repr=False)
    away_resolved: str = field(default="", repr=False)

    def played(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def home_team(self) -> str:
        return self.home_resolved or self.home

    @property
    def away_team(self) -> str:
        return self.away_resolved or self.away

    def is_resolved(self) -> bool:
        return bool(self.home_resolved) and bool(self.away_resolved)

    def winner_code(self) -> str | None:
        """Vencedor de mata-mata (considera pênaltis salvos)."""
        if not self.played():
            return None
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        if self.home_pen is not None and self.away_pen is not None:
            return self.home_team if self.home_pen > self.away_pen else self.away_team
        return None


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


def _resolve_placeholders(matches: list[Match]) -> None:
    """Resolve slots como '1A', '2B', '3ABCDF', 'W74', 'L101' usando o estado
    atual dos jogos. Modifica matches in-place setando home_resolved/away_resolved."""
    # Mapeia grupos → standings ordenadas
    by_group_standings = {}
    for g in "ABCDEFGHIJKL":
        rows = standings_for_group(g, matches)
        by_group_standings[g] = rows

    # Lista de 3ºs colocados (grupo, rows[2])
    thirds = []
    for g, rows in by_group_standings.items():
        if len(rows) >= 3 and rows[2]["P"] >= 3:   # já jogou as 3 rodadas
            thirds.append((g, rows[2]))
    # Top 8 entre os 3ºs (pts, sg, gp)
    thirds_sorted = sorted(thirds, key=lambda gr: (-gr[1]["Pts"], -gr[1]["SG"], -gr[1]["GP"]))
    top8_third_groups = {g for g, _ in thirds_sorted[:8]}
    third_by_group = {g: rows for g, rows in thirds}

    # Mapeia ids → match para olhar resultados de mata-mata
    by_id = {m.id: m for m in matches}

    def resolve(slot: str) -> str:
        # 1A, 2B (1º/2º do grupo)
        m = re.match(r"^([12])([A-L])$", slot)
        if m:
            rank = int(m.group(1)) - 1
            g = m.group(2)
            rows = by_group_standings.get(g, [])
            if len(rows) > rank and rows[0]["P"] >= 3:
                return rows[rank]["team"]
            return ""
        # 3XYZ (melhor 3º dentre esses grupos)
        m = re.match(r"^3([A-L]+)$", slot)
        if m:
            candidate_groups = list(m.group(1))
            # entre os grupos candidatos que tiveram 3º classificado
            present = [g for g in candidate_groups if g in top8_third_groups
                       and g in third_by_group]
            if not present:
                return ""
            # melhor 3º entre eles (mesma ordenação)
            best = sorted(present,
                          key=lambda g: (-third_by_group[g][2]["Pts"],
                                          -third_by_group[g][2]["SG"],
                                          -third_by_group[g][2]["GP"]))[0]
            return third_by_group[best][2]["team"]
        # W74, L101 (vencedor/perdedor de jogo N)
        m = re.match(r"^([WL])(\d+)$", slot)
        if m:
            kind = m.group(1)
            mid = int(m.group(2))
            mt = by_id.get(mid)
            if not mt or not mt.played():
                return ""
            w = mt.winner_code()
            if not w:
                return ""
            if kind == "W":
                return w
            # L: o outro
            return mt.away_team if w == mt.home_team else mt.home_team
        return ""

    # Múltiplas passadas pra cascatear (W77 depende de R32 que depende de 1A...)
    for _ in range(4):
        for m in matches:
            if not m.home_resolved:
                m.home_resolved = resolve(m.home)
            if not m.away_resolved:
                m.away_resolved = resolve(m.away)


def matches_with_results() -> list[Match]:
    fix = load_fixtures()
    res = load_results()
    for m in fix:
        if m.id in res:
            r = res[m.id]
            m.home_score = r["home_score"]
            m.away_score = r["away_score"]
            m.home_pen = r.get("home_pen")
            m.away_pen = r.get("away_pen")
            m.status = r.get("status", "simulated")
    _resolve_placeholders(fix)
    return fix


def by_date(matches: list[Match]) -> dict[str, list[Match]]:
    out = defaultdict(list)
    for m in matches:
        out[m.date].append(m)
    return dict(sorted(out.items()))


def by_stage(matches: list[Match]) -> dict[str, list[Match]]:
    out = defaultdict(list)
    order = {"GROUP": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "THIRD": 5, "FINAL": 6}
    for m in matches:
        out[m.stage].append(m)
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 99)))


def standings_for_group(group: str, matches: list[Match]) -> list[dict]:
    stats: dict[str, dict] = defaultdict(
        lambda: {"P": 0, "V": 0, "E": 0, "D": 0, "GP": 0, "GC": 0, "SG": 0, "Pts": 0}
    )
    teams = set()
    for m in matches:
        if m.group != group or m.stage != "GROUP":
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


STAGE_LABEL = {
    "GROUP": "Fase de Grupos",
    "R32": "32-avos de Final",
    "R16": "Oitavas de Final",
    "QF": "Quartas de Final",
    "SF": "Semifinal",
    "THIRD": "Disputa de 3º lugar",
    "FINAL": "Final",
}
