"""SQLite — schema e helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "simcopa.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    code        TEXT PRIMARY KEY,           -- ISO3, ex: BRA
    name        TEXT NOT NULL,
    confederation TEXT,                     -- CONMEBOL, UEFA, ...
    elo         REAL,                       -- rating atual
    group_code  TEXT                        -- A..L na Copa 2026
);

CREATE TABLE IF NOT EXISTS historical_matches (
    date        TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    home_score  INTEGER NOT NULL,
    away_score  INTEGER NOT NULL,
    tournament  TEXT,
    neutral     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_hist_date ON historical_matches(date);
CREATE INDEX IF NOT EXISTS idx_hist_teams ON historical_matches(home, away);

CREATE TABLE IF NOT EXISTS wc_matches (
    id          INTEGER PRIMARY KEY,
    stage       TEXT NOT NULL,              -- GROUP, R32, R16, QF, SF, 3RD, FINAL
    group_code  TEXT,                       -- A..L (só para GROUP)
    matchday    INTEGER,                    -- 1..3 nos grupos
    date        TEXT,
    venue       TEXT,
    home        TEXT,                       -- código do time ou placeholder (W1A, L_QF1...)
    away        TEXT,
    home_score  INTEGER,
    away_score  INTEGER,
    home_pen    INTEGER,                    -- pênaltis (mata-mata)
    away_pen    INTEGER,
    played      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    n_sims      INTEGER,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS model_probs (
    run_id      INTEGER REFERENCES model_runs(id),
    team        TEXT,
    p_group_1   REAL,                       -- 1º do grupo
    p_group_2   REAL,
    p_advance   REAL,                       -- passa da fase de grupos
    p_r16       REAL,
    p_qf        REAL,
    p_sf        REAL,
    p_final     REAL,
    p_champion  REAL,
    PRIMARY KEY (run_id, team)
);
"""


@contextmanager
def connect(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: Path = DB_PATH) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"DB inicializado em {DB_PATH}")
