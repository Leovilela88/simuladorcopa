"""Persistência dos resultados — Postgres em prod, JSON em dev.

Se `DATABASE_URL` (ou `POSTGRES_URL`) estiver no ambiente, usa Postgres.
Caso contrário, escreve em `data/processed/results.json` (que some no
redeploy do Railway, por isso a preferência pelo banco).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
JSON_PATH = DATA_DIR / "results.json"


def _env_url() -> str | None:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


class JsonStore:
    def __init__(self, path: Path = JSON_PATH):
        self.path = path

    def load(self) -> dict[int, dict]:
        if not self.path.exists():
            return {}
        return {int(k): v for k, v in json.loads(self.path.read_text()).items()}

    def save_all(self, results: dict[int, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({str(k): v for k, v in results.items()}, indent=2)
        )

    def upsert(self, match_id: int, data: dict) -> None:
        all_ = self.load()
        all_[match_id] = data
        self.save_all(all_)

    def delete(self, match_id: int) -> None:
        all_ = self.load()
        all_.pop(match_id, None)
        self.save_all(all_)


class PgStore:
    """Tabela: match_results(match_id, home_score, away_score, home_pen, away_pen, status, updated_at)"""

    def __init__(self, url: str):
        import psycopg2  # lazy import
        self._psycopg2 = psycopg2
        # Railway às vezes prefixa com "postgres://" — psycopg2 v2 aceita ambos.
        self.url = url
        self._ensure_schema()

    def _connect(self):
        return self._psycopg2.connect(self.url, connect_timeout=8)

    def _ensure_schema(self) -> None:
        with self._connect() as c, c.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id   INT PRIMARY KEY,
                    home_score INT NOT NULL,
                    away_score INT NOT NULL,
                    home_pen   INT,
                    away_pen   INT,
                    status     TEXT NOT NULL DEFAULT 'simulated',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def load(self) -> dict[int, dict]:
        with self._connect() as c, c.cursor() as cur:
            cur.execute(
                "SELECT match_id, home_score, away_score, home_pen, away_pen, status "
                "FROM match_results"
            )
            rows = cur.fetchall()
        return {
            r[0]: {
                "home_score": r[1], "away_score": r[2],
                "home_pen": r[3], "away_pen": r[4],
                "status": r[5],
            }
            for r in rows
        }

    def upsert(self, match_id: int, data: dict) -> None:
        with self._connect() as c, c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO match_results
                    (match_id, home_score, away_score, home_pen, away_pen, status, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (match_id) DO UPDATE SET
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    home_pen   = EXCLUDED.home_pen,
                    away_pen   = EXCLUDED.away_pen,
                    status     = EXCLUDED.status,
                    updated_at = NOW()
                """,
                (
                    match_id,
                    data["home_score"], data["away_score"],
                    data.get("home_pen"), data.get("away_pen"),
                    data.get("status", "simulated"),
                ),
            )

    def save_all(self, results: dict[int, dict]) -> None:
        # usado quando bulk-simulate: rewrites everything
        with self._connect() as c, c.cursor() as cur:
            cur.execute("DELETE FROM match_results")
            for mid, r in results.items():
                cur.execute(
                    "INSERT INTO match_results "
                    "(match_id, home_score, away_score, home_pen, away_pen, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        mid, r["home_score"], r["away_score"],
                        r.get("home_pen"), r.get("away_pen"),
                        r.get("status", "simulated"),
                    ),
                )

    def delete(self, match_id: int) -> None:
        with self._connect() as c, c.cursor() as cur:
            cur.execute("DELETE FROM match_results WHERE match_id = %s", (match_id,))


def get_store():
    url = _env_url()
    if url:
        try:
            return PgStore(url)
        except Exception as e:
            # falha de conexão → cai pro JSON (UX: app não quebra)
            print(f"[persistence] Postgres indisponível, usando JSON: {e}")
    return JsonStore()
