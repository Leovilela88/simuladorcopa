"""CLI: simcopa <comando>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from simcopa.db import DB_PATH, connect, init_db


def cmd_init(_args) -> None:
    init_db()
    print(f"DB pronto em {DB_PATH}")


def cmd_ingest(_args) -> None:
    from simcopa.ingest.historical import main as ingest_main
    ingest_main()


def cmd_fit(args) -> None:
    from simcopa.model.dixon_coles import fit_dixon_coles
    from simcopa.data.countries import FIFA_CODES
    with connect() as con:
        df = pd.read_sql(
            "SELECT date, home, away, home_score, away_score, neutral, tournament "
            "FROM historical_matches WHERE date >= ?",
            con, params=(args.since,),
        )
    n0 = len(df)
    df = df[df["home"].isin(FIFA_CODES) & df["away"].isin(FIFA_CODES)]
    print(f"Filtro FIFA: {n0} → {len(df)} jogos ({n0 - len(df)} removidos)")
    print(f"Ajustando Dixon-Coles em {len(df)} jogos desde {args.since}...")
    params = fit_dixon_coles(df, xi=args.xi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    out.write_bytes(pickle.dumps(params))
    print(f"Modelo salvo em {out}. gamma={params.gamma:.3f} rho={params.rho:.3f}")


def cmd_simulate(args) -> None:
    import pickle
    from simcopa.tournament.simulate import monte_carlo
    params = pickle.loads(Path(args.model).read_bytes())
    groups = json.loads(Path(args.groups).read_text())
    df = monte_carlo(params, groups, n_sims=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(df.head(20).to_string(index=False))
    print(f"\nSalvo em {args.out}")


def main() -> int:
    p = argparse.ArgumentParser(prog="simcopa")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init", help="Inicializa o banco").set_defaults(func=cmd_init)
    sp.add_parser("ingest", help="Baixa histórico").set_defaults(func=cmd_ingest)

    pf = sp.add_parser("fit", help="Ajusta Dixon-Coles")
    pf.add_argument("--since", default="2018-01-01")
    pf.add_argument("--xi", type=float, default=0.0019)
    pf.add_argument("--out", default="data/processed/dc_params.pkl")
    pf.set_defaults(func=cmd_fit)

    ps = sp.add_parser("simulate", help="Roda Monte Carlo do torneio")
    ps.add_argument("--model", default="data/processed/dc_params.pkl")
    ps.add_argument("--groups", default="data/processed/groups.json")
    ps.add_argument("-n", type=int, default=10_000)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--out", default="data/processed/probs.csv")
    ps.set_defaults(func=cmd_simulate)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
