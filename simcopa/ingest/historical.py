"""Baixa o dataset histórico de jogos internacionais e popula o SQLite.

Fonte: Mart Jürisoo — 'International football results from 1872 to YYYY'
Mirror público no GitHub mantido pela comunidade.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from simcopa.db import DB_PATH, connect, init_db

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "results.csv"
    if target.exists() and not force:
        return target
    print(f"Baixando {RESULTS_URL} ...")
    r = requests.get(RESULTS_URL, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)
    print(f"OK → {target} ({len(r.content)//1024} KB)")
    return target


COUNTRY_TO_ISO3 = {
    # Mapeamento mínimo — expandir se necessário. As 48 da Copa 2026 vêm de seed_teams.
    "Brazil": "BRA", "Argentina": "ARG", "France": "FRA", "Germany": "GER",
    "Spain": "ESP", "Portugal": "POR", "England": "ENG", "Italy": "ITA",
    "Netherlands": "NED", "Belgium": "BEL", "Croatia": "CRO", "Uruguay": "URU",
    "Colombia": "COL", "Mexico": "MEX", "United States": "USA", "Canada": "CAN",
    "Japan": "JPN", "South Korea": "KOR", "Morocco": "MAR", "Senegal": "SEN",
    "Switzerland": "SUI", "Denmark": "DEN", "Poland": "POL", "Serbia": "SRB",
    "Ecuador": "ECU", "Australia": "AUS", "Iran": "IRN", "Saudi Arabia": "KSA",
    "Tunisia": "TUN", "Ghana": "GHA", "Cameroon": "CMR", "Qatar": "QAT",
    "Wales": "WAL", "Costa Rica": "CRC", "Czech Republic": "CZE",
    "Austria": "AUT", "Norway": "NOR", "Sweden": "SWE", "Türkiye": "TUR",
    "Turkey": "TUR", "Ukraine": "UKR", "Hungary": "HUN", "Greece": "GRE",
    "Egypt": "EGY", "Algeria": "ALG", "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV", "Nigeria": "NGA", "South Africa": "RSA",
    "Mali": "MLI", "Burkina Faso": "BFA", "DR Congo": "COD",
    "New Zealand": "NZL", "Chile": "CHI", "Peru": "PER", "Paraguay": "PAR",
    "Bolivia": "BOL", "Venezuela": "VEN", "Panama": "PAN", "Jamaica": "JAM",
    "Honduras": "HON", "El Salvador": "SLV", "Curaçao": "CUW", "Haiti": "HAI",
    "Iraq": "IRQ", "United Arab Emirates": "UAE", "Jordan": "JOR",
    "Uzbekistan": "UZB", "Oman": "OMA", "Bahrain": "BHR", "Lebanon": "LBN",
    "China PR": "CHN", "China": "CHN", "Vietnam": "VIE", "Thailand": "THA",
    "Indonesia": "IDN", "Malaysia": "MAS", "Philippines": "PHI", "India": "IND",
    "Scotland": "SCO", "Republic of Ireland": "IRL", "Northern Ireland": "NIR",
    "Iceland": "ISL", "Finland": "FIN", "Slovakia": "SVK", "Slovenia": "SVN",
    "Russia": "RUS", "Romania": "ROU", "Bulgaria": "BUL", "Albania": "ALB",
    "North Macedonia": "MKD", "Bosnia and Herzegovina": "BIH", "Montenegro": "MNE",
    "Georgia": "GEO", "Armenia": "ARM", "Azerbaijan": "AZE",
    "Cape Verde": "CPV", "Cabo Verde": "CPV", "Czechia": "CZE",
    "Republic of the Congo": "CGO", "Kosovo": "KVX",
}


def load_into_db(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"home_team": "home", "away_team": "away"})
    df["home"] = df["home"].map(COUNTRY_TO_ISO3).fillna(df["home"])
    df["away"] = df["away"].map(COUNTRY_TO_ISO3).fillna(df["away"])
    df["neutral"] = df["neutral"].astype(int)
    cols = ["date", "home", "away", "home_score", "away_score", "tournament", "neutral"]
    df = df[cols].dropna()

    init_db()
    with connect(DB_PATH) as con:
        con.execute("DELETE FROM historical_matches")
        con.executemany(
            "INSERT INTO historical_matches(date, home, away, home_score, away_score, tournament, neutral) "
            "VALUES (?,?,?,?,?,?,?)",
            df.itertuples(index=False, name=None),
        )
    return len(df)


def main() -> None:
    p = download()
    n = load_into_db(p)
    print(f"Inseridos {n} jogos históricos.")


if __name__ == "__main__":
    main()
