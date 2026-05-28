"""Dixon-Coles (1997) com decay temporal.

Modelo: cada seleção tem força ofensiva alpha_i e defensiva beta_i.
Gols esperados:
    lambda_home = exp(alpha_home + beta_away + gamma)   # gamma = vantagem mando
    lambda_away = exp(alpha_away + beta_home)

Os placares (X,Y) ~ Poisson independente, com correção tau(x,y,lambda,mu,rho)
para placares baixos (0-0, 1-0, 0-1, 1-1). Likelihood é maximizada com decay
exp(-xi * (t_now - t_match_em_dias)) — jogos antigos pesam menos.

Para Copa do Mundo, gamma costuma ser ~0 (campo neutro). Mantemos parâmetro
para permitir ajuste para sede (México/EUA/Canadá).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _log_poisson(k: int, lam: float) -> float:
    # log P(K=k) com k! pré-computado para k pequeno (gols)
    if lam <= 0:
        return -np.inf
    from math import lgamma
    return k * np.log(lam) - lam - lgamma(k + 1)


@dataclass
class DCParams:
    teams: list[str]
    alpha: np.ndarray            # ataque
    beta: np.ndarray             # defesa
    gamma: float                 # mando
    rho: float                   # correção placares baixos

    def idx(self, team: str) -> int:
        return self.teams.index(team)

    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        i, j = self.idx(home), self.idx(away)
        g = 0.0 if neutral else self.gamma
        lam = np.exp(self.alpha[i] + self.beta[j] + g)
        mu = np.exp(self.alpha[j] + self.beta[i])
        return float(lam), float(mu)


TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.6,
    "FIFA World Cup qualification": 1.0,
    "UEFA Euro": 1.5,
    "UEFA Euro qualification": 0.95,
    "Copa América": 1.5,
    "African Cup of Nations": 1.3,
    "AFC Asian Cup": 1.3,
    "CONCACAF Gold Cup": 1.2,
    "UEFA Nations League": 1.1,
    "Friendly": 0.5,
}


def fit_dixon_coles(
    matches: pd.DataFrame,
    xi: float = 0.0019,
    ref_date: pd.Timestamp | None = None,
    max_iter: int = 200,
    use_tournament_weights: bool = True,
) -> DCParams:
    """Ajusta Dixon-Coles. Colunas necessárias:
    date, home, away, home_score, away_score, neutral (0/1).
    Coluna opcional `tournament` ativa pesos por tipo de jogo."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    ref = ref_date or df["date"].max()
    w_time = np.exp(-xi * (ref - df["date"]).dt.days.clip(lower=0))
    if use_tournament_weights and "tournament" in df.columns:
        w_tourn = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(0.8).to_numpy()
    else:
        w_tourn = np.ones(len(df))
    df["w"] = w_time * w_tourn

    teams = sorted(set(df["home"]).union(df["away"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Vetor de parâmetros: [alpha_1..alpha_n, beta_1..beta_n, gamma, rho]
    # Restrição: sum(alpha) = 0 (identificabilidade) — aplicada via reparametrização.
    def unpack(x):
        a_free = x[: n - 1]
        b_free = x[n - 1 : 2 * (n - 1)]
        a = np.concatenate([a_free, [-a_free.sum()]])
        b = np.concatenate([b_free, [-b_free.sum()]])
        gamma = x[2 * (n - 1)]
        rho = x[2 * (n - 1) + 1]
        return a, b, gamma, rho

    home_i = df["home"].map(idx).to_numpy()
    away_i = df["away"].map(idx).to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df.get("neutral", pd.Series(0, index=df.index)).to_numpy()
    w = df["w"].to_numpy()

    def neg_loglik(x):
        a, b, gamma, rho = unpack(x)
        g = np.where(neutral == 1, 0.0, gamma)
        lam = np.exp(a[home_i] + b[away_i] + g)
        mu = np.exp(a[away_i] + b[home_i])
        # log Poisson
        from scipy.special import gammaln
        ll = (
            hs * np.log(lam) - lam - gammaln(hs + 1)
            + as_ * np.log(mu) - mu - gammaln(as_ + 1)
        )
        # correção Dixon-Coles para placares baixos
        tau = np.ones_like(ll)
        mask00 = (hs == 0) & (as_ == 0)
        mask01 = (hs == 0) & (as_ == 1)
        mask10 = (hs == 1) & (as_ == 0)
        mask11 = (hs == 1) & (as_ == 1)
        tau[mask00] = 1 - lam[mask00] * mu[mask00] * rho
        tau[mask01] = 1 + lam[mask01] * rho
        tau[mask10] = 1 + mu[mask10] * rho
        tau[mask11] = 1 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = ll + np.log(tau)
        return -float(np.sum(w * ll))

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n - 1), [0.25], [-0.1]])
    res = minimize(
        neg_loglik, x0, method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False},
    )
    a, b, gamma, rho = unpack(res.x)
    return DCParams(teams=teams, alpha=a, beta=b, gamma=gamma, rho=rho)


def score_matrix(params: DCParams, home: str, away: str, neutral: bool = True,
                 max_goals: int = 10) -> np.ndarray:
    """Matriz (max_goals+1)×(max_goals+1) com P(X=i, Y=j)."""
    lam, mu = params.lambdas(home, away, neutral=neutral)
    from scipy.stats import poisson
    px = poisson.pmf(np.arange(max_goals + 1), lam)
    py = poisson.pmf(np.arange(max_goals + 1), mu)
    mat = np.outer(px, py)
    # aplica correção apenas nos 4 cantos
    mat[0, 0] *= 1 - lam * mu * params.rho
    mat[0, 1] *= 1 + lam * params.rho
    mat[1, 0] *= 1 + mu * params.rho
    mat[1, 1] *= 1 - params.rho
    mat = np.clip(mat, 0, None)
    mat /= mat.sum()
    return mat


def match_probs(params: DCParams, home: str, away: str, neutral: bool = True) -> dict:
    """Probabilidades 1X2, gols esperados e top 3 placares."""
    mat = score_matrix(params, home, away, neutral=neutral)
    p_home = np.tril(mat, -1).sum()
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()
    i, j = np.unravel_index(mat.argmax(), mat.shape)
    # gols esperados = soma de k*p(k)
    n = mat.shape[0]
    ks = np.arange(n)
    e_home = float((ks[:, None] * mat).sum())
    e_away = float((mat * ks[None, :]).sum())
    # top 3 placares mais prováveis
    flat_idx = np.argsort(mat.ravel())[::-1][:3]
    top3 = []
    for k in flat_idx:
        r, c = divmod(int(k), mat.shape[1])
        top3.append((r, c, float(mat[r, c])))
    return {
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "mode_score": (int(i), int(j)),
        "expected_goals": (e_home, e_away),
        "top3_scores": top3,
    }


def simulate_match(params: DCParams, home: str, away: str, rng: np.random.Generator,
                   neutral: bool = True, knockout: bool = False) -> tuple[int, int, str | None]:
    """Sorteia placar. Em mata-mata, sorteia vencedor por pênaltis 50/50 em caso de empate.
    Retorna (gols_home, gols_away, winner_code_or_None)."""
    mat = score_matrix(params, home, away, neutral=neutral)
    flat = mat.ravel()
    flat = flat / flat.sum()
    k = rng.choice(flat.size, p=flat)
    i, j = divmod(k, mat.shape[1])
    if not knockout:
        return i, j, None
    if i > j:
        return i, j, home
    if j > i:
        return i, j, away
    # empate → pênaltis (50/50 para simplificar; dá pra refinar depois)
    winner = home if rng.random() < 0.5 else away
    return i, j, winner
