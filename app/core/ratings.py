"""
ratings.py
==========
Stima dei coefficienti di ATTACCO e DIFESA di ogni nazionale dai risultati
storici, con regolarizzazione (shrinkage) verso un PRIOR di forza Elo e
parametro rho di Dixon-Coles.

==================== FONTE DATI (UNICA) ====================
Risultati storici: Mart Jurisoo (martj42), licenza CC0-1.0 (pubblico dominio).
https://github.com/martj42/international_results  (vedi DATA_SOURCES.md)
DISCLAIMER: progetto a fini EDUCATIVI/DIMOSTRATIVI, NON commerciale.
===========================================================

Metodo
------
Modello "double Poisson" (Maher 1982): forze attacco/difesa stimate per massima
verosimiglianza via iterative scaling (solo NumPy), con time-decay. Si stima poi
il parametro rho di Dixon-Coles (1997).

SHRINKAGE verso PRIOR ELO
-------------------------
La regolarizzazione aggiunge a ogni squadra alcune "partite fantasma". Invece di
tirarla verso una squadra media piatta (1.0), la tiriamo verso il suo livello di
forza ELO (calcolato dai risultati CC0, vedi elo.py): le squadre forti per Elo
restano forti anche con pochi dati recenti, le altre si appoggiano all'Elo. Se
nessun prior viene fornito, lo shrinkage e' verso 1.0 (comportamento classico).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.core.dixon_coles import estimate_rho
from app.core.models import Team
from app.data.loader import Match

DEFAULT_HALF_LIFE_DAYS: float = 365 * 4    # ~4 anni: ottimo dal backtest RPS
DEFAULT_SHRINKAGE: float = 0.0  # backtest RPS: lo shrinkage peggiora -> default 0 (modello puro)


@dataclass(frozen=True)
class RatingsResult:
    """Esito della stima: squadre con rating + parametri globali del modello."""

    teams: list[Team]
    base_goals: float
    home_advantage: float
    rho: float
    n_matches: int


def _time_weights(dates: np.ndarray, half_life_days: float) -> np.ndarray:
    """Peso esponenziale: w = 0.5 ** (eta_giorni / emivita)."""
    most_recent = dates.max()
    age_days = (most_recent - dates).astype("timedelta64[D]").astype(float)
    decay = math.log(2.0) / half_life_days
    return np.exp(-decay * age_days)


def estimate_ratings(
    matches: list[Match],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    shrinkage: float = DEFAULT_SHRINKAGE,
    strength_prior: dict[str, float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> RatingsResult:
    """
    Stima attacco/difesa + vantaggio campo + rho, con shrinkage verso un prior.

    Args:
        matches: storico partite.
        half_life_days: emivita del time-decay.
        shrinkage: n. di "partite fantasma" verso il prior (0 = nessuna).
        strength_prior: {squadra: moltiplicatore di forza} (es. da Elo). Una
            forza s>1 implica prior attacco sqrt(s) e prior difesa 1/sqrt(s)
            (squadra forte = segna di piu' e subisce di meno, a parita' di gol
            totali). Se None, il prior e' 1.0 per tutti (shrinkage verso media).
    """
    if not matches:
        raise ValueError("Nessuna partita fornita per la stima dei rating.")

    names = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
    idx = {name: i for i, name in enumerate(names)}
    n_teams = len(names)

    home = np.array([idx[m.home_team] for m in matches])
    away = np.array([idx[m.away_team] for m in matches])
    hs = np.array([m.home_score for m in matches], dtype=float)
    as_ = np.array([m.away_score for m in matches], dtype=float)
    neutral = np.array([m.neutral for m in matches], dtype=bool)
    dates = np.array([np.datetime64(m.match_date) for m in matches])

    w = _time_weights(dates, half_life_days)

    # Prior di attacco/difesa per squadra (default 1.0).
    prior_att = np.ones(n_teams)
    prior_def = np.ones(n_teams)
    if strength_prior:
        for name, i in idx.items():
            s = strength_prior.get(name, 1.0)
            s = max(s, 1e-6)
            prior_att[i] = math.sqrt(s)
            prior_def[i] = 1.0 / math.sqrt(s)

    att_num = np.zeros(n_teams)
    np.add.at(att_num, home, w * hs)
    np.add.at(att_num, away, w * as_)
    def_num = np.zeros(n_teams)
    np.add.at(def_num, away, w * hs)
    np.add.at(def_num, home, w * as_)

    eps = 1e-9
    attack = prior_att.copy()
    defense = prior_def.copy()
    base = float((w * (hs + as_)).sum() / (2.0 * w.sum()))
    gamma = 1.3

    for _ in range(max_iter):
        prev = np.concatenate([attack, defense, [gamma]])
        hf = np.where(neutral, 1.0, gamma)

        # Pseudo-conteggio verso il PRIOR (Elo): k partite fantasma in cui la
        # squadra segna/subisce al tasso del prior, non a quello medio.
        k = shrinkage * base

        att_den = np.zeros(n_teams)
        np.add.at(att_den, home, w * base * defense[away] * hf)
        np.add.at(att_den, away, w * base * defense[home] * 1.0)
        attack = (att_num + k * prior_att) / (att_den + k + eps)

        def_den = np.zeros(n_teams)
        np.add.at(def_den, away, w * base * attack[home] * hf)
        np.add.at(def_den, home, w * base * attack[away] * 1.0)
        defense = (def_num + k * prior_def) / (def_den + k + eps)

        mask = ~neutral
        num_g = (w[mask] * hs[mask]).sum()
        den_g = (w[mask] * base * attack[home[mask]] * defense[away[mask]]).sum()
        gamma = num_g / (den_g + eps)

        ma = attack.mean(); attack /= ma; base *= ma
        mb = defense.mean(); defense /= mb; base *= mb

        if np.max(np.abs(np.concatenate([attack, defense, [gamma]]) - prev)) < tol:
            break

    hf = np.where(neutral, 1.0, gamma)
    lam = base * attack[home] * defense[away] * hf
    mu = base * attack[away] * defense[home]
    rho = estimate_rho(lam, mu, hs, as_, w)

    teams = [
        Team(name=name, attack=float(attack[i]), defense=float(defense[i]))
        for name, i in idx.items()
    ]
    teams.sort(key=lambda t: t.attack, reverse=True)

    return RatingsResult(
        teams=teams,
        base_goals=float(base),
        home_advantage=float(gamma),
        rho=float(rho),
        n_matches=len(matches),
    )
