"""
ratings.py
==========
Stima dei coefficienti di ATTACCO e DIFESA di ogni nazionale a partire dai
risultati storici, per alimentare il modello di Poisson con dati REALI al posto
del dataset dummy.

==================== FONTE DATI (UNICA) ====================
Risultati storici: Mart Jürisoo (martj42), licenza CC0-1.0 (pubblico dominio).
https://github.com/martj42/international_results  (vedi DATA_SOURCES.md)
DISCLAIMER: progetto a fini EDUCATIVI/DIMOSTRATIVI, NON commerciale.
===========================================================

Metodo
------
Modello "double Poisson" con forze di attacco/difesa (Maher 1982; Dixon-Coles
1997). I gol segnati dalla squadra i contro la j hanno media:

    lambda = base * attacco_i * difesa_j   (* vantaggio_campo se i gioca in casa)

dove `attacco`/`difesa` sono moltiplicatori relativi alla media (1.0 = squadra
media); `difesa > 1` = subisce piu' gol (difesa debole). Stessa convenzione del
modello in poisson.py, quindi i Team prodotti qui sono usabili senza modifiche.

Stima: massima verosimiglianza Poisson via "iterative scaling" (aggiornamenti
moltiplicativi a punto fisso) — robusto e dipendente solo da NumPy. Le partite
sono pesate con un decadimento temporale (time-decay): le piu' recenti contano
di piu', cosi' i rating riflettono la forma attuale delle squadre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.core.models import Team
from app.data.loader import Match

# Emivita del peso temporale: una partita di ~2 anni fa pesa la meta' di una di
# oggi. Parametro di calibrazione (piu' corto = piu' reattivo, piu' rumoroso).
DEFAULT_HALF_LIFE_DAYS: float = 365 * 2


@dataclass(frozen=True)
class RatingsResult:
    """Esito della stima: squadre con rating + parametri globali del modello."""

    teams: list[Team]            # Team(name, attacco, difesa) per ogni nazionale
    base_goals: float            # gol attesi medio (squadra media vs media, neutro)
    home_advantage: float        # moltiplicatore vantaggio campo (>1)
    n_matches: int               # partite usate nella stima


def _time_weights(dates: np.ndarray, half_life_days: float) -> np.ndarray:
    """Peso esponenziale: w = 0.5 ** (eta_giorni / emivita)."""
    most_recent = dates.max()
    age_days = (most_recent - dates).astype("timedelta64[D]").astype(float)
    decay = math.log(2.0) / half_life_days
    return np.exp(-decay * age_days)


def estimate_ratings(
    matches: list[Match],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> RatingsResult:
    """
    Stima attacco/difesa di ogni squadra dai risultati osservati.

    Args:
        matches: lista di Match (dal loader CC0).
        half_life_days: emivita del decadimento temporale.
        max_iter: iterazioni massime del punto fisso.
        tol: soglia di convergenza sui parametri.

    Returns:
        RatingsResult con i Team (attacco/difesa) e i parametri globali.
    """
    if not matches:
        raise ValueError("Nessuna partita fornita per la stima dei rating.")

    # --- Indicizzazione delle squadre ---
    names = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
    idx = {name: i for i, name in enumerate(names)}
    n_teams = len(names)

    # --- Vettorializzazione dei dati ---
    home = np.array([idx[m.home_team] for m in matches])
    away = np.array([idx[m.away_team] for m in matches])
    hs = np.array([m.home_score for m in matches], dtype=float)
    as_ = np.array([m.away_score for m in matches], dtype=float)
    neutral = np.array([m.neutral for m in matches], dtype=bool)
    dates = np.array([np.datetime64(m.match_date) for m in matches])

    w = _time_weights(dates, half_life_days)

    # --- Numeratori (solo dati, costanti tra le iterazioni) ---
    # Gol segnati da ciascuna squadra (numeratore dell'attacco).
    att_num = np.zeros(n_teams)
    np.add.at(att_num, home, w * hs)
    np.add.at(att_num, away, w * as_)
    # Gol subiti da ciascuna squadra (numeratore della difesa).
    def_num = np.zeros(n_teams)
    np.add.at(def_num, away, w * hs)   # la squadra 'away' subisce i gol di 'home'
    np.add.at(def_num, home, w * as_)  # la squadra 'home' subisce i gol di 'away'

    eps = 1e-9  # evita divisioni per zero

    # --- Inizializzazione ---
    attack = np.ones(n_teams)
    defense = np.ones(n_teams)
    base = float((w * (hs + as_)).sum() / (2.0 * w.sum()))  # gol medi per squadra
    gamma = 1.3  # stima iniziale del vantaggio campo

    for _ in range(max_iter):
        prev = np.concatenate([attack, defense, [gamma]])

        # Fattore casa applicato all'evento "gol della squadra di casa".
        hf = np.where(neutral, 1.0, gamma)

        # --- Aggiorna ATTACCO ---
        att_den = np.zeros(n_teams)
        np.add.at(att_den, home, w * base * defense[away] * hf)   # i in casa
        np.add.at(att_den, away, w * base * defense[home] * 1.0)  # i in trasferta
        attack = att_num / (att_den + eps)

        # --- Aggiorna DIFESA ---
        def_den = np.zeros(n_teams)
        np.add.at(def_den, away, w * base * attack[home] * hf)    # subisce da casa
        np.add.at(def_den, home, w * base * attack[away] * 1.0)   # subisce da away
        defense = def_num / (def_den + eps)

        # --- Aggiorna vantaggio campo (solo match non neutri) ---
        mask = ~neutral
        num_g = (w[mask] * hs[mask]).sum()
        den_g = (w[mask] * base * attack[home[mask]] * defense[away[mask]]).sum()
        gamma = num_g / (den_g + eps)

        # --- Normalizza i moltiplicatori a media 1 (assorbe la scala in `base`) ---
        ma = attack.mean()
        attack /= ma
        base *= ma
        mb = defense.mean()
        defense /= mb
        base *= mb

        # --- Convergenza ---
        cur = np.concatenate([attack, defense, [gamma]])
        if np.max(np.abs(cur - prev)) < tol:
            break

    teams = [
        Team(name=name, attack=float(attack[i]), defense=float(defense[i]))
        for name, i in idx.items()
    ]
    teams.sort(key=lambda t: t.attack, reverse=True)

    return RatingsResult(
        teams=teams,
        base_goals=float(base),
        home_advantage=float(gamma),
        n_matches=len(matches),
    )
