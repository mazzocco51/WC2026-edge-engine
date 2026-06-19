"""
dixon_coles.py
==============
Correzione di Dixon-Coles (1997) al modello di Poisson.

Problema del Poisson "puro"
---------------------------
Assumendo i gol delle due squadre indipendenti, il modello SOTTOSTIMA i
risultati a basso punteggio e i pareggi (0-0, 1-0, 0-1, 1-1) che nel calcio
reale sono piu' frequenti del previsto.

Soluzione (Dixon-Coles)
-----------------------
Si moltiplica la probabilita' congiunta dei soli quattro punteggi bassi per un
fattore tau che dipende da un parametro di correlazione `rho`:

    tau(0,0) = 1 - lam*mu*rho
    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1   altrimenti

dove lam = gol attesi casa, mu = gol attesi trasferta. rho<0 (tipico) aumenta i
pareggi a basso punteggio. Stimiamo rho per massima verosimiglianza.

Solo NumPy: nessuna dipendenza extra.
"""

from __future__ import annotations

import math

import numpy as np


def _poisson_pmf(lam: float, kmax: int) -> np.ndarray:
    """pmf di Poisson per k = 0..kmax (calcolata senza scipy)."""
    k = np.arange(kmax + 1)
    # log per stabilita': k*log(lam) - lam - log(k!)
    with np.errstate(divide="ignore"):
        logp = k * np.log(lam) - lam - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Fattore di correzione DC per il punteggio (x, y)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 8) -> np.ndarray:
    """
    Matrice (max_goals+1)x(max_goals+1) delle probabilita' di ogni punteggio,
    con correzione DC sui quattro punteggi bassi, rinormalizzata a somma 1.
    """
    ph = _poisson_pmf(lam, max_goals)
    pa = _poisson_pmf(mu, max_goals)
    m = np.outer(ph, pa)  # gol casa x gol trasferta
    # Correzione DC sui 4 angoli bassi.
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)  # evita probabilita' negative per rho estremi
    return m / m.sum()


def outcome_probs(lam: float, mu: float, rho: float, max_goals: int = 8) -> tuple[float, float, float]:
    """Probabilita' (vittoria_casa, pareggio, vittoria_trasferta) con DC."""
    m = score_matrix(lam, mu, rho, max_goals)
    home = float(np.tril(m, -1).sum())  # gol_casa > gol_trasferta
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def estimate_rho(
    lam: np.ndarray,
    mu: np.ndarray,
    hs: np.ndarray,
    as_: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """
    Stima rho per massima verosimiglianza (ricerca su griglia).

    Solo il termine tau dipende da rho, quindi massimizziamo
        sum_m  w_m * log( tau(x_m, y_m; rho) )
    sui soli match a basso punteggio (gli altri hanno tau=1 -> log=0).

    Args:
        lam, mu: gol attesi casa/trasferta per ogni match.
        hs, as_: gol reali casa/trasferta.
        weights: pesi (es. time-decay).
        grid: valori di rho da provare.

    Returns:
        rho stimato.
    """
    if grid is None:
        grid = np.linspace(-0.2, 0.2, 81)

    # Maschere dei 4 punteggi bassi.
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)

    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        t = np.ones_like(lam)
        t[m00] = 1.0 - lam[m00] * mu[m00] * rho
        t[m01] = 1.0 + lam[m01] * rho
        t[m10] = 1.0 + mu[m10] * rho
        t[m11] = 1.0 - rho
        if np.any(t <= 0):
            continue  # rho non ammissibile (tau negativo)
        ll = float(np.sum(weights * np.log(t)))
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


def sample_score(lam: float, mu: float, rho: float, rng: np.random.Generator, max_goals: int = 8) -> tuple[int, int]:
    """Campiona un punteggio (gol_casa, gol_trasferta) dalla distribuzione DC."""
    m = score_matrix(lam, mu, rho, max_goals)
    idx = rng.choice(m.size, p=m.ravel())
    return divmod(int(idx), max_goals + 1)
