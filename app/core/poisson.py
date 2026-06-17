"""
poisson.py
==========
Modello dei gol basato sulla distribuzione di Poisson.

Idea statistica
---------------
Il numero di gol segnati da una squadra in una partita di calcio e' ben
approssimato da una variabile di Poisson con parametro lambda (lambda = numero
atteso di gol). Stimiamo lambda per ogni squadra a partire da:

    lambda_A = MEDIA_GOL_LEGA * attacco_A * difesa_B

dove `attacco`/`difesa` sono i moltiplicatori relativi definiti in models.py.
Esempio: se A attacca a 1.3 e B difende a 1.2 (difesa debole), con media 1.35,
allora lambda_A = 1.35 * 1.3 * 1.2 ~= 2.11 gol attesi.

Estraiamo poi i gol effettivi campionando da Poisson(lambda) con NumPy: e'
vettorizzato, velocissimo e gira benissimo su CPU (nessun GPU/LLM richiesto).
"""

from __future__ import annotations

import numpy as np

from app.core.models import Team

# Media gol per squadra a partita: ~1.35 e' un valore realistico per i Mondiali.
# Lo teniamo come costante di modulo cosi' e' facile da calibrare in seguito.
LEAGUE_AVG_GOALS: float = 1.35


def expected_goals(attacker: Team, defender: Team, avg: float = LEAGUE_AVG_GOALS) -> float:
    """
    Calcola lambda (gol attesi) per `attacker` contro `defender`.

    Args:
        attacker: squadra che attacca.
        defender: squadra che difende.
        avg: media gol del torneo (parametro di calibrazione).

    Returns:
        Numero atteso di gol (float, sempre > 0).
    """
    return avg * attacker.attack * defender.defense


def sample_goals(
    lam: float,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray | int:
    """
    Campiona i gol da una distribuzione di Poisson(lam).

    Passiamo SEMPRE un `np.random.Generator` esplicito invece di usare lo stato
    globale di NumPy: rende le simulazioni riproducibili (seed) e thread-safe.

    Args:
        lam: parametro lambda (gol attesi), deve essere >= 0.
        rng: generatore di numeri casuali di NumPy.
        size: se None ritorna un singolo int (una partita); se intero N
              ritorna un array di N estrazioni (vettorizzato, per Monte Carlo).

    Returns:
        Un int (size=None) oppure un np.ndarray di interi (size=N).
    """
    return rng.poisson(lam=lam, size=size)
