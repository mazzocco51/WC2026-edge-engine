"""
simulator.py
============
Motore Monte Carlo del torneo.

Strategia
---------
1. `simulate_match`   -> simula UNA partita (gol Poisson + eventuali rigori).
2. `simulate_bracket` -> simula UN torneo completo a eliminazione diretta.
3. `run_tournament`   -> ripete il bracket N volte (default 10.000) e ricava
                         la probabilita' di vittoria di ciascuna squadra.

Per lo Step 1 il "bracket" e' un mini-tabellone a 4 squadre (2 semifinali +
finale): abbastanza per validare la logica. Lo sostituiremo con il vero
tabellone del Mondiale 2026 in uno step successivo, senza toccare match/rigori.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from app.core.models import MatchResult, Team
from app.core.poisson import expected_goals, sample_goals


def simulate_match(
    home: Team,
    away: Team,
    rng: np.random.Generator,
    knockout: bool = True,
) -> MatchResult:
    """
    Simula una singola partita.

    I gol delle due squadre sono indipendenti e seguono Poisson(lambda), con
    lambda calcolato dal modello in poisson.py. In caso di pareggio in una
    partita a eliminazione (`knockout=True`) decidiamo il vincitore con i
    "rigori": una probabilita' ponderata sulla forza d'attacco delle squadre
    (piu' realistico di un puro 50/50).

    Args:
        home, away: le due squadre (etichette, campo neutro).
        rng: generatore casuale NumPy.
        knockout: se True forziamo un vincitore (rigori) ai pareggi.

    Returns:
        MatchResult con i gol e il vincitore.
    """
    lam_home = expected_goals(home, away)
    lam_away = expected_goals(away, home)

    home_goals = int(sample_goals(lam_home, rng))
    away_goals = int(sample_goals(lam_away, rng))

    winner: Team | None
    if home_goals > away_goals:
        winner = home
    elif away_goals > home_goals:
        winner = away
    else:
        # Pareggio.
        if not knockout:
            winner = None  # nei gironi il pareggio e' un risultato valido
        else:
            # Rigori: probabilita' proporzionale alla forza d'attacco.
            p_home = home.attack / (home.attack + away.attack)
            winner = home if rng.random() < p_home else away

    return MatchResult(
        home=home,
        away=away,
        home_goals=home_goals,
        away_goals=away_goals,
        winner=winner,
    )


def simulate_bracket(teams: list[Team], rng: np.random.Generator) -> Team:
    """
    Simula un singolo torneo a eliminazione diretta e ritorna il campione.

    Per lo Step 1 assumiamo esattamente 4 squadre (semifinali + finale).
    Accoppiamento: (0 vs 3) e (1 vs 2), tipico seeding testa-coda.

    Args:
        teams: lista di squadre (lunghezza 4 in questo step).
        rng: generatore casuale NumPy.

    Returns:
        La squadra vincitrice del torneo.
    """
    if len(teams) != 4:
        raise ValueError(f"Lo Step 1 richiede 4 squadre, ricevute {len(teams)}.")

    # Semifinali (testa-coda).
    sf1 = simulate_match(teams[0], teams[3], rng, knockout=True).winner
    sf2 = simulate_match(teams[1], teams[2], rng, knockout=True).winner
    assert sf1 is not None and sf2 is not None  # knockout garantisce un vincitore

    # Finale.
    final = simulate_match(sf1, sf2, rng, knockout=True).winner
    assert final is not None
    return final


def run_tournament(
    teams: list[Team],
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Esegue la simulazione Monte Carlo dell'intero torneo N volte.

    Args:
        teams: squadre partecipanti.
        n_simulations: numero di tornei simulati (default 10.000).
        seed: seme per la riproducibilita' (None = casuale ad ogni run).

    Returns:
        Dizionario {nome_squadra: probabilita_vittoria} ordinato in modo
        decrescente. Le probabilita' sommano a 1.0 (a meno di arrotondamenti).
    """
    rng = np.random.default_rng(seed)
    champions: Counter[str] = Counter()

    for _ in range(n_simulations):
        champion = simulate_bracket(teams, rng)
        champions[champion.name] += 1

    # Converte i conteggi in probabilita' [0, 1].
    probabilities = {
        team.name: champions[team.name] / n_simulations for team in teams
    }
    # Ordina dal favorito all'outsider.
    return dict(sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True))
