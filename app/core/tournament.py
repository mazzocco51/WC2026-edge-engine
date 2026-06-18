"""
tournament.py
=============
Simulazione Monte Carlo del Mondiale 2026 nel formato UFFICIALE a 48 squadre.

Formato (FIFA 2026)
-------------------
- 12 gironi da 4 squadre, girone all'italiana (3 partite a testa).
- Passano le prime 2 di ogni girone (24) + le 8 migliori terze (32 totali).
- Eliminazione diretta: R32 -> R16 -> QF -> SF -> Finale.

Il piazzamento esatto delle "terze" e l'accoppiamento del tabellone usano una
semplificazione DOCUMENTATA (teste di serie standard: la 1 e la 2 si possono
incontrare solo in finale), non la tabella combinatoria esatta della FIFA.

Tutte le partite sono in campo neutro. I gol seguono Poisson con media
`base * attacco_i * difesa_j`, dove `base` e' il valore stimato dai dati reali.

Nota performance: il "cuore caldo" (milioni di partite) evita la creazione di
oggetti MatchResult e legge gli attributi in variabili locali, per essere
abbastanza veloce da girare sotto il timeout di una richiesta HTTP.

DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from app.core.models import Team


def _play(a: Team, b: Team, base: float, rng: np.random.Generator, knockout: bool):
    """Partita veloce: ritorna (gol_a, gol_b, vincitore|None). Niente oggetti."""
    ga = rng.poisson(base * a.attack * b.defense)
    gb = rng.poisson(base * b.attack * a.defense)
    if ga > gb:
        return ga, gb, a
    if gb > ga:
        return ga, gb, b
    if not knockout:
        return ga, gb, None
    # Rigori: probabilita' proporzionale alla forza d'attacco.
    p_a = a.attack / (a.attack + b.attack)
    return ga, gb, (a if rng.random() < p_a else b)


def _standings(group: list[Team], base: float, rng: np.random.Generator) -> list[dict]:
    """Simula il girone all'italiana e ritorna le squadre ordinate in classifica."""
    stats = {t.name: {"team": t, "pts": 0, "gf": 0, "ga": 0} for t in group}
    for i, j in itertools.combinations(range(len(group)), 2):
        gi, gj, _ = _play(group[i], group[j], base, rng, knockout=False)
        si, sj = stats[group[i].name], stats[group[j].name]
        si["gf"] += gi; si["ga"] += gj
        sj["gf"] += gj; sj["ga"] += gi
        if gi > gj:
            si["pts"] += 3
        elif gj > gi:
            sj["pts"] += 3
        else:
            si["pts"] += 1; sj["pts"] += 1
    table = list(stats.values())
    for s in table:
        s["gd"] = s["gf"] - s["ga"]
        s["rand"] = rng.random()
    table.sort(key=lambda s: (s["pts"], s["gd"], s["gf"], s["rand"]), reverse=True)
    return table


def _bracket_seed_order(n: int) -> list[int]:
    """Ordine standard delle teste di serie per un tabellone da n (potenza di 2)."""
    seeds = [1]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        nxt: list[int] = []
        for s in seeds:
            nxt.append(s)
            nxt.append(m - s)
        seeds = nxt
    return seeds


# Ordine di tabellone per 32 squadre, calcolato una sola volta.
_ORDER_32 = _bracket_seed_order(32)


def _knockout(seeded_teams: list[Team], base: float, rng: np.random.Generator) -> Team:
    """Eliminazione diretta su 32 squadre gia' ordinate per testa di serie."""
    participants = [seeded_teams[s - 1] for s in _ORDER_32]
    while len(participants) > 1:
        nxt: list[Team] = []
        for k in range(0, len(participants), 2):
            _, _, w = _play(participants[k], participants[k + 1], base, rng, knockout=True)
            nxt.append(w)
        participants = nxt
    return participants[0]


def _simulate_once(groups: list[list[Team]], base: float, rng: np.random.Generator) -> Team:
    """Simula un intero torneo e ritorna il campione."""
    winners: list[dict] = []
    runners: list[dict] = []
    thirds: list[dict] = []
    for group in groups:
        table = _standings(group, base, rng)
        winners.append(table[0])
        runners.append(table[1])
        thirds.append(table[2])

    thirds.sort(key=lambda s: (s["pts"], s["gd"], s["gf"], s["rand"]), reverse=True)
    best_thirds = thirds[:8]

    key = lambda s: (s["pts"], s["gd"], s["gf"], s["rand"])
    ranked = (
        sorted(winners, key=key, reverse=True)
        + sorted(runners, key=key, reverse=True)
        + sorted(best_thirds, key=key, reverse=True)
    )
    return _knockout([s["team"] for s in ranked], base, rng)


def simulate_world_cup(
    groups: dict[str, list[Team]],
    base_goals: float,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Esegue la simulazione Monte Carlo del Mondiale 2026 a 48 squadre.

    Returns:
        {nome_squadra: probabilita_vittoria}, ordinato in modo decrescente.
        Le probabilita' sommano a 1.0.
    """
    n_teams = sum(len(g) for g in groups.values())
    if n_teams != 48 or len(groups) != 12:
        raise ValueError(f"Atteso 12 gironi x 4 = 48 squadre, ricevute {n_teams} in {len(groups)} gironi.")

    group_list = list(groups.values())
    rng = np.random.default_rng(seed)
    champions: Counter[str] = Counter()
    for _ in range(n_simulations):
        champions[_simulate_once(group_list, base_goals, rng).name] += 1

    all_teams = [t.name for g in group_list for t in g]
    probs = {name: champions[name] / n_simulations for name in all_teams}
    return dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
