"""
tournament.py
=============
Monte Carlo simulation of the 2026 World Cup in the official 48-team format.

Format (FIFA 2026): 12 groups of 4 (round robin); top 2 per group + 8 best
third-placed teams advance (32 teams); single elimination R32 -> ... -> Final.

Host advantage: USA, Canada and Mexico host the tournament, so their matches are
NOT neutral. A host nation's expected goals are multiplied by HOST_ADVANTAGE,
a literature-calibrated value (~+0.2 expected goals, i.e. ~1.2x).

Third-placed allocation and bracket pairing use a documented simplification
(standard seeding), not FIFA's exact combinatorial table.

DISCLAIMER: educational/demonstrative project, non-commercial.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

from app.core.models import Team

HOSTS: frozenset[str] = frozenset({"United States", "Canada", "Mexico"})
HOST_ADVANTAGE: float = 1.2   # ~+0.2 gol attesi (fattore campo prudente)

_ROUND_NAMES = {
    32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
    4: "Semi-finals", 2: "Final",
}


def _play(a, b, base, rng, knockout, hosts, host_adv):
    """Partita veloce: ritorna (gol_a, gol_b, vincitore|None, rigori?)."""
    la = base * a.attack * b.defense
    lb = base * b.attack * a.defense
    if a.name in hosts:
        la *= host_adv
    if b.name in hosts:
        lb *= host_adv
    ga = int(rng.poisson(la))
    gb = int(rng.poisson(lb))
    if ga > gb:
        return ga, gb, a, False
    if gb > ga:
        return ga, gb, b, False
    if not knockout:
        return ga, gb, None, False
    p_a = a.attack / (a.attack + b.attack)
    return ga, gb, (a if rng.random() < p_a else b), True


def _standings(group, base, rng, hosts, host_adv) -> list[dict]:
    """Simula il girone all'italiana e ritorna le squadre ordinate in classifica."""
    stats = {t.name: {"team": t, "pts": 0, "gf": 0, "ga": 0} for t in group}
    for i, j in itertools.combinations(range(len(group)), 2):
        gi, gj, _, _ = _play(group[i], group[j], base, rng, False, hosts, host_adv)
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
    seeds = [1]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        nxt: list[int] = []
        for s in seeds:
            nxt.append(s); nxt.append(m - s)
        seeds = nxt
    return seeds


_ORDER_32 = _bracket_seed_order(32)


def _qualified_seeds(groups, base, rng, hosts, host_adv) -> list[Team]:
    """Simula i gironi e ritorna le 32 qualificate ordinate per testa di serie."""
    winners, runners, thirds = [], [], []
    for group in groups:
        table = _standings(group, base, rng, hosts, host_adv)
        winners.append(table[0]); runners.append(table[1]); thirds.append(table[2])
    key = lambda s: (s["pts"], s["gd"], s["gf"], s["rand"])
    thirds.sort(key=key, reverse=True)
    ranked = (
        sorted(winners, key=key, reverse=True)
        + sorted(runners, key=key, reverse=True)
        + sorted(thirds[:8], key=key, reverse=True)
    )
    return [s["team"] for s in ranked]


def _knockout(seeded, base, rng, hosts, host_adv) -> Team:
    """Eliminazione veloce (solo campione), per il calcolo delle probabilita'."""
    participants = [seeded[s - 1] for s in _ORDER_32]
    while len(participants) > 1:
        nxt = []
        for k in range(0, len(participants), 2):
            _, _, w, _ = _play(participants[k], participants[k + 1], base, rng, True, hosts, host_adv)
            nxt.append(w)
        participants = nxt
    return participants[0]


def _knockout_detail(seeded, base, rng, hosts, host_adv):
    """Eliminazione con registrazione dei risultati di ogni round."""
    participants = [seeded[s - 1] for s in _ORDER_32]
    rounds = []
    while len(participants) > 1:
        rname = _ROUND_NAMES.get(len(participants), f"Round of {len(participants)}")
        matches, nxt = [], []
        for k in range(0, len(participants), 2):
            a, b = participants[k], participants[k + 1]
            ga, gb, w, pens = _play(a, b, base, rng, True, hosts, host_adv)
            matches.append({
                "a": a.name, "b": b.name, "ga": ga, "gb": gb,
                "winner": w.name, "pens": pens,
            })
            nxt.append(w)
        rounds.append({"name": rname, "matches": matches})
        participants = nxt
    return participants[0].name, rounds


def _simulate_once(groups, base, rng, hosts, host_adv) -> Team:
    return _knockout(_qualified_seeds(groups, base, rng, hosts, host_adv), base, rng, hosts, host_adv)


def simulate_world_cup(
    groups: dict[str, list[Team]],
    base_goals: float,
    n_simulations: int = 10_000,
    seed: int | None = 42,
    hosts: frozenset[str] = HOSTS,
    host_advantage: float = HOST_ADVANTAGE,
) -> dict[str, float]:
    """Monte Carlo: {squadra: probabilita_vittoria}, ordinato, somma 1.0."""
    n_teams = sum(len(g) for g in groups.values())
    if n_teams != 48 or len(groups) != 12:
        raise ValueError(f"Atteso 12 gironi x 4 = 48 squadre, ricevute {n_teams} in {len(groups)} gironi.")
    group_list = list(groups.values())
    rng = np.random.default_rng(seed)
    champions: Counter[str] = Counter()
    for _ in range(n_simulations):
        champions[_simulate_once(group_list, base_goals, rng, hosts, host_advantage).name] += 1
    all_teams = [t.name for g in group_list for t in g]
    probs = {name: champions[name] / n_simulations for name in all_teams}
    return dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))


def simulate_brackets(
    groups: dict[str, list[Team]],
    base_goals: float,
    n: int = 3,
    seed: int | None = 7,
    hosts: frozenset[str] = HOSTS,
    host_advantage: float = HOST_ADVANTAGE,
) -> list[dict]:
    """
    Ritorna `n` tabelloni a eliminazione completi con i risultati di ogni round.

    Ogni elemento: {"champion": str, "rounds": [{"name", "matches": [...]}]}.
    Serve alla UI per mostrare un esempio concreto di torneo simulato.
    """
    group_list = list(groups.values())
    rng = np.random.default_rng(seed)
    brackets = []
    for _ in range(n):
        seeds = _qualified_seeds(group_list, base_goals, rng, hosts, host_advantage)
        champ, rounds = _knockout_detail(seeds, base_goals, rng, hosts, host_advantage)
        brackets.append({"champion": champ, "rounds": rounds})
    return brackets
