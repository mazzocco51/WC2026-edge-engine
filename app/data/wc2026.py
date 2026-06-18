"""
wc2026.py
=========
Costruzione del "field" del Mondiale 2026: 48 squadre in 12 gironi.

Approccio data-driven (riproducibile)
-------------------------------------
Per non dipendere da un sorteggio ufficiale codificato a mano (fonte di
possibili errori), costruiamo i gironi a partire dai RATING stimati:
1. si prendono le 48 nazionali piu' forti per rating;
2. si dividono in 4 fasce (pot) da 12, come fa la FIFA;
3. si sorteggia una squadra da ogni fascia in ciascuno dei 12 gironi.

Cosi' la struttura rispecchia il metodo reale (teste di serie per fascia) ed e'
100% riproducibile dato un seme. NB: e' un'approssimazione del sorteggio vero;
se hai i gironi ufficiali puoi passarli direttamente a simulate_world_cup().

DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

import string

import numpy as np

from app.core.models import Team
from app.core.ratings import RatingsResult


def _strength(team: Team) -> float:
    """Scalare di forza: attacco alto e difesa solida -> valore alto."""
    return team.attack / max(team.defense, 1e-6)


def build_groups_from_ratings(
    ratings: RatingsResult,
    n_teams: int = 48,
    seed: int | None = 7,
) -> dict[str, list[Team]]:
    """
    Seleziona le migliori `n_teams` nazionali e le distribuisce in 12 gironi.

    Args:
        ratings: risultato della stima (lista di Team con attacco/difesa).
        n_teams: dimensione del field (48 per il Mondiale 2026).
        seed: seme del sorteggio (riproducibilita').

    Returns:
        {nome_girone ('A'..'L'): [4 Team]}.
    """
    if n_teams != 48:
        raise ValueError("Il formato 2026 richiede 48 squadre.")

    # Le migliori 48 per forza.
    pool = sorted(ratings.teams, key=_strength, reverse=True)[:n_teams]

    # 4 fasce da 12 (pot 1 = piu' forti).
    pots = [pool[i * 12 : (i + 1) * 12] for i in range(4)]

    rng = np.random.default_rng(seed)
    for pot in pots:
        rng.shuffle(pot)

    # Una squadra per fascia in ciascuno dei 12 gironi.
    group_names = list(string.ascii_uppercase[:12])  # A..L
    groups: dict[str, list[Team]] = {g: [] for g in group_names}
    for pot in pots:
        for idx, team in enumerate(pot):
            groups[group_names[idx]].append(team)
    return groups
