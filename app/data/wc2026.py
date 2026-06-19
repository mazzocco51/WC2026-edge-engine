"""
wc2026.py
=========
Field del Mondiale 2026: 48 squadre in 12 gironi.

Due modalita':
1. build_real_groups()  -> usa il SORTEGGIO UFFICIALE 2026 (default). Ogni
   squadra ha il suo girone reale, quindi il cammino simulato e' quello vero.
2. build_groups_from_ratings() -> sorteggio fittizio a fasce dai rating
   (riproducibile, usato come fallback / per esperimenti "what-if").

Perche' il sorteggio reale conta: con un sorteggio casuale la probabilita' di
vincere dipende dal girone capitato per caso; per squadre di forza simile e'
il sorteggio a decidere l'ordine, non la forza. Usando i gironi veri questo
artefatto sparisce.

DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

import string
import unicodedata

import numpy as np

from app.core.models import Team
from app.core.ratings import RatingsResult

# Sorteggio ufficiale del Mondiale 2026 (12 gironi da 4). Fonte: NBC Sports.
REAL_GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Alias dei nomi del sorteggio -> nomi usati nel dataset martj42 (normalizzati).
_ALIASES: dict[str, str] = {
    "usa": "united states",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "bosnia herzegovina": "bosnia and herzegovina",
}


def _norm(name: str) -> str:
    """minuscolo, niente accenti/trattini, spazi compattati."""
    s = name.strip().lower().replace("-", " ")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.split())


def build_real_groups(ratings: RatingsResult) -> dict[str, list[Team]]:
    """
    Costruisce i 12 gironi UFFICIALI mappando ogni squadra ai rating stimati.

    Raises:
        ValueError: se qualche squadra del sorteggio non e' presente nei rating
                    (segnala il problema invece di ignorarlo silenziosamente).
    """
    lookup = {_norm(t.name): t for t in ratings.teams}
    groups: dict[str, list[Team]] = {}
    missing: list[str] = []
    for g, names in REAL_GROUPS_2026.items():
        teams: list[Team] = []
        for n in names:
            key = _ALIASES.get(_norm(n), _norm(n))
            team = lookup.get(key)
            if team is None:
                missing.append(n)
            else:
                teams.append(team)
        groups[g] = teams
    if missing:
        raise ValueError(f"Squadre del sorteggio non trovate nei rating: {missing}")
    return groups


def build_groups_from_ratings(
    ratings: RatingsResult,
    n_teams: int = 48,
    seed: int | None = 7,
) -> dict[str, list[Team]]:
    """
    Sorteggio fittizio a fasce (fallback / esperimenti what-if).

    Seleziona le migliori `n_teams` per forza, le divide in 4 fasce da 12 e ne
    pesca una per fascia in ciascun girone. Riproducibile dato un seme.
    """
    if n_teams != 48:
        raise ValueError("Il formato 2026 richiede 48 squadre.")

    def strength(t: Team) -> float:
        return t.attack / max(t.defense, 1e-6)

    pool = sorted(ratings.teams, key=strength, reverse=True)[:n_teams]
    pots = [pool[i * 12 : (i + 1) * 12] for i in range(4)]
    rng = np.random.default_rng(seed)
    for pot in pots:
        rng.shuffle(pot)
    group_names = list(string.ascii_uppercase[:12])
    groups: dict[str, list[Team]] = {g: [] for g in group_names}
    for pot in pots:
        for idx, team in enumerate(pot):
            groups[group_names[idx]].append(team)
    return groups
