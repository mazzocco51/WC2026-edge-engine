"""
engine.py
=========
Orchestratore del modello: collega dati reali (CC0) -> rating -> torneo 48 squadre
e fornisce le probabilita' del modello, con caching per non rifare tutto a ogni
richiesta dell'API.

- I rating e i gironi dipendono solo dai dati storici: calcolati UNA volta (cache).
- Le probabilita' del modello dipendono solo da `n_simulations`: anch'esse messe
  in cache, cosi' richieste ripetute con lo stesso valore sono istantanee.

DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.models import Team
from app.core.ratings import RatingsResult, estimate_ratings
from app.core.tournament import simulate_world_cup
from app.data.loader import load_results
from app.data.wc2026 import build_groups_from_ratings


@dataclass(frozen=True)
class Field:
    """Rating stimati + gironi del Mondiale, pronti per la simulazione."""

    ratings: RatingsResult
    groups: dict[str, list[Team]]


@lru_cache(maxsize=1)
def load_field(since_year: int = 2018) -> Field:
    """Scarica i dati CC0, stima i rating e costruisce i 12 gironi (in cache)."""
    matches = load_results(since_year=since_year)
    ratings = estimate_ratings(matches)
    groups = build_groups_from_ratings(ratings)
    return Field(ratings=ratings, groups=groups)


@lru_cache(maxsize=16)
def model_probabilities(n_simulations: int = 10_000) -> dict[str, float]:
    """Probabilita' di vittoria delle 48 squadre dal Monte Carlo (dati reali, in cache)."""
    field = load_field()
    return simulate_world_cup(
        field.groups, field.ratings.base_goals, n_simulations=n_simulations
    )
