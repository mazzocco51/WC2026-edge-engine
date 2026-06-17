"""
dummy.py
========
Dataset fittizio a 4 squadre per testare la logica del core SENZA dipendere
da API esterne o file storici. Numeri scelti a mano per avere un favorito
chiaro (Brazil) e un outsider (Canada), utile a validare le probabilita'.
"""

from __future__ import annotations

from app.core.models import Team

# attack > 1.0 = attacco sopra la media; defense < 1.0 = difesa solida.
DUMMY_TEAMS: list[Team] = [
    Team(name="Brazil",      attack=1.45, defense=0.75),  # forte favorito
    Team(name="France",      attack=1.30, defense=0.85),  # contender
    Team(name="Argentina",   attack=1.25, defense=0.90),  # contender
    Team(name="Canada",      attack=0.85, defense=1.20),  # outsider
]
