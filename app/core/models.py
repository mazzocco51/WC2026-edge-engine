"""
models.py
=========
Strutture dati fondamentali del core matematico.

Usiamo `dataclass` (frozen=True) perche' sono immutabili, leggere e
auto-documentanti grazie ai type hints. Niente dipendenze pesanti.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    """
    Rappresenta una squadra nazionale e la sua "forza" statistica.

    I parametri `attack` e `defense` sono moltiplicatori RELATIVI alla media
    del torneo (1.0 = squadra perfettamente media):

    - attack  > 1.0  -> segna piu' della media
    - defense < 1.0  -> subisce MENO della media (difesa forte)
    - defense > 1.0  -> subisce PIU' della media (difesa debole)

    Questa normalizzazione attorno a 1.0 rende il modello di Poisson
    immediato da leggere e calibrare (vedi poisson.py).
    """

    name: str
    attack: float   # forza offensiva relativa alla media (1.0 = media)
    defense: float  # solidita' difensiva relativa alla media (1.0 = media)


@dataclass(frozen=True)
class MatchResult:
    """
    Risultato di una singola partita simulata.

    `home`/`away` sono solo etichette: nel Mondiale le partite sono in campo
    neutro, quindi non applichiamo alcun vantaggio casalingo.
    """

    home: Team
    away: Team
    home_goals: int
    away_goals: int
    # Vincitore della partita. In un girone puo' essere None (pareggio);
    # in un match a eliminazione viene SEMPRE valorizzato (eventuali rigori).
    winner: Team | None

    @property
    def is_draw(self) -> bool:
        """True se i tempi regolamentari finiscono in parita'."""
        return self.home_goals == self.away_goals
