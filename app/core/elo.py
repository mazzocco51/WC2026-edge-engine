"""
elo.py
======
Rating Elo delle nazionali, calcolato DA NOI sui risultati storici CC0.

Perche' un Elo nostro
---------------------
Il ranking FIFA, dal 2018, e' a sua volta un sistema Elo. Calcolando l'Elo
direttamente dai risultati (martj42, CC0) otteniamo un rating di forza
equivalente al ranking FIFA ma 100% pubblico dominio: nessun dato proprietario,
nessuna licenza grigia. L'Elo cattura la forza-squadra meglio del semplice
rapporto-gol perche' aggiorna in base a sorpresa del risultato, margine di
vittoria e importanza della partita (come eloratings.net / FIFA).

Formula (World Football Elo)
----------------------------
Per ogni partita: R_new = R_old + K * G * (W - We)
  - W  = esito reale (1 vittoria, 0.5 pareggio, 0 sconfitta)
  - We = esito atteso = 1 / (1 + 10^(-dr/400)), dr = diff. rating (+ vantaggio casa)
  - K  = peso dell'importanza della partita (Mondiale > amichevole)
  - G  = moltiplicatore margine di vittoria (vincere 4-0 sposta piu' di 1-0)

Solo NumPy/stdlib. DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

from app.data.loader import Match

# Pesi di importanza (K) per tipo di torneo: i Mondiali contano piu' delle amichevoli.
_TOURNAMENT_K: dict[str, float] = {
    "FIFA World Cup": 60.0,
    "Copa America": 50.0,
    "UEFA Euro": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "UEFA Nations League": 40.0,
    "FIFA World Cup qualification": 40.0,
    "Friendly": 20.0,
}
_DEFAULT_K: float = 30.0
_HOME_ADVANTAGE_ELO: float = 65.0   # bonus rating per chi gioca in casa
_BASE_RATING: float = 1500.0


def _k_factor(tournament: str) -> float:
    """K per importanza: match esatto o per parola chiave, altrimenti default."""
    if tournament in _TOURNAMENT_K:
        return _TOURNAMENT_K[tournament]
    for key, k in _TOURNAMENT_K.items():
        if key.lower() in tournament.lower():
            return k
    return _DEFAULT_K


def _goal_multiplier(goal_diff: int) -> float:
    """Moltiplicatore margine di vittoria (Dixon-Coles/eloratings style)."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0   # 3 gol -> 1.75, 4 -> 1.875, ...


def compute_elo(matches: list[Match]) -> dict[str, float]:
    """
    Calcola l'Elo finale di ogni nazionale processando le partite in ordine.

    Args:
        matches: storico partite (idealmente gia' ordinato per data).

    Returns:
        {nome_squadra: rating_elo}. Piu' alto = piu' forte.
    """
    ordered = sorted(matches, key=lambda m: m.match_date)
    ratings: dict[str, float] = {}

    for m in ordered:
        rh = ratings.get(m.home_team, _BASE_RATING)
        ra = ratings.get(m.away_team, _BASE_RATING)

        # Vantaggio campo solo se non neutro.
        adv = 0.0 if m.neutral else _HOME_ADVANTAGE_ELO
        dr = (rh + adv) - ra
        we_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        if m.home_score > m.away_score:
            w_home = 1.0
        elif m.home_score < m.away_score:
            w_home = 0.0
        else:
            w_home = 0.5

        k = _k_factor(m.tournament) * _goal_multiplier(m.home_score - m.away_score)
        delta = k * (w_home - we_home)

        ratings[m.home_team] = rh + delta
        ratings[m.away_team] = ra - delta

    return ratings


def elo_strength_multipliers(elo: dict[str, float]) -> dict[str, float]:
    """
    Converte i rating Elo in moltiplicatori di forza centrati su 1.0
    (media geometrica = 1), pronti come prior per attacco/difesa.

    Una differenza di ~scale punti Elo corrisponde a un fattore ~e di forza.
    """
    import math

    if not elo:
        return {}
    mean = sum(elo.values()) / len(elo)
    scale = 400.0  # 400 punti Elo ~ fattore e nella forza relativa
    return {team: math.exp((r - mean) / scale) for team, r in elo.items()}
