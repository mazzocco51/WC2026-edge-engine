"""
divergence.py
=============
Confronto tra il NOSTRO modello statistico e il prediction market Polymarket.

Obiettivo (analitico, NON di scommessa)
---------------------------------------
Misurare *quanto* e *dove* la stima probabilistica del modello diverge dal
consenso del mercato. È uno studio di **efficienza di mercato**: se mercato e
modello concordano, il mercato sta "prezzando" bene quella squadra; dove
divergono, almeno una delle due fonti la sta sopra/sotto-stimando.

NB: questo strumento ha finalità esclusivamente educative/dimostrative. Non
fornisce consigli di scommessa e non calcola ritorni economici.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.analysis.names import normalize_team
from app.market.polymarket import MarketProbability


class Assessment(str, Enum):
    """Esito qualitativo del confronto, in ottica modello-vs-mercato."""

    MODEL_HIGHER = "Model > Market"   # il modello assegna piu' probabilita' del mercato
    ALIGNED = "Aligned"               # modello e mercato sostanzialmente d'accordo
    MODEL_LOWER = "Model < Market"    # il modello assegna meno probabilita' del mercato


@dataclass(frozen=True)
class DivergenceResult:
    """
    Riga di confronto modello-vs-mercato per una squadra.

    Attributi:
        team:        nome della squadra.
        model_prob:  probabilita' di vittoria dal Monte Carlo [0, 1].
        market_prob: probabilita' implicita di mercato (de-vig) [0, 1].
        divergence:  model_prob - market_prob (con segno).
        abs_divergence: |divergence|, per ordinare le discrepanze.
        assessment:  Assessment (Model > / = / < Market).
    """

    team: str
    model_prob: float
    market_prob: float
    divergence: float
    abs_divergence: float
    assessment: Assessment


def _classify(divergence: float, threshold: float) -> Assessment:
    """Classifica la divergenza data una soglia di tolleranza (in probabilita')."""
    if divergence > threshold:
        return Assessment.MODEL_HIGHER
    if divergence < -threshold:
        return Assessment.MODEL_LOWER
    return Assessment.ALIGNED


def compute_divergence(
    model_probs: dict[str, float],
    market_probs: list[MarketProbability],
    threshold: float = 0.02,
) -> list[DivergenceResult]:
    """
    Incrocia probabilita' del modello e del mercato e misura la divergenza.

    Il join avviene sui nomi normalizzati (vedi names.py). Vengono incluse solo
    le squadre presenti in ENTRAMBE le fonti.

    Args:
        model_probs:  {nome_squadra: probabilita'} dal Monte Carlo.
        market_probs: lista di MarketProbability (probabilita' di mercato de-vig).
        threshold:    soglia (in punti di probabilita') oltre la quale la
                      divergenza e' considerata "notevole". Default 0.02.

    Returns:
        Lista di DivergenceResult ordinata per divergenza assoluta decrescente
        (le discrepanze piu' grandi in cima).
    """
    market_by_key: dict[str, MarketProbability] = {
        normalize_team(mp.team): mp for mp in market_probs
    }

    results: list[DivergenceResult] = []
    for team, model_prob in model_probs.items():
        mp = market_by_key.get(normalize_team(team))
        if mp is None:
            continue

        divergence = model_prob - mp.fair_prob
        results.append(
            DivergenceResult(
                team=team,
                model_prob=model_prob,
                market_prob=mp.fair_prob,
                divergence=divergence,
                abs_divergence=abs(divergence),
                assessment=_classify(divergence, threshold),
            )
        )

    results.sort(key=lambda r: r.abs_divergence, reverse=True)
    return results
