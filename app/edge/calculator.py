"""
calculator.py
=============
Edge Calculator: il cuore "quant" del progetto.

Confronta la probabilita' del NOSTRO modello Monte Carlo con quella implicita
del mercato Polymarket e quantifica se esiste un valore atteso positivo (+EV).

Due metriche, due significati
-----------------------------
1. EDGE (mispricing, vig-neutral):
       edge = model_prob - market_fair_prob
   Misura di quanto il nostro modello "non e' d'accordo" col consenso del
   mercato, a parita' di condizioni (entrambe le prob sommano a 100%). Utile
   per capire DOVE il modello vede le cose diversamente.

2. EV (valore atteso reale della scommessa):
       ev      = model_prob - market_raw_price
       ev_roi  = ev / market_raw_price
   Su Polymarket compri una quota "Yes" al prezzo `market_raw_price` (in $, tra
   0 e 1) e incassi $1 se la squadra vince. Se la nostra probabilita' `q` supera
   il prezzo pagato `p`, il valore atteso per quota e' (q - p) > 0: scommessa
   +EV. `ev_roi` esprime lo stesso valore come rendimento percentuale sullo
   stake. Usiamo il prezzo GREZZO (col vig) perche' e' quello che paghi davvero.

Il segnale operativo si basa su EV (il denaro vero), non sull'edge teorico.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.edge.names import normalize_team
from app.market.polymarket import MarketProbability


class Signal(str, Enum):
    """Esito sintetico del confronto per una squadra."""

    VALUE = "+EV"      # il modello vede valore: scommessa potenzialmente profittevole
    FAIR = "FAIR"      # modello e mercato sostanzialmente d'accordo
    OVERPRICED = "-EV"  # il mercato prezza la squadra piu' di quanto valga per noi


@dataclass(frozen=True)
class EdgeResult:
    """
    Riga di confronto modello-vs-mercato per una squadra.

    Attributi:
        team:            nome (etichetta del nostro modello).
        model_prob:      probabilita' di vittoria dal Monte Carlo [0, 1].
        market_raw:      prezzo "Yes" grezzo Polymarket (col vig) [0, 1].
        market_fair:     probabilita' di mercato dopo de-vig [0, 1].
        edge:            model_prob - market_fair (mispricing).
        ev:              model_prob - market_raw (valore atteso per quota).
        ev_roi:          ev / market_raw (rendimento atteso sullo stake).
        signal:          Signal (+EV / FAIR / -EV).
    """

    team: str
    model_prob: float
    market_raw: float
    market_fair: float
    edge: float
    ev: float
    ev_roi: float
    signal: Signal


def _classify(ev: float, threshold: float) -> Signal:
    """Mappa l'EV su un segnale, data una soglia di tolleranza (in prob)."""
    if ev > threshold:
        return Signal.VALUE
    if ev < -threshold:
        return Signal.OVERPRICED
    return Signal.FAIR


def compute_edges(
    model_probs: dict[str, float],
    market_probs: list[MarketProbability],
    threshold: float = 0.02,
) -> list[EdgeResult]:
    """
    Incrocia probabilita' del modello e del mercato e calcola edge/EV.

    Il join avviene sui nomi normalizzati (vedi names.py). Vengono incluse solo
    le squadre presenti in ENTRAMBE le fonti.

    Args:
        model_probs:  {nome_squadra: probabilita'} dal Monte Carlo.
        market_probs: lista di MarketProbability da Polymarket.
        threshold:    soglia (in punti di probabilita') oltre la quale l'EV
                      diventa un segnale. Default 0.02 = 2 punti percentuali.

    Returns:
        Lista di EdgeResult ordinata per EV decrescente (le occasioni migliori
        in cima).
    """
    # Indicizza il mercato per nome canonico, per un lookup O(1).
    market_by_key: dict[str, MarketProbability] = {
        normalize_team(mp.team): mp for mp in market_probs
    }

    results: list[EdgeResult] = []
    for team, model_prob in model_probs.items():
        key = normalize_team(team)
        mp = market_by_key.get(key)
        if mp is None:
            continue  # nessuna corrispondenza sul mercato: salta

        edge = model_prob - mp.fair_prob
        ev = model_prob - mp.raw_price
        ev_roi = ev / mp.raw_price if mp.raw_price > 0 else 0.0

        results.append(
            EdgeResult(
                team=team,
                model_prob=model_prob,
                market_raw=mp.raw_price,
                market_fair=mp.fair_prob,
                edge=edge,
                ev=ev,
                ev_roi=ev_roi,
                signal=_classify(ev, threshold),
            )
        )

    results.sort(key=lambda r: r.ev, reverse=True)
    return results
