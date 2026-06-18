"""
metrics.py
==========
Metriche per quantificare l'EFFICIENZA del mercato e la qualita' (calibrazione)
delle previsioni probabilistiche.

Due famiglie di metriche
------------------------
1) DIVERGENZA modello-vs-mercato (calcolabile SUBITO, non serve l'esito):
   - mean_absolute_divergence: scostamento medio tra le due distribuzioni.
   - kl_divergence: quanto la distribuzione del modello "sorprende" rispetto a
     quella del mercato (asimmetrica: KL(modello || mercato)).

2) CALIBRAZIONE delle previsioni (richiede l'ESITO reale -> per il backtest su
   tornei gia' conclusi):
   - brier_score: errore quadratico medio multi-classe (piu' basso = meglio).
   - log_loss: penalizza la sicurezza mal riposta (piu' basso = meglio).

Confrontando Brier/log-loss del modello vs quelli del mercato su eventi passati
si misura *chi prevede meglio* — il vero cuore di uno studio di efficienza.
"""

from __future__ import annotations

import math


def _common_normalized(
    model: dict[str, float], market: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    """Restringe alle squadre comuni e rinormalizza entrambe a somma 1."""
    keys = [k for k in model if k in market]
    sm = sum(model[k] for k in keys) or 1.0
    sk = sum(market[k] for k in keys) or 1.0
    p = {k: model[k] / sm for k in keys}
    q = {k: market[k] / sk for k in keys}
    return p, q


def mean_absolute_divergence(
    model: dict[str, float], market: dict[str, float]
) -> float:
    """Scostamento medio |p - q| sulle squadre comuni (0 = perfetta concordanza)."""
    p, q = _common_normalized(model, market)
    if not p:
        return 0.0
    return sum(abs(p[k] - q[k]) for k in p) / len(p)


def kl_divergence(model: dict[str, float], market: dict[str, float]) -> float:
    """
    Divergenza di Kullback-Leibler KL(modello || mercato), in nats.

    Misura quanta "informazione extra" porta il modello rispetto al mercato.
    0 = distribuzioni identiche; valori alti = forte disaccordo.
    """
    p, q = _common_normalized(model, market)
    total = 0.0
    for k in p:
        if p[k] > 0 and q[k] > 0:
            total += p[k] * math.log(p[k] / q[k])
    return total


def brier_score(probabilities: dict[str, float], winner: str) -> float:
    """
    Brier score multi-classe: somma degli errori quadratici sull'esito reale.

    Per ogni squadra: (probabilita' assegnata - esito{0/1})^2. Piu' basso = meglio.
    Da usare nel BACKTEST su tornei conclusi (serve il vincitore reale).

    Args:
        probabilities: {squadra: probabilita'} previste prima dell'evento.
        winner: nome della squadra effettivamente vincitrice.
    """
    total = 0.0
    for team, p in probabilities.items():
        outcome = 1.0 if team == winner else 0.0
        total += (p - outcome) ** 2
    return total


def log_loss(probabilities: dict[str, float], winner: str, eps: float = 1e-12) -> float:
    """
    Log-loss (cross-entropy) sull'esito reale: -log(prob assegnata al vincitore).

    Penalizza pesantemente l'essere sicuri e sbagliati. Piu' basso = meglio.
    """
    p = max(probabilities.get(winner, 0.0), eps)
    return -math.log(p)
