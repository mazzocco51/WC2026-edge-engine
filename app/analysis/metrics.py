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


def ranked_probability_score(probs: list[float], outcome_index: int) -> float:
    """
    Ranked Probability Score (RPS) per esiti ORDINATI (es. 1X2: casa/pareggio/ospite).

    A differenza del Brier, tiene conto dell'ordine: un errore "vicino" (prevedere
    pareggio quando vince la casa) e' penalizzato meno di uno "lontano" (prevedere
    la vittoria ospite). Piu' basso = meglio (0 = perfetto, 1 = pessimo).

    Args:
        probs: probabilita' degli esiti in ORDINE (devono sommare ~1).
        outcome_index: indice dell'esito realmente avvenuto (0-based).

    Returns:
        RPS in [0, 1].
    """
    r = len(probs)
    cum_p = 0.0
    cum_e = 0.0
    total = 0.0
    for i in range(r - 1):
        cum_p += probs[i]
        cum_e += 1.0 if i == outcome_index else 0.0
        total += (cum_p - cum_e) ** 2
    return total / (r - 1)


def reliability_curve(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int], float]:
    """
    Curva di calibrazione (reliability diagram) per previsioni binarie.

    Raggruppa le probabilita' previste in `n_bins` intervalli; per ciascun bin
    calcola la probabilita' media prevista e la frequenza reale dell'evento.
    Un modello ben calibrato ha (previsto ~ reale) -> punti sulla diagonale.

    Args:
        predictions: probabilita' previste in [0,1] (es. P(vittoria casa)).
        outcomes: esiti reali 0/1 allineati alle previsioni.
        n_bins: numero di intervalli di probabilita'.

    Returns:
        (prob_media_prevista, freq_reale, conteggio_per_bin, ece)
        dove `ece` (Expected Calibration Error) e' lo scostamento medio assoluto
        previsto-vs-reale pesato per la numerosita' dei bin (piu' basso = meglio).
    """
    if len(predictions) != len(outcomes):
        raise ValueError("predictions e outcomes devono avere la stessa lunghezza.")

    edges = [i / n_bins for i in range(n_bins + 1)]
    pred_means: list[float] = []
    obs_freqs: list[float] = []
    counts: list[int] = []
    ece = 0.0
    total = len(predictions) or 1

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # L'ultimo bin include l'estremo destro (prob = 1.0).
        idxs = [
            i for i, p in enumerate(predictions)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)
        ]
        if not idxs:
            continue
        pm = sum(predictions[i] for i in idxs) / len(idxs)
        of = sum(outcomes[i] for i in idxs) / len(idxs)
        pred_means.append(pm)
        obs_freqs.append(of)
        counts.append(len(idxs))
        ece += (len(idxs) / total) * abs(pm - of)

    return pred_means, obs_freqs, counts, ece
