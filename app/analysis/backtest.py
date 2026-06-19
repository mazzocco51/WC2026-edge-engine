"""
backtest.py
===========
Backtest "walk-forward" che misura l'accuratezza delle previsioni 1X2 del
modello su partite PASSATE, confrontando il Poisson puro con la correzione
Dixon-Coles tramite il Ranked Probability Score (RPS).

Protocollo corretto (no data leakage)
-------------------------------------
1. Si allena il modello SOLO sulle partite prima di una data di split.
2. Si prevedono le partite del periodo di test (dopo lo split).
3. Si calcola l'RPS medio per ciascun modello.

Confrontare l'RPS Poisson vs Dixon-Coles mostra se la correzione migliora
davvero la calibrazione. (Si puo' poi confrontare anche col mercato.)

DISCLAIMER: progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from app.analysis.metrics import ranked_probability_score
from app.core.dixon_coles import outcome_probs
from app.core.elo import compute_elo, elo_strength_multipliers
from app.core.ratings import estimate_ratings
from app.data.loader import Match


@dataclass(frozen=True)
class BacktestResult:
    """Esito del backtest: RPS medio dei due modelli sul periodo di test."""

    rps_poisson: float
    rps_dixon_coles: float
    n_test_matches: int
    rho: float

    @property
    def improvement_pct(self) -> float:
        """Miglioramento percentuale di DC sul Poisson (positivo = meglio)."""
        if self.rps_poisson == 0:
            return 0.0
        return (self.rps_poisson - self.rps_dixon_coles) / self.rps_poisson * 100.0


def _outcome_index(home_score: int, away_score: int) -> int:
    """0 = vittoria casa, 1 = pareggio, 2 = vittoria ospite."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def backtest_rps(
    matches: list[Match],
    split: date,
    test_window_days: int = 400,
    half_life_days: float = 365 * 4,
    shrinkage: float = 4.0,
    use_elo_prior: bool = True,
) -> BacktestResult:
    """
    Allena prima di `split`, testa nei `test_window_days` successivi.

    Returns:
        BacktestResult con l'RPS medio di Poisson e Dixon-Coles.
    """
    train = [m for m in matches if m.match_date < split]
    end = split + timedelta(days=test_window_days)
    test = [m for m in matches if split <= m.match_date < end]
    if not train or not test:
        raise ValueError("Train o test vuoto: scegli una data di split diversa.")

    prior = elo_strength_multipliers(compute_elo(train)) if use_elo_prior else None
    res = estimate_ratings(train, half_life_days=half_life_days,
                          shrinkage=shrinkage, strength_prior=prior)
    rating = {t.name: t for t in res.teams}
    base, gamma, rho = res.base_goals, res.home_advantage, res.rho

    sum_pois = 0.0
    sum_dc = 0.0
    n = 0
    for m in test:
        a = rating.get(m.home_team)
        b = rating.get(m.away_team)
        if a is None or b is None:
            continue  # squadra mai vista in training
        hf = 1.0 if m.neutral else gamma
        lam = base * a.attack * b.defense * hf
        mu = base * b.attack * a.defense
        oi = _outcome_index(m.home_score, m.away_score)
        sum_pois += ranked_probability_score(list(outcome_probs(lam, mu, 0.0)), oi)
        sum_dc += ranked_probability_score(list(outcome_probs(lam, mu, rho)), oi)
        n += 1

    if n == 0:
        raise ValueError("Nessun match di test con entrambe le squadre note.")
    return BacktestResult(
        rps_poisson=sum_pois / n,
        rps_dixon_coles=sum_dc / n,
        n_test_matches=n,
        rho=rho,
    )


@dataclass(frozen=True)
class TuningResult:
    """Esito della taratura dell'emivita del time-decay."""

    results: list[tuple[float, float]]   # (emivita_giorni, RPS Dixon-Coles)
    best_half_life_days: float
    best_rps: float


def tune_half_life(
    matches: list[Match],
    split: date,
    candidates_days: tuple[float, ...] = (180.0, 365.0, 730.0, 1095.0, 1460.0, 1825.0),
    test_window_days: int = 500,
) -> TuningResult:
    """
    Tara l'emivita del time-decay scegliendo quella col miglior RPS out-of-sample.

    Per ogni emivita candidata esegue lo stesso backtest walk-forward e misura
    l'RPS (Dixon-Coles) sul periodo di test. Vince l'emivita con RPS minimo:
    e' una scelta di iperparametro guidata dai DATI, non a occhio.

    Args:
        matches: storico partite.
        split: data di separazione train/test.
        candidates_days: emivite da provare (in giorni).
        test_window_days: ampiezza della finestra di test.

    Returns:
        TuningResult con la tabella (emivita, RPS) e la migliore.
    """
    results: list[tuple[float, float]] = []
    for hl in candidates_days:
        r = backtest_rps(matches, split, test_window_days, half_life_days=hl)
        results.append((hl, r.rps_dixon_coles))
    best_hl, best_rps = min(results, key=lambda x: x[1])
    return TuningResult(results=results, best_half_life_days=best_hl, best_rps=best_rps)


@dataclass(frozen=True)
class ShrinkageTuning:
    """Esito della taratura dello shrinkage."""

    results: list[tuple[float, float]]   # (shrinkage, RPS Dixon-Coles)
    best_shrinkage: float
    best_rps: float


def tune_shrinkage(
    matches: list[Match],
    split: date,
    candidates: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 16.0),
    test_window_days: int = 500,
    half_life_days: float = 365 * 4,
) -> ShrinkageTuning:
    """
    Tara lo shrinkage scegliendo quello col miglior RPS out-of-sample.

    `shrinkage = 0` equivale a nessuna regolarizzazione: il confronto mostra se
    e quanto la regolarizzazione migliora le previsioni.
    """
    results: list[tuple[float, float]] = []
    for s in candidates:
        r = backtest_rps(matches, split, test_window_days,
                         half_life_days=half_life_days, shrinkage=s)
        results.append((s, r.rps_dixon_coles))
    best_s, best_rps = min(results, key=lambda x: x[1])
    return ShrinkageTuning(results=results, best_shrinkage=best_s, best_rps=best_rps)


def reliability_data(
    matches: list[Match],
    split: date,
    test_window_days: int = 500,
    half_life_days: float = 365 * 4,
) -> dict[str, list[float]]:
    """
    Raccoglie coppie (probabilita' prevista, esito 0/1) sul periodo di test,
    mettendo in pool i tre esiti 1X2, per Poisson puro e Dixon-Coles.

    Serve a costruire il reliability diagram (calibrazione). Ritorna un dict con
    'poisson_pred', 'poisson_out', 'dc_pred', 'dc_out'.
    """
    train = [m for m in matches if m.match_date < split]
    end = split + timedelta(days=test_window_days)
    test = [m for m in matches if split <= m.match_date < end]
    if not train or not test:
        raise ValueError("Train o test vuoto.")

    res = estimate_ratings(train, half_life_days=half_life_days)
    rating = {t.name: t for t in res.teams}
    base, gamma, rho = res.base_goals, res.home_advantage, res.rho

    out: dict[str, list[float]] = {
        "poisson_pred": [], "poisson_out": [], "dc_pred": [], "dc_out": []
    }
    for m in test:
        a = rating.get(m.home_team)
        b = rating.get(m.away_team)
        if a is None or b is None:
            continue
        hf = 1.0 if m.neutral else gamma
        lam = base * a.attack * b.defense * hf
        mu = base * b.attack * a.defense
        oi = _outcome_index(m.home_score, m.away_score)
        outcomes = [1.0 if oi == k else 0.0 for k in range(3)]
        for prob, o in zip(outcome_probs(lam, mu, 0.0), outcomes):
            out["poisson_pred"].append(prob); out["poisson_out"].append(o)
        for prob, o in zip(outcome_probs(lam, mu, rho), outcomes):
            out["dc_pred"].append(prob); out["dc_out"].append(o)
    return out
