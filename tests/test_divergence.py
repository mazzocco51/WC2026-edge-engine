"""
test_divergence.py
==================
Test dell'analisi di efficienza (divergenza + metriche) e dei nomi. Eseguire:

    pytest -q
"""

from __future__ import annotations

from app.analysis.divergence import Assessment, compute_divergence
from app.analysis.metrics import (
    brier_score,
    kl_divergence,
    log_loss,
    mean_absolute_divergence,
)
from app.analysis.names import normalize_team
from app.market.polymarket import MarketProbability


def test_normalize_alias_e_accenti() -> None:
    assert normalize_team("USA") == normalize_team("United States")
    assert normalize_team("Curaçao") == "curacao"
    assert normalize_team(" South Korea ") == normalize_team("Korea Republic")


def _market() -> list[MarketProbability]:
    return [
        MarketProbability("Brazil", raw_price=0.20, fair_prob=0.18),
        MarketProbability("France", raw_price=0.30, fair_prob=0.28),
        MarketProbability("United States", raw_price=0.10, fair_prob=0.09),
    ]


def test_divergenza_segno_e_classificazione() -> None:
    model = {"Brazil": 0.30, "France": 0.20}
    by = {r.team: r for r in compute_divergence(model, _market())}
    assert abs(by["Brazil"].divergence - (0.30 - 0.18)) < 1e-9
    assert by["Brazil"].assessment == Assessment.MODEL_HIGHER
    assert by["France"].assessment == Assessment.MODEL_LOWER  # 0.20 < 0.28


def test_ordinamento_per_divergenza_assoluta() -> None:
    model = {"Brazil": 0.30, "France": 0.27}  # France quasi allineata
    res = compute_divergence(model, _market())
    assert res[0].team == "Brazil"  # divergenza assoluta maggiore in cima


def test_join_via_alias() -> None:
    res = compute_divergence({"USA": 0.11}, _market())
    assert len(res) == 1 and res[0].team == "USA"


def test_metriche_efficienza() -> None:
    model = {"A": 0.5, "B": 0.5}
    market = {"A": 0.5, "B": 0.5}
    assert mean_absolute_divergence(model, market) == 0.0
    assert abs(kl_divergence(model, market)) < 1e-12  # distribuzioni identiche


def test_brier_e_logloss() -> None:
    probs = {"A": 0.7, "B": 0.2, "C": 0.1}
    # Brier: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09+0.04+0.01 = 0.14
    assert abs(brier_score(probs, "A") - 0.14) < 1e-9
    # log-loss: -log(0.7)
    import math
    assert abs(log_loss(probs, "A") - (-math.log(0.7))) < 1e-9
