"""
test_calculator.py
==================
Test dell'Edge Calculator e della normalizzazione dei nomi. Eseguire:

    pytest -q
"""

from __future__ import annotations

from app.edge.calculator import Signal, compute_edges
from app.edge.names import normalize_team
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


def test_ev_e_roi_corretti() -> None:
    model = {"Brazil": 0.30}
    r = compute_edges(model, _market())[0]
    assert abs(r.ev - 0.10) < 1e-9          # 0.30 - 0.20
    assert abs(r.ev_roi - 0.5) < 1e-9        # 0.10 / 0.20
    assert r.signal == Signal.VALUE


def test_segnale_overpriced() -> None:
    model = {"France": 0.20}
    r = compute_edges(model, _market())[0]
    assert r.signal == Signal.OVERPRICED


def test_join_via_alias() -> None:
    """USA nel modello deve agganciare 'United States' sul mercato."""
    model = {"USA": 0.11}
    res = compute_edges(model, _market())
    assert len(res) == 1 and res[0].team == "USA"


def test_ordinamento_per_ev() -> None:
    model = {"Brazil": 0.30, "France": 0.20}
    res = compute_edges(model, _market())
    assert res[0].team == "Brazil"  # EV piu' alto in cima
