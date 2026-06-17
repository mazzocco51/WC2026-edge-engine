"""
test_simulator.py
=================
Test minimi (pytest) del core matematico. Eseguire con:

    pytest -q
"""

from __future__ import annotations

import numpy as np

from app.core.poisson import expected_goals
from app.core.simulator import run_tournament, simulate_match
from app.data.dummy import DUMMY_TEAMS


def test_expected_goals_positivo() -> None:
    """Lambda deve essere sempre positivo."""
    a, b = DUMMY_TEAMS[0], DUMMY_TEAMS[3]
    assert expected_goals(a, b) > 0


def test_knockout_ha_sempre_un_vincitore() -> None:
    """In modalita' eliminazione un pareggio non e' ammesso come esito finale."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        res = simulate_match(DUMMY_TEAMS[1], DUMMY_TEAMS[2], rng, knockout=True)
        assert res.winner is not None


def test_probabilita_sommano_a_uno() -> None:
    """Le probabilita' del torneo devono sommare a ~1.0."""
    probs = run_tournament(DUMMY_TEAMS, n_simulations=2_000, seed=1)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_favorito_vince_piu_spesso() -> None:
    """Il Brazil (forte) deve avere probabilita' maggiore del Canada (debole)."""
    probs = run_tournament(DUMMY_TEAMS, n_simulations=5_000, seed=7)
    assert probs["Brazil"] > probs["Canada"]
