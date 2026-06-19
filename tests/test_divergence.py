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


def test_rps_esempi_noti() -> None:
    from app.analysis.metrics import ranked_probability_score as rps
    assert abs(rps([0, 1, 0], 0) - 0.5) < 1e-9      # pareggio quando vince casa
    assert abs(rps([0, 0, 1], 0) - 1.0) < 1e-9      # ospite quando vince casa
    assert rps([0.6, 0.3, 0.1], 0) < rps([0, 1, 0], 0)


def test_dixon_coles_aumenta_pareggi() -> None:
    import numpy as np
    from app.core.dixon_coles import score_matrix
    base = score_matrix(1.4, 1.1, 0.0)
    corrected = score_matrix(1.4, 1.1, -0.1)
    assert np.trace(corrected) > np.trace(base)       # rho<0 -> piu' pareggi
    assert abs(corrected.sum() - 1.0) < 1e-9


def test_shrinkage_tira_verso_la_media() -> None:
    from datetime import date, timedelta
    from app.data.loader import load_results_from_text
    from app.core.ratings import estimate_ratings
    rows = ["date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"]
    d = date(2022, 1, 1)
    for _ in range(300):
        d += timedelta(days=1)
        rows.append(f"{d.isoformat()},A,B,1,1,Friendly,X,X,TRUE")
    # 'Padder': poche partite con goleade -> attacco gonfiato senza shrinkage
    for _ in range(3):
        d += timedelta(days=1)
        rows.append(f"{d.isoformat()},Padder,B,6,0,Friendly,X,X,TRUE")
    m = load_results_from_text("\n".join(rows), since_year=2021)
    a0 = {t.name: t for t in estimate_ratings(m, shrinkage=0.0).teams}["Padder"].attack
    a1 = {t.name: t for t in estimate_ratings(m, shrinkage=8.0).teams}["Padder"].attack
    assert a1 < a0  # lo shrinkage abbassa il rating della squadra con pochi dati


def test_elo_ordina_forte_debole() -> None:
    from datetime import date, timedelta
    from app.data.loader import load_results_from_text
    from app.core.elo import compute_elo
    import numpy as np, random
    true = {"Strong": 1.7, "Mid": 1.0, "Weak": 0.6}
    teams = list(true); rng = np.random.default_rng(0)
    rows = ["date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"]
    d = date(2021, 1, 1)
    for _ in range(900):
        h, a = random.sample(teams, 2)
        lam = 1.35 * true[h] / true[a]; mu = 1.35 * true[a] / true[h]
        d += timedelta(days=1)
        rows.append(f"{d.isoformat()},{h},{a},{int(rng.poisson(lam))},{int(rng.poisson(mu))},Friendly,X,X,TRUE")
    elo = compute_elo(load_results_from_text("\n".join(rows), since_year=2020))
    assert elo["Strong"] > elo["Mid"] > elo["Weak"]


def test_reliability_curve_ece() -> None:
    import random
    from app.analysis.metrics import reliability_curve
    random.seed(0)
    preds, outs = [], []
    for p10 in range(11):
        p = p10 / 10
        for _ in range(200):
            preds.append(p); outs.append(1 if random.random() < p else 0)
    _, _, _, ece = reliability_curve(preds, outs, n_bins=10)
    assert ece < 0.05  # modello ben calibrato -> ECE basso
    _, _, _, ece_bad = reliability_curve([0.5] * 100, [1] * 100, n_bins=10)
    assert ece_bad > 0.4  # sempre 0.5 ma evento sempre 1 -> scalibrato


def test_reliability_curve_ece() -> None:
    import random
    from app.analysis.metrics import reliability_curve
    random.seed(0)
    preds, outs = [], []
    for p10 in range(11):
        p = p10 / 10
        for _ in range(200):
            preds.append(p); outs.append(1 if random.random() < p else 0)
    _, _, _, ece = reliability_curve(preds, outs, n_bins=10)
    assert ece < 0.05  # modello ben calibrato -> ECE basso
    _, _, _, ece_bad = reliability_curve([0.5] * 100, [1] * 100, n_bins=10)
    assert ece_bad > 0.4  # sempre 0.5 ma evento sempre 1 -> scalibrato
