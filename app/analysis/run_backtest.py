"""
run_backtest.py
===============
Backtest con DATI REALI (CC0):
1. Poisson puro vs Dixon-Coles (RPS, walk-forward);
2. taratura dell'emivita del time-decay;
3. taratura dello shrinkage;
4. effetto del prior Elo (on/off).

    python -m app.analysis.run_backtest

FONTE: martj42 (CC0). Progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

from datetime import date

from app.analysis.backtest import backtest_rps, tune_half_life, tune_shrinkage
from app.data.loader import load_results


def main() -> None:
    print("Scarico i risultati storici (CC0)...")
    matches = load_results(since_year=2010)
    print(f"Partite caricate: {len(matches)}")
    split = date(2024, 1, 1)

    r = backtest_rps(matches, split=split, test_window_days=500)
    print(f"\n[1] Poisson vs Dixon-Coles (train < {split}, test ~500 giorni)")
    print(f"    match di test : {r.n_test_matches}")
    print(f"    rho stimato   : {r.rho:+.3f}")
    print(f"    RPS Poisson   : {r.rps_poisson:.4f}")
    print(f"    RPS Dixon-Coles: {r.rps_dixon_coles:.4f}  ({r.improvement_pct:+.2f}%)")

    print("\n[2] Taratura emivita (RPS piu' basso = meglio)")
    t = tune_half_life(matches, split=split, test_window_days=500)
    for hl, rps in t.results:
        flag = "  <- migliore" if hl == t.best_half_life_days else ""
        print(f"    {hl / 365:>4.1f} anni -> RPS {rps:.4f}{flag}")

    print("\n[3] Taratura shrinkage")
    sh = tune_shrinkage(matches, split=split, test_window_days=500)
    for s, rps in sh.results:
        flag = "  <- migliore" if s == sh.best_shrinkage else ""
        print(f"    shrinkage {s:>4.1f} -> RPS {rps:.4f}{flag}")

    print("\n[4] Effetto del prior Elo")
    off = backtest_rps(matches, split=split, test_window_days=500, use_elo_prior=False)
    on = backtest_rps(matches, split=split, test_window_days=500, use_elo_prior=True)
    print(f"    senza prior Elo : RPS {off.rps_dixon_coles:.4f}")
    print(f"    con prior Elo   : RPS {on.rps_dixon_coles:.4f}")
    delta = (off.rps_dixon_coles - on.rps_dixon_coles) / off.rps_dixon_coles * 100
    print(f"    variazione      : {delta:+.2f}%  (positivo = l'Elo migliora)")


if __name__ == "__main__":
    main()
