"""
run_demo.py
===========
Script di prova manuale dello Step 1. Esegui dalla root del progetto:

    python -m app.core.run_demo

Stampa le probabilita' di vittoria del torneo dummy a 4 squadre.
"""

from __future__ import annotations

from app.core.simulator import run_tournament
from app.data.dummy import DUMMY_TEAMS


def main() -> None:
    probs = run_tournament(DUMMY_TEAMS, n_simulations=10_000, seed=42)

    print("\n=== Probabilita' vittoria Mondiale (dataset dummy, 10.000 sim) ===\n")
    print(f"{'Squadra':<12} {'Vittoria %':>10}")
    print("-" * 24)
    for name, p in probs.items():
        print(f"{name:<12} {p * 100:>9.1f}%")

    total = sum(probs.values())
    print("-" * 24)
    print(f"{'TOTALE':<12} {total * 100:>9.1f}%  (deve essere ~100%)\n")


if __name__ == "__main__":
    main()
