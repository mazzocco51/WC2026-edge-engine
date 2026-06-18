"""
run_demo.py
===========
Prova manuale della pipeline completa Model + Market -> Edge. Eseguire:

    python -m app.edge.run_demo

Esegue il Monte Carlo sul dataset dummy, scarica il mercato live da Polymarket
e stampa la tabella di confronto con i segnali +EV.
"""

from __future__ import annotations

import asyncio

from app.core.simulator import run_tournament
from app.data.dummy import DUMMY_TEAMS
from app.edge.calculator import compute_edges
from app.market.polymarket import get_market_probabilities


async def main() -> None:
    # 1. Probabilita' del nostro modello (dummy a 4 squadre per ora).
    model_probs = run_tournament(DUMMY_TEAMS, n_simulations=10_000, seed=42)

    # 2. Probabilita' implicite di Polymarket (live).
    market_probs = await get_market_probabilities()

    # 3. Confronto.
    edges = compute_edges(model_probs, market_probs, threshold=0.02)

    print("\n=== Edge: Modello vs Polymarket ===\n")
    header = f"{'Squadra':<14}{'Model %':>9}{'Market %':>10}{'Edge':>8}{'EV ROI':>9}  Segnale"
    print(header)
    print("-" * len(header))
    for r in edges:
        print(
            f"{r.team:<14}"
            f"{r.model_prob * 100:>8.1f}%"
            f"{r.market_fair * 100:>9.1f}%"
            f"{r.edge * 100:>+7.1f}"
            f"{r.ev_roi * 100:>+8.1f}%"
            f"  {r.signal.value}"
        )

    if not edges:
        print("(Nessuna squadra in comune tra modello e mercato.)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
