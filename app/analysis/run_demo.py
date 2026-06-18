"""
run_demo.py
===========
Prova manuale dell'analisi di efficienza: Modello + Market -> Divergenza + metriche.

    python -m app.analysis.run_demo

Esegue il Monte Carlo sul dataset dummy, scarica il mercato live da Polymarket
e stampa la tabella di confronto con la divergenza modello-vs-mercato.

Finalita' educativa/dimostrativa. Nessun consiglio di scommessa.
"""

from __future__ import annotations

import asyncio

from app.analysis.divergence import compute_divergence
from app.analysis.metrics import kl_divergence, mean_absolute_divergence
from app.core.simulator import run_tournament
from app.data.dummy import DUMMY_TEAMS
from app.market.polymarket import get_market_probabilities


async def main() -> None:
    model_probs = run_tournament(DUMMY_TEAMS, n_simulations=10_000, seed=42)
    market_probs = await get_market_probabilities()

    rows = compute_divergence(model_probs, market_probs, threshold=0.02)

    print("\n=== Efficienza di mercato: Modello vs Polymarket ===\n")
    header = f"{'Squadra':<14}{'Model %':>9}{'Market %':>10}{'Divergenza':>12}  Valutazione"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.team:<14}"
            f"{r.model_prob * 100:>8.1f}%"
            f"{r.market_prob * 100:>9.1f}%"
            f"{r.divergence * 100:>+11.1f}"
            f"  {r.assessment.value}"
        )

    if not rows:
        print("(Nessuna squadra in comune tra modello e mercato.)")
        return

    model_map = {r.team: r.model_prob for r in rows}
    market_map = {r.team: r.market_prob for r in rows}
    print(
        f"\nDivergenza media assoluta: {mean_absolute_divergence(model_map, market_map) * 100:.1f} pt"
        f" | KL(modello||mercato): {kl_divergence(model_map, market_map):.3f} nats\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
