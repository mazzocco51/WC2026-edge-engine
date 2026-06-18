"""
run_demo.py
===========
Prova manuale del Market Engine. Eseguire dalla root del progetto:

    python -m app.market.run_demo

Stampa le probabilita' implicite di Polymarket per il vincitore del Mondiale,
sia grezze (col vig) sia "eque" (normalizzate a 100%).
"""

from __future__ import annotations

import asyncio

from app.market.polymarket import get_market_probabilities


async def main() -> None:
    markets = await get_market_probabilities()

    print("\n=== Polymarket - World Cup Winner (probabilita' implicite) ===\n")
    print(f"{'Squadra':<18}{'Grezza %':>10}{'Equa %':>10}")
    print("-" * 38)
    for mp in markets[:15]:  # top 15
        print(f"{mp.team:<18}{mp.raw_price * 100:>9.1f}%{mp.fair_prob * 100:>9.1f}%")

    raw_sum = sum(mp.raw_price for mp in markets)
    fair_sum = sum(mp.fair_prob for mp in markets)
    print("-" * 38)
    print(f"{'SOMMA':<18}{raw_sum * 100:>9.1f}%{fair_sum * 100:>9.1f}%")
    print(f"\nOverround (vig) di mercato: {(raw_sum - 1) * 100:+.1f}%\n")


if __name__ == "__main__":
    asyncio.run(main())
