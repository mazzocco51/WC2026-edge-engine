"""
run_real_demo.py
================
Pipeline con DATI REALI (CC0): scarica i risultati storici, stima attacco/difesa
di ogni nazionale e mostra i rating + un esempio di gol attesi.

    python -m app.core.run_real_demo

FONTE: Mart Jürisoo (martj42), licenza CC0-1.0. https://github.com/martj42/international_results
DISCLAIMER: progetto a fini educativi/dimostrativi, NON commerciale.
"""

from __future__ import annotations

from app.core.poisson import expected_goals
from app.core.ratings import estimate_ratings
from app.data.loader import load_results


def main() -> None:
    print("Scarico i risultati storici (CC0, martj42)...")
    matches = load_results(since_year=2018)
    print(f"Partite caricate: {len(matches)}")

    res = estimate_ratings(matches)
    print(f"Squadre valutate: {len(res.teams)} | gol base: {res.base_goals:.2f} "
          f"| vantaggio campo: {res.home_advantage:.2f}\n")

    print(f"{'Top 15 per attacco':<22}{'Attacco':>9}{'Difesa':>9}")
    print("-" * 40)
    for t in res.teams[:15]:
        print(f"{t.name:<22}{t.attack:>9.2f}{t.defense:>9.2f}")

    # Esempio: gol attesi nello scontro tra le due squadre piu' forti (campo neutro).
    a, b = res.teams[0], res.teams[1]
    lam = expected_goals(a, b, avg=res.base_goals)
    print(f"\nEsempio gol attesi — {a.name} vs {b.name}: {lam:.2f}")


if __name__ == "__main__":
    main()
