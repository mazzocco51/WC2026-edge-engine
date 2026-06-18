"""
run_wc_demo.py
==============
Demo END-TO-END con dati REALI: scarica i risultati CC0, stima i rating,
costruisce i 12 gironi e simula il Mondiale 2026 a 48 squadre.

    python -m app.core.run_wc_demo

FONTE: martj42 (CC0). Progetto educativo/dimostrativo, non commerciale.
"""

from __future__ import annotations

from app.core.engine import load_field
from app.core.tournament import simulate_world_cup


def main() -> None:
    print("Scarico i dati CC0 e stimo i rating...")
    field = load_field()
    print(f"Gironi costruiti: {len(field.groups)} | squadre: "
          f"{sum(len(g) for g in field.groups.values())}\n")

    for name, teams in field.groups.items():
        print(f"  Girone {name}: " + ", ".join(t.name for t in teams))

    print("\nSimulo il torneo (5.000 run)...")
    probs = simulate_world_cup(field.groups, field.ratings.base_goals, n_simulations=5_000)

    print(f"\n{'Top 15 - probabilita vittoria':<26}{'%':>7}")
    print("-" * 33)
    for name, p in list(probs.items())[:15]:
        print(f"{name:<26}{p * 100:>6.1f}%")
    print(f"\nTotale: {sum(probs.values()) * 100:.1f}%")


if __name__ == "__main__":
    main()
