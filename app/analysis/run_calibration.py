"""
run_calibration.py
==================
Genera il RELIABILITY DIAGRAM (grafico di calibrazione) del modello sui dati
reali CC0: confronta Poisson puro vs Dixon-Coles contro la diagonale di
calibrazione perfetta, e stampa l'ECE (Expected Calibration Error).

    python -m app.analysis.run_calibration

Salva 'calibration_diagram.png' nella cartella del progetto.

Nota: confrontiamo la calibrazione del MODELLO contro la realta'. Non
includiamo il mercato perche' non disponiamo di quote storiche per-partita
(solo il market live del vincitore). FONTE dati: martj42 (CC0).
"""

from __future__ import annotations

from datetime import date

import matplotlib
matplotlib.use("Agg")  # nessun display: salva su file
import matplotlib.pyplot as plt

from app.analysis.backtest import reliability_data
from app.analysis.metrics import reliability_curve
from app.data.loader import load_results


def main() -> None:
    print("Scarico i risultati storici (CC0)...")
    matches = load_results(since_year=2010)
    data = reliability_data(matches, split=date(2024, 1, 1), test_window_days=500)

    pm_p, of_p, _, ece_p = reliability_curve(data["poisson_pred"], data["poisson_out"])
    pm_d, of_d, _, ece_d = reliability_curve(data["dc_pred"], data["dc_out"])

    print(f"ECE Poisson    : {ece_p:.4f}")
    print(f"ECE Dixon-Coles: {ece_d:.4f}  (piu' basso = meglio calibrato)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibrazione perfetta")
    ax.plot(pm_p, of_p, "o-", color="#888780", label=f"Poisson (ECE {ece_p:.3f})")
    ax.plot(pm_d, of_d, "o-", color="#378ADD", label=f"Dixon-Coles (ECE {ece_d:.3f})")
    ax.set_xlabel("Probabilità prevista")
    ax.set_ylabel("Frequenza reale")
    ax.set_title("Reliability diagram — WC2026 model (1X2, backtest)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    out = "calibration_diagram.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
