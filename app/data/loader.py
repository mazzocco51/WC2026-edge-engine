"""
loader.py
=========
Caricamento dei risultati storici delle nazionali.

==================== FONTE DATI (UNICA) ====================
Dataset:  "International football results from 1872 to present"
Autore:   Mart Jürisoo (martj42)
Licenza:  CC0-1.0 (Public Domain Dedication) -> uso libero, anche commerciale,
          senza obbligo di attribuzione. Vedi DATA_SOURCES.md.
Origine:  https://github.com/martj42/international_results
Raw CSV:  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
===========================================================

DISCLAIMER: progetto a fini ESCLUSIVAMENTE EDUCATIVI e DIMOSTRATIVI.
NON commerciale. Nessun consiglio di scommessa.

Nota: il download avviene a RUNTIME dalla fonte originale (non ridistribuiamo
i dati nel repository), usando solo la libreria standard di Python (urllib),
quindi nessuna dipendenza aggiuntiva e nessuna API key.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

# URL del CSV grezzo (CC0). Centralizzato per essere facile da cambiare/mockare.
RESULTS_CSV_URL: str = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)


@dataclass(frozen=True)
class Match:
    """Una partita internazionale (riga del dataset martj42)."""

    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    neutral: bool  # True se campo neutro (niente vantaggio casalingo)


def _parse_rows(rows: list[dict[str, str]], since: date) -> list[Match]:
    """Converte le righe CSV grezze in oggetti Match, filtrando per data."""
    matches: list[Match] = []
    for row in rows:
        # Salta righe senza punteggio (partite future/annullate).
        if not row.get("home_score") or not row.get("away_score"):
            continue
        try:
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if d < since:
            continue
        matches.append(
            Match(
                match_date=d,
                home_team=row["home_team"].strip(),
                away_team=row["away_team"].strip(),
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                neutral=row.get("neutral", "FALSE").strip().upper() == "TRUE",
            )
        )
    return matches


def load_results(
    since_year: int = 2018,
    url: str = RESULTS_CSV_URL,
    timeout: float = 30.0,
) -> list[Match]:
    """
    Scarica e filtra i risultati dalla fonte CC0 (martj42).

    Args:
        since_year: tiene solo le partite da questo anno in poi (la "forma"
                    recente conta di piu'; il time-decay raffina ulteriormente).
        url: URL del CSV grezzo (override utile nei test).
        timeout: timeout di rete in secondi.

    Returns:
        Lista di Match ordinata per data.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (URL fidato, CC0)
        text = resp.read().decode("utf-8")
    return load_results_from_text(text, since_year=since_year)


def load_results_from_text(text: str, since_year: int = 2018) -> list[Match]:
    """
    Variante che parsa un CSV gia' in memoria (per test offline / cache locale).

    Separata dal download cosi' la logica di parsing e' testabile senza rete.
    """
    reader = csv.DictReader(io.StringIO(text))
    matches = _parse_rows(list(reader), since=date(since_year, 1, 1))
    matches.sort(key=lambda m: m.match_date)
    return matches
