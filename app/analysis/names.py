"""
names.py
========
Normalizzazione dei nomi delle squadre per fare il "join" tra due fonti che
usano convenzioni diverse (il nostro modello vs. Polymarket).

Esempi di disallineamento tipici:
    "USA"  vs "United States"
    "South Korea" vs "Korea Republic"
    "Curaçao" vs "Curacao"  (accenti)

Strategia: portiamo ogni nome a una forma canonica minuscola, senza accenti e
senza spazi superflui, applicando prima una tabella di alias noti.
"""

from __future__ import annotations

import unicodedata

# Mappa alias -> nome canonico. Le chiavi sono gia' in minuscolo.
# Estendibile man mano che incontriamo nuovi disallineamenti sui dati reali.
_ALIASES: dict[str, str] = {
    "usa": "united states",
    "us": "united states",
    "korea republic": "south korea",
    "korea": "south korea",
    "ir iran": "iran",
    "china pr": "china",
    "czechia": "czech republic",
    "turkiye": "turkey",
}


def normalize_team(name: str) -> str:
    """
    Riduce un nome squadra alla sua forma canonica per il confronto.

    Passi: trim -> minuscolo -> rimozione accenti -> applicazione alias.

    Args:
        name: nome grezzo della squadra.

    Returns:
        Chiave canonica usata per il match (es. "united states").
    """
    # Trim + minuscolo.
    cleaned = name.strip().lower()

    # Rimozione accenti: "curaçao" -> "curacao".
    cleaned = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", cleaned)
        if not unicodedata.combining(ch)
    )

    # Applica alias se presente, altrimenti tieni il nome normalizzato.
    return _ALIASES.get(cleaned, cleaned)
