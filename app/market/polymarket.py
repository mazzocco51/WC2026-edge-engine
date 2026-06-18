"""
polymarket.py
=============
Market Engine: client ASINCRONO per il prediction market Polymarket.

Cosa fa
-------
1. Interroga la Gamma API pubblica (nessuna API key richiesta) per l'evento
   "World Cup Winner".
2. Estrae, per ogni squadra, il prezzo "Yes" = probabilita' implicita grezza.
3. Rimuove il "vig" (overround) normalizzando le probabilita' a somma 1.0,
   cosi' sono confrontabili 1:1 con le probabilita' del modello Monte Carlo.

Note di design
--------------
- httpx.AsyncClient: I/O di rete non bloccante, perfetto per il backend
  FastAPI che costruiremo nello Step 4.
- Gamma codifica `outcomes`/`outcomePrices` come STRINGHE JSON, quindi vanno
  ri-parsate con json.loads prima di indicizzarle.
- La Gamma API blocca a intermittenza le richieste senza User-Agent reale:
  ne impostiamo sempre uno.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

# Endpoint pubblico e slug dell'evento. Centralizzati come costanti cosi' sono
# facili da cambiare (es. mercati per girone) senza toccare la logica.
GAMMA_BASE_URL: str = "https://gamma-api.polymarket.com"
WORLD_CUP_SLUG: str = "world-cup-winner"
_HEADERS: dict[str, str] = {"User-Agent": "WC2026-EdgeEngine/1.0"}


@dataclass(frozen=True)
class MarketProbability:
    """
    Probabilita' di mercato per una singola squadra.

    Attributi:
        team:       nome della squadra (come riportato da Polymarket).
        raw_price:  prezzo "Yes" grezzo [0, 1] = probabilita' implicita col vig.
        fair_prob:  probabilita' "equa" dopo rimozione del vig (somma a 1.0).
    """

    team: str
    raw_price: float
    fair_prob: float


async def fetch_event(
    slug: str = WORLD_CUP_SLUG,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """
    Scarica l'oggetto evento grezzo dalla Gamma API.

    Args:
        slug: identificativo dell'evento Polymarket.
        client: AsyncClient opzionale da riusare (utile nei test e in FastAPI
                per non aprire una connessione nuova ad ogni richiesta).

    Returns:
        Il dizionario dell'evento (il primo della lista restituita dall'API).

    Raises:
        httpx.HTTPStatusError: se la risposta ha status >= 400.
        ValueError: se nessun evento corrisponde allo slug.
    """
    # Se non ci passano un client, ne creiamo uno usa-e-getta con timeout.
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=10.0, headers=_HEADERS)

    try:
        resp = await client.get(
            f"{GAMMA_BASE_URL}/events",
            params={"slug": slug},
            headers=_HEADERS,
        )
        resp.raise_for_status()
        events = resp.json()
    finally:
        # Chiudiamo solo il client che abbiamo creato noi.
        if owns_client:
            await client.aclose()

    if not events:
        raise ValueError(f"Nessun evento Polymarket trovato per slug='{slug}'.")
    return events[0]


def _parse_raw_prices(event: dict) -> dict[str, float]:
    """
    Estrae {squadra: prezzo_yes} dai market dell'evento.

    Ogni market e' un sotto-mercato Yes/No relativo a una squadra; il prezzo
    dell'esito "Yes" e' la probabilita' implicita (col vig) che quella squadra
    vinca il torneo. I market chiusi/inattivi vengono ignorati.
    """
    prices: dict[str, float] = {}

    for market in event.get("markets", []):
        # Salta i market non piu' negoziabili.
        if market.get("closed") or not market.get("active", True):
            continue

        team = market.get("groupItemTitle") or market.get("question")
        outcomes_raw = market.get("outcomes")
        prices_raw = market.get("outcomePrices")
        if not (team and outcomes_raw and prices_raw):
            continue

        # outcomes / outcomePrices sono stringhe JSON -> vanno deserializzate.
        outcomes: list[str] = json.loads(outcomes_raw)
        outcome_prices: list[str] = json.loads(prices_raw)

        # Troviamo l'indice dell'esito "Yes" (robusto a maiuscole/ordine).
        try:
            yes_idx = [o.lower() for o in outcomes].index("yes")
        except ValueError:
            yes_idx = 0  # fallback: primo esito

        prices[team] = float(outcome_prices[yes_idx])

    return prices


def remove_vig(raw_prices: dict[str, float]) -> dict[str, float]:
    """
    Rimuove l'overround (vig) normalizzando i prezzi a somma 1.0.

    La somma dei prezzi "Yes" su tutte le squadre e' > 1 (e' il margine del
    book). Dividendo ogni prezzo per la somma totale otteniamo probabilita'
    "eque" confrontabili col nostro modello.

    Metodo usato: normalizzazione proporzionale (la piu' semplice e diffusa).
    """
    total = sum(raw_prices.values())
    if total <= 0:
        return {team: 0.0 for team in raw_prices}
    return {team: price / total for team, price in raw_prices.items()}


async def get_market_probabilities(
    slug: str = WORLD_CUP_SLUG,
    client: httpx.AsyncClient | None = None,
) -> list[MarketProbability]:
    """
    Pipeline completa del Market Engine: fetch -> parse -> de-vig.

    Returns:
        Lista di MarketProbability ordinata per probabilita' equa decrescente.
    """
    event = await fetch_event(slug, client=client)
    raw = _parse_raw_prices(event)
    fair = remove_vig(raw)

    results = [
        MarketProbability(team=team, raw_price=raw[team], fair_prob=fair[team])
        for team in raw
    ]
    results.sort(key=lambda mp: mp.fair_prob, reverse=True)
    return results
