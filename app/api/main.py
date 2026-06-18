"""
main.py
=======
Applicazione FastAPI asincrona: espone modello, mercato ed edge via HTTP.

Endpoint
--------
GET /health        -> stato del servizio
GET /predictions   -> probabilita' del modello Monte Carlo
GET /market        -> probabilita' implicite (de-vig) di Polymarket
GET /edge          -> confronto modello vs mercato con segnali +EV

Note di concorrenza
-------------------
- Le chiamate a Polymarket sono I/O di rete: gia' async (httpx).
- Il Monte Carlo e' CPU-bound e BLOCCANTE: se lo eseguissimo nel loop async
  bloccherebbe l'intero server. Lo spostiamo quindi in un thread separato con
  `asyncio.to_thread`, mantenendo l'endpoint reattivo.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Query

from app.analysis.divergence import compute_divergence
from app.analysis.metrics import kl_divergence, mean_absolute_divergence
from app.api.schemas import (
    DivergenceRow,
    HealthResponse,
    MetricsResponse,
    TeamProbability,
)
from app.core.simulator import run_tournament
from app.data.dummy import DUMMY_TEAMS
from app.market.polymarket import get_market_probabilities

app = FastAPI(
    title="WC2026 Market Efficiency Engine",
    version="2.0.0",
    description=(
        "Studio di efficienza di mercato: confronto tra un modello Monte Carlo + "
        "Poisson e il prediction market Polymarket. Fini educativi/dimostrativi."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Health check banale, utile per Docker/orchestratori."""
    return HealthResponse(status="ok")


@app.get("/predictions", response_model=list[TeamProbability], tags=["model"])
async def predictions(
    n_simulations: int = Query(10_000, ge=1_000, le=200_000),
) -> list[TeamProbability]:
    """
    Probabilita' di vittoria dal modello Monte Carlo.

    Il calcolo (CPU-bound) gira in un thread per non bloccare il loop async.
    """
    probs = await asyncio.to_thread(run_tournament, DUMMY_TEAMS, n_simulations)
    return [TeamProbability(team=t, probability=p) for t, p in probs.items()]


@app.get("/market", response_model=list[TeamProbability], tags=["market"])
async def market() -> list[TeamProbability]:
    """Probabilita' implicite (post de-vig) dal mercato Polymarket."""
    try:
        markets = await get_market_probabilities()
    except Exception as exc:  # rete/API non disponibili
        raise HTTPException(status_code=502, detail=f"Polymarket non raggiungibile: {exc}")
    return [TeamProbability(team=m.team, probability=m.fair_prob) for m in markets]


async def _model_and_market(
    n_simulations: int,
) -> tuple[dict[str, float], list]:
    """Recupera modello (in thread) e mercato IN PARALLELO. Helper condiviso."""
    model_task = asyncio.to_thread(run_tournament, DUMMY_TEAMS, n_simulations)
    try:
        return await asyncio.gather(model_task, get_market_probabilities())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Polymarket non raggiungibile: {exc}")


@app.get("/divergence", response_model=list[DivergenceRow], tags=["analysis"])
async def divergence(
    n_simulations: int = Query(10_000, ge=1_000, le=200_000),
    threshold: float = Query(0.02, ge=0.0, le=0.5),
) -> list[DivergenceRow]:
    """
    Confronto modello-vs-mercato: dove e quanto divergono le due stime.

    Studio di efficienza di mercato (non un consiglio di scommessa).
    """
    model_probs, market_probs = await _model_and_market(n_simulations)
    rows = compute_divergence(model_probs, market_probs, threshold=threshold)
    return [
        DivergenceRow(
            team=r.team,
            model_prob=r.model_prob,
            market_prob=r.market_prob,
            divergence=r.divergence,
            abs_divergence=r.abs_divergence,
            assessment=r.assessment.value,
        )
        for r in rows
    ]


@app.get("/metrics", response_model=MetricsResponse, tags=["analysis"])
async def metrics(
    n_simulations: int = Query(10_000, ge=1_000, le=200_000),
) -> MetricsResponse:
    """Metriche aggregate di efficienza (divergenza media, KL) sulle squadre comuni."""
    model_probs, market_probs = await _model_and_market(n_simulations)
    rows = compute_divergence(model_probs, market_probs)
    model_map = {r.team: r.model_prob for r in rows}
    market_map = {r.team: r.market_prob for r in rows}
    return MetricsResponse(
        mean_absolute_divergence=mean_absolute_divergence(model_map, market_map),
        kl_divergence=kl_divergence(model_map, market_map),
        teams_compared=len(rows),
    )
