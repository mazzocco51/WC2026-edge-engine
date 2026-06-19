"""
main.py
=======
Async FastAPI app exposing the model, the market and the analysis over HTTP.

Endpoints
---------
GET /health       -> service status
GET /predictions  -> Monte Carlo model win probabilities
GET /market       -> Polymarket implied probabilities (de-vigged)
GET /divergence   -> model-vs-market divergence (efficiency study)
GET /metrics      -> aggregate efficiency metrics (mean divergence, KL)
GET /bracket      -> sample knockout brackets with results

Concurrency: Polymarket calls are async (httpx); the CPU-bound Monte Carlo runs
in a thread (`asyncio.to_thread`) and the two are fetched in parallel.
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
from app.core.engine import model_probabilities, sample_brackets
from app.market.polymarket import get_market_probabilities

app = FastAPI(
    title="WC2026 Market Efficiency Engine",
    version="2.0.0",
    description=(
        "Market-efficiency study: a Monte Carlo + Poisson model vs the Polymarket "
        "prediction market. Educational/demonstrative, non-commercial."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Trivial health check (useful for Docker/orchestrators)."""
    return HealthResponse(status="ok")


@app.get("/predictions", response_model=list[TeamProbability], tags=["model"])
async def predictions(
    n_simulations: int = Query(5_000, ge=1_000, le=50_000),
) -> list[TeamProbability]:
    """Monte Carlo win probabilities (CPU-bound work runs in a thread)."""
    probs = await asyncio.to_thread(model_probabilities, n_simulations)
    return [TeamProbability(team=t, probability=p) for t, p in probs.items()]


@app.get("/market", response_model=list[TeamProbability], tags=["market"])
async def market() -> list[TeamProbability]:
    """De-vigged implied probabilities from Polymarket."""
    try:
        markets = await get_market_probabilities()
    except Exception as exc:  # network/API unavailable
        raise HTTPException(status_code=502, detail=f"Polymarket unreachable: {exc}")
    return [TeamProbability(team=m.team, probability=m.fair_prob) for m in markets]


async def _model_and_market(n_simulations: int) -> tuple[dict[str, float], list]:
    """Fetch model (in a thread) and market IN PARALLEL. Shared helper."""
    model_task = asyncio.to_thread(model_probabilities, n_simulations)
    try:
        return await asyncio.gather(model_task, get_market_probabilities())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Polymarket unreachable: {exc}")


@app.get("/divergence", response_model=list[DivergenceRow], tags=["analysis"])
async def divergence(
    n_simulations: int = Query(5_000, ge=1_000, le=50_000),
    threshold: float = Query(0.02, ge=0.0, le=0.5),
) -> list[DivergenceRow]:
    """Model-vs-market divergence (efficiency study; not betting advice)."""
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
    n_simulations: int = Query(5_000, ge=1_000, le=50_000),
) -> MetricsResponse:
    """Aggregate efficiency metrics (mean abs divergence, KL) over common teams."""
    model_probs, market_probs = await _model_and_market(n_simulations)
    rows = compute_divergence(model_probs, market_probs)
    model_map = {r.team: r.model_prob for r in rows}
    market_map = {r.team: r.market_prob for r in rows}
    return MetricsResponse(
        mean_absolute_divergence=mean_absolute_divergence(model_map, market_map),
        kl_divergence=kl_divergence(model_map, market_map),
        teams_compared=len(rows),
    )


@app.get("/bracket", tags=["model"])
async def bracket(n: int = Query(3, ge=1, le=5)) -> list[dict]:
    """`n` sample knockout brackets (with scores) from the Monte Carlo engine."""
    return await asyncio.to_thread(sample_brackets, n)
