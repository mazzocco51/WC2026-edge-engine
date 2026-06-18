"""
schemas.py
==========
Modelli di risposta (pydantic) esposti dall'API.

Tenere gli schemi separati dalla logica di routing rende l'API auto-documentata
(FastAPI genera lo schema OpenAPI da questi modelli) e disaccoppia il "contratto"
HTTP dalle dataclass interne del dominio.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeamProbability(BaseModel):
    """Probabilita' di vittoria di una squadra secondo una singola fonte."""

    team: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilita' [0,1]")


class DivergenceRow(BaseModel):
    """Riga di confronto modello-vs-mercato (studio di efficienza)."""

    team: str
    model_prob: float = Field(..., ge=0.0, le=1.0)
    market_prob: float = Field(..., ge=0.0, le=1.0, description="probabilita di mercato (de-vig)")
    divergence: float = Field(..., description="model_prob - market_prob (con segno)")
    abs_divergence: float = Field(..., ge=0.0, description="valore assoluto della divergenza")
    assessment: str = Field(..., description="Model > Market | Aligned | Model < Market")


class MetricsResponse(BaseModel):
    """Metriche aggregate di efficienza modello-vs-mercato."""

    mean_absolute_divergence: float = Field(..., description="scostamento medio assoluto")
    kl_divergence: float = Field(..., description="KL(modello||mercato) in nats")
    teams_compared: int


class HealthResponse(BaseModel):
    """Risposta dell'health check."""

    status: str = "ok"
