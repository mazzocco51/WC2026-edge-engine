"""
streamlit_app.py
================
Dashboard di ANALISI DELL'EFFICIENZA DI MERCATO: legge l'endpoint /divergence
del backend FastAPI e mostra quanto il modello statistico diverge dal prediction
market Polymarket.

Finalita' esclusivamente educativa/dimostrativa. Nessun consiglio di scommessa.

La dashboard NON contiene logica di dominio: e' solo un client del backend.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL: str = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="WC2026 Market Efficiency", page_icon="football", layout="wide")
st.title("WC2026 - Analisi di efficienza di mercato")
st.caption(
    "Modello Monte Carlo + Poisson vs prediction market Polymarket - "
    "studio educativo/dimostrativo (nessun consiglio di scommessa)"
)

col1, col2 = st.columns(2)
with col1:
    n_sims = st.slider("Simulazioni Monte Carlo", 1_000, 50_000, 10_000, step=1_000)
with col2:
    threshold = st.slider("Soglia divergenza notevole", 0.0, 0.10, 0.02, step=0.01)

refresh = st.button("Aggiorna dati", type="primary")


@st.cache_data(ttl=60)
def fetch_divergence(n_simulations: int, thr: float) -> list[dict]:
    """Chiama l'API /divergence. Cache di 60s per non martellare il backend."""
    resp = requests.get(
        f"{API_URL}/divergence",
        params={"n_simulations": n_simulations, "threshold": thr},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _style_assessment(val: str) -> str:
    """Colora la valutazione modello-vs-mercato."""
    color = {"Model > Market": "#1a7f37", "Model < Market": "#cf222e"}.get(val, "#57606a")
    return f"color: white; background-color: {color}; font-weight: 600;"


if refresh:
    fetch_divergence.clear()

try:
    rows = fetch_divergence(n_sims, threshold)
except Exception as exc:
    st.error(f"Impossibile contattare il backend ({API_URL}). Dettagli: {exc}")
    st.stop()

if not rows:
    st.warning("Nessuna squadra in comune tra modello e mercato.")
    st.stop()

df = pd.DataFrame(rows)
view = pd.DataFrame(
    {
        "Team": df["team"],
        "Model %": (df["model_prob"] * 100).round(1),
        "Polymarket %": (df["market_prob"] * 100).round(1),
        "Divergence (pt)": (df["divergence"] * 100).round(1),
        "Assessment": df["assessment"],
    }
)

styled = view.style.applymap(_style_assessment, subset=["Assessment"])
st.dataframe(styled, use_container_width=True, hide_index=True)

mad = (df["abs_divergence"] * 100).mean()
n_notable = int((df["assessment"] != "Aligned").sum())
c1, c2 = st.columns(2)
c1.metric("Divergenza media assoluta", f"{mad:.1f} pt")
c2.metric("Squadre con divergenza notevole", n_notable)
st.caption(
    "Una divergenza bassa indica che mercato e modello concordano "
    "(mercato efficiente su quella squadra); una alta indica disaccordo."
)
