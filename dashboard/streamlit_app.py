"""
streamlit_app.py
================
"Who'll win the World Cup?" — a Monte Carlo simulation dashboard.

Pure presentation layer: it only consumes the FastAPI backend (no domain logic).
Shows model-vs-market win probabilities and example knockout brackets, drawn as
a classic single-elimination tree (a fresh sample of simulations on every refresh).

Educational/demonstrative project. Not betting advice.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
GOLD = "#C9A227"
BLUE = "#1D3DB5"

st.set_page_config(page_title="Who'll win the World Cup?", page_icon="⚽", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(90deg,#1D3DB5,#00A878,#F4C430,#FF6B35);
                padding:22px 26px;border-radius:14px;margin-bottom:18px;">
      <div style="font-size:34px;font-weight:800;color:#fff;
                  text-shadow:0 2px 6px rgba(0,0,0,.35);">Who'll win the World Cup?</div>
      <div style="font-size:16px;color:#fff;opacity:.95;
                  text-shadow:0 1px 4px rgba(0,0,0,.35);">
        Monte Carlo simulation · 2026 FIFA World Cup · model vs Polymarket</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Educational / demonstrative project — not betting advice.")

col1, col2 = st.columns(2)
with col1:
    n_sims = st.slider("Monte Carlo simulations", 1_000, 20_000, 5_000, step=1_000)
with col2:
    threshold = st.slider("Notable-divergence threshold", 0.0, 0.10, 0.02, step=0.01)

refresh = st.button("Refresh data", type="primary")


@st.cache_data(ttl=60)
def fetch_divergence(n_simulations: int, thr: float) -> list[dict]:
    r = requests.get(f"{API_URL}/divergence",
                     params={"n_simulations": n_simulations, "threshold": thr}, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def fetch_brackets(n: int) -> list[dict]:
    r = requests.get(f"{API_URL}/bracket", params={"n": n}, timeout=120)
    r.raise_for_status()
    return r.json()


if refresh:
    fetch_divergence.clear()
    fetch_brackets.clear()


def _style_assessment(val: str) -> str:
    color = {"Model > Market": "#1a7f37", "Model < Market": "#cf222e"}.get(val, "#57606a")
    return f"color: white; background-color: {color}; font-weight: 600;"


# ============================ Win probabilities ============================
st.subheader("Win probability — model vs market")
try:
    rows = fetch_divergence(n_sims, threshold)
    df = pd.DataFrame(rows)
    view = pd.DataFrame({
        "Team": df["team"],
        "Model %": (df["model_prob"] * 100).round(1),
        "Market %": (df["market_prob"] * 100).round(1),
        "Divergence (pt)": (df["divergence"] * 100).round(1),
        "Assessment": df["assessment"],
    })
    styled = view.style.format(
        {"Model %": "{:.1f}", "Market %": "{:.1f}", "Divergence (pt)": "{:+.1f}"}
    ).applymap(_style_assessment, subset=["Assessment"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    mad = (df["abs_divergence"] * 100).mean()
    c1, c2 = st.columns(2)
    c1.metric("Mean absolute divergence", f"{mad:.1f} pt")
    c2.metric("Teams with notable divergence", int((df["assessment"] != "Aligned").sum()))
except Exception as exc:
    st.warning(f"Market comparison unavailable ({exc}). Brackets below still work.")


# ============================ Knockout bracket =============================
st.subheader("Knockout bracket — sample simulations (resampled on refresh)")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _short(s: str) -> str:
    return s if len(s) <= 12 else s[:11] + "…"


def bracket_svg(bk: dict) -> str:
    """Bracket as a compact left-to-right tree, from the Round of 16 onwards."""
    mr = [rnd["matches"] for rnd in bk["rounds"]][1:]   # drop R32: R16, QF, SF, Final
    NS, ROW, TOP, COLW = 16, 20, 12, 72
    height = NS * ROW + TOP * 2
    width = 5 * COLW

    ys = [[TOP + i * ROW + ROW / 2 for i in range(NS)]]
    for _ in range(4):
        prev = ys[-1]
        ys.append([(prev[2 * j] + prev[2 * j + 1]) / 2 for j in range(len(prev) // 2)])

    names = [[]]
    lvl0: list[str] = []
    for m in mr[0]:
        lvl0 += [m["a"], m["b"]]
    names[0] = lvl0
    for k in range(4):
        names.append([m["winner"] for m in mr[k]])

    parts: list[str] = []
    for k in range(5):
        for i, nm in enumerate(names[k]):
            x0, x1, y = k * COLW, (k + 1) * COLW, ys[k][i]
            win = (nm == mr[k][i // 2]["winner"]) if k < 4 else True
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#9a9a9a" stroke-width="1"/>')
            col = GOLD if win else "currentColor"
            wt = "700" if win else "400"
            parts.append(
                f'<text x="{x0 + 3}" y="{y - 3}" font-size="9.5" font-weight="{wt}" fill="{col}">{_esc(_short(nm))}</text>'
            )
    for k in range(4):
        for p in range(len(names[k]) // 2):
            x = (k + 1) * COLW
            yt, yb = ys[k][2 * p], ys[k][2 * p + 1]
            parts.append(f'<line x1="{x}" y1="{yt}" x2="{x}" y2="{yb}" stroke="#9a9a9a" stroke-width="1"/>')
            m = mr[k][p]
            pens = " p" if m["pens"] else ""
            parts.append(
                f'<text x="{x - 3}" y="{(yt + yb) / 2 - 2}" font-size="8" fill="{BLUE}" text-anchor="end">{m["ga"]}–{m["gb"]}{pens}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">'
        f'{"".join(parts)}</svg>'
    )


try:
    brackets = fetch_brackets(3)
    cols = st.columns(3)
    for col, (i, bk) in zip(cols, enumerate(brackets)):
        with col:
            st.markdown(
                f"<div style='font-size:14px'>Simulation {i + 1}</div>"
                f"<div style='font-size:18px;font-weight:800;color:{GOLD}'>{bk['champion']}</div>",
                unsafe_allow_html=True,
            )
            html = (
                "<style>html,body{margin:0;background:transparent;color:inherit}</style>"
                + bracket_svg(bk)
            )
            components.html(html, height=16 * 20 + 30, scrolling=False)
    st.caption("\* Brackets are shown from the Round of 16 onwards (the Round of 32 is omitted for readability).")
except Exception as exc:
    st.error(f"Could not load brackets ({exc}).")
