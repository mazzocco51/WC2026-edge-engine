# WC2026 — Quant Sports Betting Engine

Predizione del vincitore della **FIFA World Cup 2026** confrontando un modello
statistico proprietario (Monte Carlo + Poisson) con il prediction market
**Polymarket**, per individuare situazioni di valore atteso positivo (**+EV / Edge**).

Progetto pensato per girare su hardware modesto: **niente deep learning, niente LLM** —
solo statistica pura, NumPy vettorizzato e software engineering modulare.

## Architettura

| Componente | Modulo | Stato |
|---|---|---|
| Math Engine (Monte Carlo + Poisson) | `app/core/` | ✅ Step 1 |
| Market Engine (Polymarket API async) | `app/market/` | ✅ Step 2 |
| Edge Calculator (Model vs Market) | `app/edge/` | 🔜 Step 3 |
| Backend API (FastAPI async) | `app/api/` | 🔜 Step 4 |
| Dashboard (Streamlit) | `dashboard/` | 🔜 Step 4 |
| Containerizzazione (Docker) | `Dockerfile`, `docker-compose.yml` | 🔜 Step 4 |

## Quickstart

```bash
pip install -r requirements.txt

# Demo del core matematico (torneo dummy a 4 squadre, 10.000 simulazioni)
python -m app.core.run_demo

# Demo del market engine (probabilità implicite live da Polymarket)
python -m app.market.run_demo

# Test
pytest -q
```

## Il modello (Step 1)

I gol di ogni squadra seguono una distribuzione di Poisson con parametro:

```
λ = MEDIA_GOL × attacco_squadra × difesa_avversario
```

dove `attacco` e `difesa` sono moltiplicatori relativi alla media del torneo
(1.0 = squadra media). La simulazione Monte Carlo ripete il tabellone N volte
e stima la probabilità di vittoria di ciascuna squadra.

## Il mercato (Step 2)

Il Market Engine interroga in modo asincrono (`httpx`) la Gamma API pubblica di
Polymarket per l'evento *World Cup Winner*. Per ogni squadra estrae il prezzo
"Yes" (= probabilità implicita) e rimuove il **vig** (overround) normalizzando
le probabilità a somma 100%, così sono confrontabili 1:1 con il modello.
