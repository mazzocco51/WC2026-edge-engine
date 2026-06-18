# WC2026 — Market Efficiency Engine

Studio di **efficienza di un prediction market**: un modello statistico
proprietario (Monte Carlo + Poisson) stima la probabilità di vittoria di ogni
nazionale alla **FIFA World Cup 2026** e viene confrontato con le probabilità
implicite del mercato **Polymarket**, per misurare **dove e quanto** le due
fonti divergono e quanto il mercato è efficiente.

Progetto pensato per girare su hardware modesto: **niente deep learning, niente LLM** —
solo statistica pura, NumPy vettorizzato e software engineering modulare.

> ⚠️ **Disclaimer** — Progetto a fini **esclusivamente educativi e dimostrativi**
> (portfolio). **Non commerciale**, **non** è un consiglio di scommessa o finanziario.
>
> 📊 **Dati** — Risultati storici da [martj42 / international_results](https://github.com/martj42/international_results),
> licenza **CC0-1.0** (pubblico dominio). Dettagli e licenze in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Architettura

| Componente | Modulo | Stato |
|---|---|---|
| Math Engine (Monte Carlo + Poisson) | `app/core/` | ✅ |
| Rating reali da dati CC0 (Dixon-Coles) | `app/data/`, `app/core/ratings.py` | ✅ |
| Torneo 48 squadre formato 2026 | `app/core/tournament.py` | ✅ |
| Market Engine (Polymarket API async) | `app/market/` | ✅ |
| Analisi efficienza (divergenza + metriche) | `app/analysis/` | ✅ |
| Backend API (FastAPI async) | `app/api/` | ✅ |
| Dashboard (Streamlit) | `dashboard/` | ✅ |
| Containerizzazione (Docker) | `Dockerfile`, `docker-compose.yml` | ✅ |

## Quickstart

```bash
pip install -r requirements.txt

# Demo del core matematico (torneo dummy a 4 squadre, 10.000 simulazioni)
python -m app.core.run_demo

# Demo del market engine (probabilità implicite live da Polymarket)
python -m app.market.run_demo

# Demo con DATI REALI (CC0): scarica i risultati e stima attacco/difesa
python -m app.core.run_real_demo

# Demo Mondiale 2026 COMPLETO: 48 squadre, 12 gironi, eliminazione (dati reali)
python -m app.core.run_wc_demo

# Demo analisi di efficienza (divergenza modello-vs-mercato + metriche)
python -m app.analysis.run_demo

# Test
pytest -q
```

## Avvio con Docker (consigliato)

```bash
docker compose up --build
```

- API (FastAPI + docs OpenAPI): http://localhost:8000/docs
- Dashboard (Streamlit): http://localhost:8501

## Avvio manuale (senza Docker)

```bash
# Terminale 1 — backend
uvicorn app.api.main:app --reload

# Terminale 2 — dashboard
streamlit run dashboard/streamlit_app.py
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

## L'analisi di efficienza

Il modulo `app/analysis/` confronta le due distribuzioni di probabilità e ne
misura la **divergenza** (`model − market`, con segno): dove è vicina a zero il
mercato e il modello concordano (mercato efficiente su quella squadra), dove è
ampia almeno una delle due fonti sta sopra/sotto-stimando. A livello aggregato
si calcolano la **divergenza media assoluta** e la **KL divergence**; per il
backtest su tornei già conclusi sono disponibili **Brier score** e **log-loss**,
che misurano quale fonte prevede meglio una volta noto l'esito reale.

> Strumento di sola analisi statistica: non produce indicazioni di scommessa.

## Il torneo (formato 2026)

`app/core/tournament.py` simula il Mondiale nel formato ufficiale a **48 squadre**:
12 gironi da 4 all'italiana, passano le prime 2 di ogni girone + le 8 migliori
terze (32 squadre), poi eliminazione diretta fino alla finale. I gironi sono
costruiti dai rating stimati con un sorteggio a fasce (pot) riproducibile
(`app/data/wc2026.py`). Il piazzamento delle terze e l'accoppiamento del
tabellone usano una semplificazione documentata (teste di serie standard), non
la tabella combinatoria esatta della FIFA.
