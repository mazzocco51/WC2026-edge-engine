# WC2026 — Market Efficiency Engine

Studio di **efficienza di un prediction market**: un modello statistico
proprietario (Monte Carlo + Poisson) stima la probabilità di vittoria di ogni
nazionale alla **FIFA World Cup 2026** e la confronta con le probabilità
implicite del mercato **Polymarket**, per misurare **dove e quanto** le due
fonti divergono e quanto il mercato è efficiente.

Pensato per girare su hardware modesto: **niente deep learning, niente LLM** —
solo statistica, NumPy e software engineering modulare.

> ⚠️ **Disclaimer** — Progetto a fini **esclusivamente educativi e dimostrativi**
> (portfolio). **Non commerciale**, **non** è un consiglio di scommessa o
> finanziario. Le previsioni vengono da un modello statistico e non garantiscono
> alcun esito.
>
> 📊 **Dati** — Risultati storici da
> [martj42 / international_results](https://github.com/martj42/international_results),
> licenza **CC0-1.0** (pubblico dominio). Dettagli in
> [`DATA_SOURCES.md`](DATA_SOURCES.md).

---

## Indice

- [Architettura](#architettura)
- [Quickstart](#quickstart)
- [Pipeline e decisioni di design](#pipeline-e-decisioni-di-design)
- [Le metriche spiegate](#le-metriche-spiegate)
- [Limiti noti e prossimi passi](#limiti-noti-e-prossimi-passi)

---

## Architettura

| Componente | Modulo | Ruolo |
|---|---|---|
| Modelli dati | `app/core/models.py` | dataclass `Team`, `MatchResult` |
| Modello gol (Poisson) | `app/core/poisson.py` | gol attesi λ e campionamento |
| Stima rating | `app/core/ratings.py` | attacco/difesa + vantaggio campo + ρ |
| Correzione Dixon-Coles | `app/core/dixon_coles.py` | ρ, matrice punteggi, 1X2 |
| Rating Elo (FIFA-like) | `app/core/elo.py` | forza-squadra dai risultati CC0 |
| Torneo 48 squadre | `app/core/tournament.py` | gironi + eliminazione, formato 2026 |
| Orchestratore | `app/core/engine.py` | dati → rating → torneo (con cache) |
| Caricamento dati | `app/data/loader.py` | download CC0 a runtime |
| Gironi 2026 | `app/data/wc2026.py` | sorteggio UFFICIALE (48 squadre, 12 gironi) |
| Market engine | `app/market/polymarket.py` | probabilità implicite Polymarket |
| Analisi efficienza | `app/analysis/divergence.py` | divergenza modello-vs-mercato |
| Metriche | `app/analysis/metrics.py` | divergenza media, KL, Brier, log-loss, RPS |
| Backtest & tuning | `app/analysis/backtest.py` | RPS walk-forward + taratura emivita e shrinkage |
| Backend API | `app/api/` | FastAPI asincrono |
| Dashboard | `dashboard/streamlit_app.py` | tabella interattiva |
| Container | `Dockerfile`, `docker-compose.yml` | deploy riproducibile |

---

## Quickstart

```bash
pip install -r requirements.txt

# Core matematico (torneo dummy a 4 squadre)
python -m app.core.run_demo

# Market engine (probabilità live da Polymarket)
python -m app.market.run_demo

# Rating reali da dati CC0 (attacco/difesa per nazionale)
python -m app.core.run_real_demo

# Mondiale 2026 completo: 48 squadre, 12 gironi, eliminazione
python -m app.core.run_wc_demo

# Analisi di efficienza (divergenza modello-vs-mercato)
python -m app.analysis.run_demo

# Backtest 1X2: Poisson vs Dixon-Coles via RPS + taratura emivita
python -m app.analysis.run_backtest

# Reliability diagram (calibrazione): genera calibration_diagram.png
python -m app.analysis.run_calibration

# Test
pytest -q
```

### Con Docker (consigliato)

```bash
docker compose up --build
```

- API + documentazione OpenAPI: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Senza Docker

```bash
uvicorn app.api.main:app --reload                 # terminale 1
streamlit run dashboard/streamlit_app.py          # terminale 2
```

---

## Pipeline e decisioni di design

Questa sezione spiega **ogni scelta**: non solo *cosa* fa il codice, ma *perché*.

### 1. Dati — `app/data/loader.py`

**Fonte unica, CC0.** Usiamo solo il dataset martj42 (pubblico dominio): zero
vincoli legali, gratuito, nessuna API key. Scelta deliberata per restare nel
perimetro "100% libero" — Elo, FBref/Opta, ranking FIFA sono stati esclusi
perché proprietari o con licenze non aperte (vedi `DATA_SOURCES.md`).

**Download a runtime, niente ridistribuzione.** Il CSV viene scaricato dalla
fonte originale alla prima esecuzione (con `urllib`, libreria standard): non
copiamo i dati nel repository, così rispettiamo la fonte e teniamo il repo
leggero. Nessuna dipendenza aggiuntiva a questo livello.

**Parsing robusto.** Il Mondiale 2026 è in corso, quindi il dataset contiene
anche partite future con punteggio `NA` o vuoto: il parser le scarta (qualsiasi
valore non numerico fa saltare la riga). Si tiene anche il flag `neutral`,
cruciale perché in campo neutro non c'è vantaggio casalingo.

### 2. Modello dei gol — `app/core/poisson.py`

I gol di una squadra in una partita sono ben descritti da una **distribuzione di
Poisson**: contano eventi discreti, rari e (in prima approssimazione)
indipendenti, con un solo parametro λ pari sia alla media sia alla varianza.

I gol attesi sono `λ = base × attacco_i × difesa_j`, con `attacco`/`difesa`
moltiplicatori relativi alla media (1.0 = squadra media; `difesa > 1` = subisce
di più). Questa parametrizzazione moltiplicativa rende i rating leggibili e
calibrabili. Il campionamento usa un `np.random.Generator` esplicito: simulazioni
**riproducibili** (seed) e thread-safe.

### 3. Stima dei rating — `app/core/ratings.py`

**Perché la massima verosimiglianza e non i rapporti grezzi.** Il metodo
"ingenuo" (gol fatti / media) non funziona per le nazionali, perché il calendario
è sbilanciato (il Brasile gioca tante amichevoli facili). Stimiamo invece
attacco e difesa *congiuntamente* massimizzando la verosimiglianza Poisson
(modello Maher/Dixon-Coles): l'aggiustamento per la forza dell'avversario è
automatico.

**Perché l'iterative scaling e non scipy.** Usiamo aggiornamenti moltiplicativi
a punto fisso: convergono in modo robusto e dipendono **solo da NumPy** — niente
scipy/statsmodels, coerente col vincolo di hardware leggero. I moltiplicatori
vengono normalizzati a media 1 a ogni iterazione (la scala finisce in `base`).

**Time-decay.** Ogni partita pesa `0.5 ** (età / emivita)`: le più recenti
contano di più, così i rating riflettono la forma attuale. L'emivita di default
è **4 anni**: non scelta a occhio, ma il valore che minimizza l'RPS nel backtest
(vedi §8). Per le nazionali l'ottimo è lungo perché cambiano lentamente e
giocano poche partite l'anno — serve una finestra ampia per avere segnale.

**Shrinkage (regolarizzazione) — testato e disattivato di default.** Lo shrinkage
aggiunge "partite fantasma" che tirano i rating verso un prior, utile in teoria
per le squadre con pochi dati. **Ma il backtest RPS dice il contrario:** sugli
internazionali, con finestra di 4 anni, le squadre hanno dati a sufficienza e la
regolarizzazione introduce solo bias — l'RPS peggiora in modo monotòno
(`0` → 0.1666, `2` → 0.1682, `4` → 0.1692, `8` → 0.1705). Per questo
`DEFAULT_SHRINKAGE = 0`: lasciamo decidere ai dati, non all'occhio. Il parametro
resta disponibile e tarabile (§8) per chi volesse sperimentarlo.

**Prior di forza Elo (`app/core/elo.py`) — implementato, ma non migliora l'RPS.**
L'Elo è calcolato DA NOI sui risultati CC0 (il ranking FIFA è esso stesso un Elo
dal 2018), quindi è un equivalente del ranking FIFA **senza dati proprietari**.
L'idea: usarlo come prior dello shrinkage per non appiattire le big con pochi
risultati recenti brillanti (es. Francia). **Onestà dai dati:** poiché il prior
agisce solo tramite lo shrinkage — e lo shrinkage peggiora l'RPS — il prior Elo
non migliora la previsione out-of-sample (a parità di shrinkage aiuta, ma il
modello globalmente migliore è senza shrinkage). È un risultato istruttivo:
*migliorare la plausibilità visiva del ranking non equivale a migliorare
l'accuratezza.* Elo e shrinkage restano nel codice come opzioni tarabili.

**Vantaggio campo.** Stimiamo un fattore `γ` (gamma), ma nel Mondiale tutte le
partite sono in campo neutro, quindi non viene applicato nella simulazione del
torneo (resta disponibile per il backtest sulle partite reali, che includono
gare in casa/trasferta).

### 4. Correzione Dixon-Coles — `app/core/dixon_coles.py`

Il Poisson puro **sottostima i pareggi e i risultati a basso punteggio**.
Dixon-Coles (1997) moltiplica la probabilità dei soli quattro punteggi bassi
(0-0, 1-0, 0-1, 1-1) per un fattore `τ` dipendente da un parametro `ρ`, stimato
per massima verosimiglianza. Lo stimiamo con una **ricerca su griglia**: solo il
termine `τ` dipende da `ρ`, quindi l'ottimizzazione è economica e senza scipy.

### 5. Torneo a 48 squadre — `app/core/tournament.py`

Formato ufficiale 2026: 12 gironi da 4 all'italiana, prime 2 + 8 migliori terze
(32 squadre), poi eliminazione diretta fino alla finale.

**Semplificazione documentata.** Il piazzamento esatto delle terze segue una
tabella combinatoria FIFA molto complessa: usiamo invece un seeding standard
(le qualificate sono ordinate per forza e inserite in un tabellone a teste di
serie, dove la 1 e la 2 si possono incontrare solo in finale). I gironi sono
quelli ufficiali (vedi §6); resta semplificato solo il piazzamento delle terze
e l'accoppiamento del tabellone, non la replica esatta della procedura FIFA.

**Perché Poisson indipendente (e non Dixon-Coles) nel loop.** Una run da 20.000
tornei sono ~2 milioni di partite. Il campionamento dalla matrice DC corretta
costa ~40 µs/partita (~80 s a run): troppo. Dato che l'effetto di `ρ` sul
*vincitore del torneo* è minimo, nel loop campioniamo da Poisson indipendente
(veloce) e applichiamo Dixon-Coles dove conta ed è economico: stima di `ρ`,
probabilità 1X2 e backtest. Trade-off prestazioni/precisione esplicito.

### 6. Sorteggio dei gironi — `app/data/wc2026.py`

Usiamo il **sorteggio UFFICIALE del 2026** (`REAL_GROUPS_2026`): i 12 gironi
reali, con ogni squadra mappata ai rating stimati tramite un matching dei nomi
(gestisce accenti e alias come USA→United States, Czechia→Czech Republic). Così
il cammino simulato di ogni squadra è quello vero, non uno casuale.

Perché conta: con un sorteggio casuale la probabilità di vincere dipende dal
girone capitato per caso — e per squadre di forza simile è il sorteggio, non la
forza, a deciderne l'ordine. Usando i gironi veri questo artefatto sparisce.
Resta disponibile `build_groups_from_ratings()` (sorteggio fittizio a fasce) per
esperimenti "what-if".

### 7. Market engine — `app/market/polymarket.py`

Client **asincrono** (`httpx`) sulla Gamma API pubblica di Polymarket (nessuna
API key). Per ogni squadra si estrae il prezzo "Yes" (= probabilità implicita).
Dettaglio insidioso gestito: `outcomes`/`outcomePrices` arrivano come **stringhe
JSON** e vanno ri-parsate.

**Rimozione del vig.** La somma dei prezzi "Yes" supera il 100% (margine del
book). Normalizziamo in modo **proporzionale** (ogni prezzo diviso per la somma):
metodo semplice e standard. Esiste l'alternativa di Shin (assume trader
informati, più accurata sui favoriti) — citata come possibile estensione.

### 8. Analisi, metriche e backtest — `app/analysis/`

**Divergenza** (`divergence.py`): `model − market` con segno; valutazione neutra
`Model > / = / < Market`. Niente linguaggio di scommessa: è uno studio di
efficienza.

**Backtest walk-forward** (`backtest.py`): si allena **solo** sul passato e si
testa sul futuro — mai split casuale, che con dati temporali introdurrebbe *data
leakage*. Confronta l'RPS di Poisson puro vs Dixon-Coles.

**Taratura dell'emivita** (`tune_half_life`): prova più emivite e sceglie quella
col **miglior RPS out-of-sample**. L'iperparametro è scelto dai dati, non a
occhio.

**Taratura dello shrinkage** (`tune_shrinkage`): stessa logica per la forza
della regolarizzazione; `shrinkage = 0` (nessuna regolarizzazione) è incluso nel
confronto, così l'eventuale miglioramento è misurato.

### 9. Infrastruttura — `app/api/`, `dashboard/`, Docker

**FastAPI asincrono.** Endpoint `/health`, `/predictions`, `/market`,
`/divergence`, `/metrics`. Il Monte Carlo è CPU-bound: gira in un thread
(`asyncio.to_thread`) per non bloccare il loop; modello e mercato vengono
recuperati **in parallelo** (`asyncio.gather`).

**Caching.** Rating e gironi (dipendono solo dai dati) e le probabilità del
modello (dipendono solo da `n_simulations`) sono in `lru_cache`: richieste
ripetute sono istantanee.

**Dashboard.** Streamlit fa **solo** da client dell'API (nessuna matematica):
separazione netta presentazione/logica, così i due servizi scalano e si
deployano indipendentemente.

**Docker.** Un solo `Dockerfile` condiviso (dipendenze installate prima del
codice per sfruttare la cache dei layer) e un `docker-compose.yml` con due
servizi; la dashboard raggiunge l'API via DNS interno e parte solo quando
l'healthcheck dell'API è "healthy".

---

## Le metriche spiegate

**Divergenza media assoluta** — scostamento medio `|modello − mercato|`; quanto
le due fonti, in media, non concordano.

**KL divergence** — `KL(modello ‖ mercato)`: quanta "informazione extra" porta il
modello rispetto al mercato; 0 = identici.

**Brier score** — errore quadratico medio multi-classe sull'esito reale; più
basso = meglio. Per il vincitore del torneo.

**Log-loss** — `−log(prob assegnata al vincitore)`: penalizza l'essere sicuri e
sbagliati. Standard nelle competizioni di forecasting.

**RPS (Ranked Probability Score)** — per esiti **ordinati** (1X2:
casa → pareggio → ospite). A differenza del Brier tiene conto dell'ordine: un
errore "vicino" è penalizzato meno di uno "lontano". È lo standard per le
previsioni calcistiche 1X2.

**Calibrazione — reliability diagram + ECE.** Risponde a una domanda diversa
dall'accuratezza: *"quando il modello dice 30%, succede davvero il ~30% delle
volte?"*. Si raggruppano le previsioni in intervalli e si confronta la
probabilità media prevista (asse X) con la frequenza reale (asse Y): un modello
ben calibrato sta sulla diagonale. L'**ECE (Expected Calibration Error)** ne è la
sintesi numerica (scostamento medio previsto-vs-reale; più basso = meglio).
`app/analysis/run_calibration.py` genera il grafico (`calibration_diagram.png`)
confrontando Poisson puro e Dixon-Coles. Nota: confrontiamo la calibrazione del
*modello*, non del mercato, perché non disponiamo di quote storiche per-partita
(solo il market live del vincitore).

---

## Limiti noti e prossimi passi

- Il modello è "pre-torneo" (rating sull'intero storico), mentre Polymarket
  riflette i risultati dei gironi già giocati: una certa divergenza è attesa.
  Possibile estensione: emivita più aggressiva o incorporare i risultati live.
- I gironi sono quelli ufficiali del 2026; restano semplificati il piazzamento
  delle migliori terze e l'accoppiamento del tabellone (seeding standard).
- Dixon-Coles è applicato a livello di match/backtest, non nel loop del torneo
  (per performance).
- Possibili evoluzioni: rimozione del vig con metodo di Shin, Poisson bivariato,
  rating dinamici (stato-spazio/Kalman), calibrazione (reliability diagram / ECE).

---

## Fonti & licenze

Dettaglio completo in [`DATA_SOURCES.md`](DATA_SOURCES.md). In sintesi: risultati
storici da martj42 (**CC0-1.0**, pubblico dominio), scaricati a runtime; quote di
sola lettura da Polymarket (Gamma API pubblica). Progetto educativo/dimostrativo,
non commerciale.
