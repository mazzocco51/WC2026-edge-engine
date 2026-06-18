# Fonti dati & licenze

Questo progetto usa **una sola fonte dati storica**, scelta deliberatamente per
garantire massima sicurezza legale (licenza di pubblico dominio).

## 1. Risultati storici delle nazionali

- **Dataset:** *International football results from 1872 to present*
- **Autore:** Mart Jürisoo ([@martj42](https://github.com/martj42))
- **Repository:** https://github.com/martj42/international_results
- **File usato:** `results.csv` (scaricato a runtime dal raw GitHub)
- **Licenza:** **CC0-1.0** — Creative Commons Public Domain Dedication
  (https://creativecommons.org/publicdomain/zero/1.0/)

La licenza CC0 colloca l'opera nel pubblico dominio: l'uso, la modifica e la
ridistribuzione sono liberi, **anche a scopo commerciale, senza obbligo di
attribuzione**. Citiamo comunque l'autore per correttezza e trasparenza.

> Nota: i dati **non** vengono ridistribuiti dentro questo repository; vengono
> scaricati a runtime dalla fonte originale (`app/data/loader.py`).

## 2. Quote di mercato (sola lettura)

- **Fonte:** Polymarket — Gamma API pubblica (`https://gamma-api.polymarket.com`)
- **Uso:** lettura delle probabilità implicite del mercato "World Cup Winner".
  Nessun dato rivenduto o ridistribuito; nessuna affiliazione con Polymarket.

## Cosa NON usiamo (per scelta)

Per restare nel perimetro "100% libero e sicuro" abbiamo **escluso**:

- Ranking FIFA ufficiale e dati Opta/FBref (proprietari, non ridistribuibili)
- World Football Elo (licenza non aperta in modo formale)
- StatsBomb / dati xG (gratuiti ma con obblighi di attribuzione/uso)

---

## ⚠️ Disclaimer

Questo è un progetto **a fini esclusivamente educativi e dimostrativi**, creato
per portfolio personale. **Non è un servizio commerciale** e **non costituisce
consiglio di scommessa o finanziario**. Le previsioni sono il risultato di un
modello statistico e non garantiscono alcun esito. L'autore non si assume
responsabilità per eventuali usi diversi da quello dimostrativo.
