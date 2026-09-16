# Prognoza sprzedaży — Demand Forecasting

Prognoza dziennej sprzedaży sieci detalicznej (`Units Sold`, sztuki) na najbliższe 4 tygodnie,
z uwzględnieniem czynników zewnętrznych — z aplikacją webową w Streamlicie.

> 🚧 Projekt w budowie — sekcje "Model", "Wyniki" i "Aplikacja" będą uzupełniane.

## Dane

[Demand forecasting dataset](https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset)
(Kaggle, pobieranie przez `kagglehub`): 76 000 transakcji — 5 sklepów × 20 produktów × 760 dni
(2022-01-01 → 2024-01-30, brak luk w kalendarzu).

Agregacja do **dziennego szeregu sieciowego**: suma `Units Sold` po sklepach i produktach
(760 punktów). Szczegóły agregacji zmiennych zewnętrznych: `src/data.py` (docstring `aggregate_daily`).

## Uruchomienie

```bash
nix develop              # środowisko: uv + git (nixpkgs, spikowane w flake.lock)
uv sync --frozen         # zależności Pythona dokładnie z uv.lock
uv run python src/data.py                      # pobranie + agregacja → outputs/daily_sales.csv
uv run streamlit run src/app.py                # (docelowo) aplikacja
```

Bez Nixa: wystarczy `uv sync --frozen` (Python ≥ 3.10).

## Wybór czynników zewnętrznych

Pełna analiza: [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb). Kluczowe ustalenia:

| Czynnik | Efekt | Decyzja |
|---|---|---|
| `Epidemic` | dni z epidemią: **6 086** vs 9 582 szt./dzień (−36%), korelacja −0.90 | ✅ do modelu (scenariusz out-of-sample: brak epidemii) |
| `Seasonality` (pora roku) | lato 9 576 vs wiosna 8 346 (~15% rozstrzału) | ✅ do modelu (sezonowość roczna) |
| `Price`, `Competitor Pricing` | wysokie korelacje (0.92/0.91) — najpewniej artefakt generatora danych | ⚠️ kontekst; uczciwe zastrzeżenie: korelacja ≠ przyczynowość |
| `Promotion`, `Discount` | efekt rozmyty po agregacji (udział dnia; 8.1–9.5 tys. bez wyraźnego trendu) | ❌ pominięte w modelu sieciowym (wracają per produkt w bonusie) |
| `Weather Condition` | słaby (0.0–0.16 po agregacji) | ❌ pominięte, opisane |
| `Demand` vs `Units Sold` | Demand = cenzurowany popyt; różnica (lost sales ~1.5 tys. szt./dzień) rośnie przy niskim zapasie per wiersz | 📌 do EDA i bonusu; prognozujemy `Units Sold` zgodnie z zadaniem |

### Czego nie włączyliśmy i dlaczego (udokumentowany test)

**Udziały produktów/kategorii w sprzedaży** (mikstura) — sprawdzono empirycznie (EDA §5b):
korelacje udziałów z przyszłą sumą sprzedaży są znikome (produkty ~0.05–0.10), a udział
Electronics po odjęciu informacji z perzyystencji szeregu (corr 0.825) dodaje tylko ~2%
wyjaśnionej wariancji. Udziały opisują *strukturę* koszyka, nie *poziom* sprzedaży —
dlatego nie trafiają do agregacji/modelu głównego.

Kolumny `Category`, `Region`, `Store ID`, `Product ID` są wymiarami produktu/lokalizacji —
po zagregowaniu sieci tracą sens wprost (suma po 20 produktach z 5 kategorii nie ma kategorii)
i wracają jako cechy w modelach per produkt (bonus).

## Struktura

```
├── src/data.py       # pobieranie (kagglehub), kontrola jakości, agregacja, split
├── src/model.py      # (TODO) baseline + model, prognoza h=28, metryki
├── src/app.py        # (TODO) Streamlit: wykres, tabela metryk, fragmentator
├── notebooks/EDA.ipynb
├── outputs/figures/  # wykresy EDA
└── priv/             # notatki robocze (niewidoczne w repo)
```
