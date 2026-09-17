# Prognoza sprzedaży — Demand Forecasting

Prognoza dziennej sprzedaży sieci detalicznej (`Units Sold`, sztuki) na najbliższe 4 tygodnie,
z uwzględnieniem czynników zewnętrznych — z aplikacją webową w Streamlicie.

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
uv run streamlit run src/app.py                # aplikacja webowa
```

Bez Nixa: wystarczy `uv sync --frozen` (Python ≥ 3.10).

Aplikacja czyta gotowe wyniki z `outputs/` (są w repo — regeneruje je deterministycznie
`src/data.py` + `src/model.py`), więc uruchamia się natychmiast, bez pobierania danych.

### Wdrożenie (Streamlit Community Cloud)

Aplikacja działa online na [share.streamlit.io](https://share.streamlit.io) (darmowe dla
repozytoriów publicznych): po zalogowaniu kontem GitHub → *Create app* → wskaż to repo,
branch `main`, plik `src/app.py`. Zależności dla chmury definiuje `requirements.txt`
(dane i wyniki są w repo, więc chmura nie potrzebuje dostępu do Kaggle).

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

## Model i wyniki

Kod: [`src/model.py`](src/model.py). Pełny pipeline: `src/data.py` → agregacja,
`src/model.py` → prognoza i metryki, `src/app.py` → aplikacja (czyta tylko gotowe pliki z `outputs/`).

**Baseline** (obowiązkowy punkt odniesienia): średnia krocząca z ostatnich 28 dni, prognoza płaska.

**Model główny**: regresja liniowa na cechach kalendarzowych — sezonowość roczna
(rozwinięcie Fouriera K=2 wg dnia roku), dzień tygodnia (6 dichotomii), flaga epidemii
i sezon (etykieta `Seasonality` z danych).
Bez trendu liniowego: w rolling CV wariant z trendem nie poprawiał MAE, a ekstrapolacja
trendu poza zakres treningu jest ryzykowna.

Epidemia jest **nieznana ex ante**, więc prognoza powstaje w wariantach scenariuszowych.
W aplikacji okres epidemii w oknie prognozy ustawia się **suwakiem dat** (dowolny
przedział w horyzoncie 28 dni) — prognoza reaguje natychmiast, bo jest dokładnie liniowa
względem flagi epidemii (`pred(ep) = pred(0) + [pred(1) − pred(0)] · ep`, bez
przeliczeń w aplikacji). Na walidacji dodatkowo liczymy wariant *rzeczywista epidemia*
(oracle) — w praktyce status epidemii jest ogłaszany z wyprzedzeniem, więc to
realistyczny wariant.

### Metryki (walidacja: ostatnie 28 dni; CV: rolling-origin, 5 foldów × 28 dni)

| model | okno | scenariusz | MAE | RMSE | MAPE | MPE | bias |
|---|---|---|---|---|---|---|---|
| baseline | val | — | 1053.5 | 1417.9 | 14.4% | 6.0% | +226 |
| model | val | bez epidemii | 1284.4 | 1748.6 | 18.3% | 16.1% | +1060 |
| model | val | rzeczywista epidemia | **630.2** | **762.8** | **7.5%** | 3.8% | +307 |
| LightGBM | val | bez epidemii | 1226.5 | 1640.1 | 17.3% | 14.3% | +930 |
| LightGBM | val | rzeczywista epidemia | 670.4 | 807.1 | 8.1% | 3.5% | **+265** |
| baseline | CV (5×28d) | — | 890.9 | 1162.9 | 11.8% | 8.0% | +484 |
| model | CV (5×28d) | bez epidemii | **828.6** | **1109.6** | **11.3%** | 7.3% | +406 |
| LightGBM | CV (5×28d) | bez epidemii | 865.4 | 1167.5 | 11.6% | 7.0% | **+381** |

**Interpretacja (uczciwie):**
- Na tym konkretnym oknie walidacji 6 z 28 dni to dni epidemiczne — scenariusz
  "bez epidemii" jest wtedy systematycznie zbyt optymistyczny (bias +1060), a baseline
  "zbiegiem okoliczności" trafił nisko (jego okno 28-dniowe zawiera dołek epidemiczny
  z końca treningu). Dlatego model przegrywa z baseline'em na val w tym scenariuszu.
- Ze znanym harmonogramem epidemii model jest **o 40% lepszy od baseline'u** (MAE 630
  vs 1053; MAPE 7.5% vs 14.4%).
- W rolling-origin CV (5 foldów w treningu, scenariusz ex ante) model wygrywa
  z baseline'em MAE/RMSE/MAPE — to najuczciwszy obraz średniej jakości.

**Cechy zewnętrzne — zmierzone, dwie decyzje:**
- **Sezon (Seasonality z danych) — ZOSTAŁ, po teście A/B trzech wariantów.**
  Kolumna jest deterministyczna per data, więc współbieżna bez wycieku. Porównanie
  (MAE: val / val-oracle / CV): bez cechy sezonu 1328.1 / 663.1 / 840.7; sezon
  astronomiczny z daty 1298.8 / 652.5 / 835.5; sezon z danych **1284.4 / 630.2 /
  828.6 — najlepszy w każdym oknie**. Detal, który łatwo przeoczyć: kolumna z danych
  to sezony *meteorologiczne* (wiosna od 1. III itd.), a wcześniejsza wersja kodu
  liczyła sezon *astronomiczny* — etykiety różniły się na 162 z 760 dni. Dla dat
  poza danymi (horyzont prognozy) sezon jest wyliczany regułą meteorologiczną,
  zgodną z kolumną 1:1 (zweryfikowane na całym szeregu).
- **Cechy z lagiem 28 dni (zapas, promocja, rabat, cena, cena konkurencji,
  jednostki per kategoria) — ODRZUCONE, choć zaimplementowane** (`make_features(
  ..., use_lag_features=True)`). Lag 28 = horyzont prognozy, więc są znane ex ante
  (bez wycieku), ale zmierzone eksperymentem **pogarszają** wyniki (val-oracle
  MAE 630 → 660). Szereg sieciowy jest tak gładki, że te sygnały to szum, nie
  informacja — spójne z testem udziałów kategorii w EDA (sekcja 5b). Zostają
  w kodzie jako udokumentowany eksperyment i materiał na rozmowę.

**Porównanie z LightGBM:** do projektu dodaliśmy model gradient boostingowy
(LightGBM) na **dokładnie tych samych cechach** — to kontrolowane porównanie klas
modeli przy tej samej informacji. Wynik: LightGBM **nie bije** regresji liniowej
(CV MAE 865.4 vs 828.6; RMSE i MAPE też lepsze u modelu liniowego; LightGBM wygrywa
tylko bias). To spodziewane i pouczające: przy 760 punktach jednego, gładkiego
szeregu elastyczność drzew daje głównie ryzyko przeuczenia, a nie dodatkową wiedzę.
W naszym drugim projekcie (M5 Forecasting: 30 490 szeregów × 1941 dni) ta zależność
jest odwrotna — tam LightGBM ma masę danych i wygrywa. **Wniosek na rozmowę:
wybór modelu podlega skali danych; cechy są ważniejsze niż klasa modelu.**

### Bonus: prognozy dla top-3 produktów

Top-3 produkty wg łącznej sprzedaży: **P0007, P0004, P0009** (kategoria Groceries,
~500 szt./dzień każdy). Ten sam model per produkt (pliki `forecast_products.csv`,
`metrics_products.csv`, wykres i tabela w aplikacji):

| produkt | baseline MAE | model MAE (bez epidemii) |
|---|---|---|
| P0007 | 113.0 | 143.9 |
| P0004 | 103.6 | 105.4 |
| P0009 | 106.3 | **102.5** |

Poziom produktu jest wyraźnie bardziej szumny (CV serii 0.24–0.30 vs 0.18 sieci), a okno
walidacji z epidemicznym początkiem uderza w P0007. W realnym wdrożeniu model
per produkt miałby dodatkowo sezon tygodniowy i cechy lag (`shift(1)`) — w tym
zadaniu celowo zostaje prosty i spójny z modelem sieciowym.

## Aplikacja

`uv run streamlit run src/app.py` — wykres (rzeczywista vs baseline vs model),
tabela metryk, fragmentator zakresu dat i scenariusza epidemii, prognoza na
najbliższe 4 tygodnie oraz sekcja bonusowa z top-3 produktami:

![Aplikacja](outputs/figures/06_app_screenshot.png)

## Co różni to rozwiązanie od typowych notebooków Kaggle

Przejrzeliśmy publiczne rozwiązania tego zbioru i celowo robimy kilka rzeczy inaczej:

- **Podział out-of-time** (ostatnie 28 dni), nie losowy `train_test_split` — losowy
  podział w szeregach czasowych zawyża R² i nie mierzy zdolności prognozowania,
- **Brak współbieżnych regresorów bez scenariusza** — zmienne "z tego samego dnia"
  (stan magazynu, cena) są nieznane w momencie prognozy; wysokie R² przy losowym
  podziale, widoczne w niektórych publicznych notebookach, jest artefaktem wycieku,
- **Baseline jako punkt odniesienia** — bez tego nie wiadomo, czy model w ogóle
  bije średnią kroczącą,
- **MAPE i bias** obok MAE/RMSE — bias pokazuje systematyczne przeszacowanie,
  którego MAE nie widzi,
- **Aplikacja webowa z fragmentatorem** zamiast statycznego notebooka,
- **Udokumentowane decyzje** — np. udziały produktów/kategorii w sprzedaży
  odrzucone po teście (EDA §5b: ~2% wyjaśnionej wariancji ponad perzyystencję).

## Struktura

```
├── src/data.py       # pobieranie (kagglehub), kontrola jakości, agregacja, split
├── src/model.py      # baseline + model, scenariusze epidemii, metryki, CV, bonus top-3
├── src/app.py        # Streamlit: wykres, tabela metryk, fragmentator, bonus
├── notebooks/EDA.ipynb
├── outputs/          # daily_sales, forecast(_products), metrics(_products) [gitignored]
├── outputs/figures/  # wykresy EDA + screenshot aplikacji
└── priv/             # notatki robocze (niewidoczne w repo)
```
