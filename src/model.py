"""Prognoza dziennej sprzedaży sieci (Units Sold) na 28 dni w przód.

Architektura (pkt 3-4 zadania):
  * baseline - średnia krocząca z ostatnich 28 dni, prognoza płaska;
  * model główny - regresja liniowa na cechach kalendarzowych:
      - sezonowość roczna: rozwinięcie Fouriera (K=2) wg dnia roku,
      - dzień tygodnia: 6 dichotomii,
      - epidemia: flaga dnia (efekt -36% z EDA),
      - sezon kalendarzowy (Seasonality): współbieżny bez wycieku, poprawia MAE.
    Opcjonalnie (use_lag_features=True): cechy zewnętrzne z lagiem 28 (zapas,
    promocja, rabat, ceny, jednostki per kategoria) - zmierzone i ODRZUCONE
    (pogarszają wyniki), w kodzie jako udokumentowany eksperyment.
    Bez trendu liniowego: w rolling CV wariant z trendem nie poprawiał MAE,
    a ekstrapolacja trendu poza zakres treningu jest ryzykowna.
  * model porównawczy - LightGBM na DOKŁADNIE TYCH SAMYCH cechach (uczciwe
    porównanie: czy elastyczny model bije liniowy przy 760 punktach jednego
    szeregu). Parametry ostrożne (płytkie drzewa), by nie przeuczyć małych danych.

Epidemia jest zmienną NIEZNANĄ ex ante - dlatego prognoza powstaje w wariantach
scenariuszowych: "bez epidemii" (domyślny) i "epidemia trwa". Na oknie walidacji
dodatkowo liczymy wariant "rzeczywista epidemia" (oracle) wyłącznie do oceny,
jak wyglądałaby prognoza ze znanym harmonogramem epidemii.

Split: ostatnie 28 dni = walidacja (zgodnie z zadaniem), reszta = trening.
Prognoza produkcyjna: 28 dni po końcu danych, model dopasowany na pełnej historii.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from src.data import load_raw          # uruchomienie z katalogu projektu
except ImportError:                        # noqa: F401
    from data import load_raw              # uruchomienie: python src/model.py

HORIZON = 28          # horyzont prognozy [dni] - zadanie: 4 tygodnie
FOURIER_K = 2         # pary harmonicznych sezonu rocznego
ROLLING_WINDOW = 28   # okno baseline'u (średnia krocząca)
CV_FOLDS = 5          # foldy rolling-origin CV (po 28 dni) w obrębie treningu
FEATURE_LAG = 28      # lag cech zewnętrznych (= horyzont => znane ex ante)
BONUS_TOP_N = 3       # bonus: prognoza dla top N produktów

OUTPUTS_DIR = Path("outputs")


def load_daily(path: Path = OUTPUTS_DIR / "daily_sales.csv") -> pd.DataFrame:
    """Wczytuje zagregowany szereg dzienny z outputs/daily_sales.csv."""
    daily = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    missing = {"units", "epidemic"} - set(daily.columns)
    if missing:
        raise ValueError(f"Brak kolumn w {path}: {missing} - uruchom najpierw src/data.py")
    return daily


EXTRA_COLS = ["inventory", "promotion", "discount", "price", "competitor_price"]


def load_daily_enriched(path: Path = OUTPUTS_DIR / "daily_sales.csv") -> pd.DataFrame:
    """Szereg dzienny sieci + jednostki per kategoria (do cech lag-28)."""
    daily = load_daily(path)
    raw = load_raw()
    cat = (raw.groupby(["Date", "Category"])["Units Sold"].sum()
              .unstack().add_prefix("cat_").rename(columns=lambda c: c.lower()))
    return daily.join(cat, how="left")


def fourier_terms(index: pd.DatetimeIndex, k: int = FOURIER_K) -> dict:
    """Rozwinięcie Fouriera sezonu rocznego wg dnia roku (k par sin/cos)."""
    doy = index.dayofyear.to_numpy()
    terms = {}
    for j in range(1, k + 1):
        terms[f"sin{j}"] = np.sin(2 * np.pi * j * doy / 365.25)
        terms[f"cos{j}"] = np.cos(2 * np.pi * j * doy / 365.25)
    return terms


def make_features(index: pd.DatetimeIndex, epidemic,
                  history: pd.DataFrame | None = None,
                  use_lag_features: bool = False) -> pd.DataFrame:
    """Macierz cech: sezon roczny (Fourier), dzień tygodnia, flaga epidemii.

    Zawsze dochodzi `Seasonality` - kalendarzowy atrybut dnia (deterministyczny
    wg daty, więc współbieżny bez wycieku). W teście z EDA poprawia MAE
    (val 663 -> 652).

    Jeśli dodatkowo podano `history` i `use_lag_features=True`, dochodzą cechy
    z lagiem FEATURE_LAG (= horyzont prognozy, więc znane ex ante): zapas,
    promocja, rabat, cena, cena konkurencji, jednostki per kategoria.
    DOMYŚLNIE WYŁĄCZONE: zmierzone w eksperymencie pogarszają wyniki (val-oracle
    MAE 652 -> 705) - szereg sieciowy jest tak gładki, że to szum, nie informacja.
    Zostawione w kodzie jako udokumentowany eksperyment (wzgl. EDA sekcja 5b).
    """
    X = pd.DataFrame(fourier_terms(index), index=index)
    for d in range(6):
        X[f"dow{d}"] = (index.dayofweek == d).astype(int)
    X["epidemic"] = np.asarray(epidemic, dtype=float)
    season = pd.Series(index, index=index).map(_season_of)
    for s in ("Autumn", "Spring", "Summer"):      # Winter = referencja
        X[f"season_{s.lower()}"] = (season == s).astype(float)
    if history is None or not use_lag_features:
        return X

    idx_ext = history.index.union(index)
    ext_cols = [c for c in EXTRA_COLS if c in history.columns]
    if not ext_cols:                      # szereg produktu (bonus): tylko cechy bazowe
        return X
    ext = history[ext_cols].reindex(idx_ext).shift(FEATURE_LAG).loc[index]
    ext.columns = [f"{c}_lag{FEATURE_LAG}" for c in ext_cols]
    cat_cols = [c for c in history.columns if c.startswith("cat_")]
    if cat_cols:
        cat = history[cat_cols].reindex(idx_ext).shift(FEATURE_LAG).loc[index]
        cat.columns = [f"{c}_lag{FEATURE_LAG}" for c in cat_cols]
        ext = pd.concat([ext, cat], axis=1)
    return pd.concat([X, ext], axis=1)


def _season_of(date) -> str:
    """Sezon astronomiczny z daty (fallback, gdy brak kolumny Seasonality)."""
    mth, day = date.month, date.day
    if (mth, day) >= (3, 20) and (mth, day) < (6, 21): return "Spring"
    if (mth, day) >= (6, 21) and (mth, day) < (9, 23): return "Summer"
    if (mth, day) >= (9, 23) and (mth, day) < (12, 21): return "Autumn"
    return "Winter"


def baseline_forecast(train_units: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline: średnia krocząca (ROLLING_WINDOW dni) z końca treningu, prognoza płaska."""
    level = train_units.rolling(ROLLING_WINDOW).mean().iloc[-1]
    return pd.Series(level, index=future_index, name="baseline")


def fit_model(daily_chunk: pd.DataFrame,
              use_lag_features: bool = False) -> LinearRegression:
    """Dopasowuje regresję liniową na podanym fragmencie danych."""
    X = make_features(daily_chunk.index, daily_chunk["epidemic"], history=daily_chunk,
                      use_lag_features=use_lag_features)
    ok = X.notna().all(axis=1)          # pierwsze FEATURE_LAG dni nie mają lagów
    return LinearRegression().fit(X[ok], daily_chunk.loc[ok, "units"])


def predict(model: LinearRegression, index: pd.DatetimeIndex, epidemic,
            history: pd.DataFrame | None = None,
            use_lag_features: bool = False) -> pd.Series:
    """Prognoza modelu dla podanych dat i scenariusza epidemii (flaga lub seria)."""
    X = make_features(index, epidemic, history=history,
                      use_lag_features=use_lag_features)
    return pd.Series(model.predict(X), index=index, name="model")


LGBM_PARAMS = dict(
    n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20,
    subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=1, verbose=-1,
)


def fit_lgbm(daily_chunk: pd.DataFrame, use_lag_features: bool = False) -> lgb.LGBMRegressor:
    """Dopasowuje LightGBM na TYCH SAMYCH cechach co regresję liniową.

    Cel: uczciwe porównanie klas modeli przy tej samej informacji (sezon roczny,
    dzień tygodnia, epidemia). Parametry celowo konserwatywne - 760 punktów
    jednego szeregu to mało danych na głębokie drzewa.
    """
    X = make_features(daily_chunk.index, daily_chunk["epidemic"], history=daily_chunk,
                      use_lag_features=use_lag_features)
    ok = X.notna().all(axis=1)
    return lgb.LGBMRegressor(**LGBM_PARAMS).fit(X[ok], daily_chunk.loc[ok, "units"])


def predict_lgbm(model: lgb.LGBMRegressor, index: pd.DatetimeIndex, epidemic,
                 history: pd.DataFrame | None = None,
                 use_lag_features: bool = False) -> pd.Series:
    """Prognoza LightGBM dla podanych dat i scenariusza epidemii."""
    X = make_features(index, epidemic, history=history,
                      use_lag_features=use_lag_features)
    return pd.Series(model.predict(X), index=index, name="lgbm").clip(lower=0)


def score(pred: pd.Series, actual: pd.Series) -> dict:
    """Metryki: MAE, RMSE, MAPE (%), bias (szt.; dodatni = przeszacowanie)."""
    err = pred - actual
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE": float((np.abs(err) / actual).mean() * 100),
        "bias": float(err.mean()),
    }


def rolling_origin_cv(daily: pd.DataFrame, n_folds: int = CV_FOLDS,
                      horizon: int = HORIZON) -> dict:
    """Rolling-origin CV w obrębie treningu (bez dotykania walidacji).

    n_folds kolejnych okien po `horizon` dni, kończących się `horizon` dni
    przed walidacją. Scenariusz epidemii w prognozie: brak (uczciwy, ex ante).
    Zwraca {nazwa_modelu: {"MAE": średnia po foldach}}.
    """
    starts = [len(daily) - 2 * horizon - i * horizon for i in range(n_folds, 0, -1)]
    base_scores, model_scores, lgbm_scores = [], [], []
    for cut in starts:
        tr, te = daily.iloc[:cut], daily.iloc[cut:cut + horizon]
        base_scores.append(score(baseline_forecast(tr["units"], te.index), te["units"]))
        model_scores.append(score(predict(fit_model(tr), te.index, epidemic=0.0,
                                          history=tr), te["units"]))
        lgbm_scores.append(score(predict_lgbm(fit_lgbm(tr), te.index, epidemic=0.0,
                                              history=tr), te["units"]))
    # średnia po foldach (RMSE: średnia z RMSE foldów, nie pooled)
    def mean_scores(scores: list) -> dict:
        return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    return {"baseline": mean_scores(base_scores),
            "model_bez_epidemii": mean_scores(model_scores),
            "lgbm_bez_epidemii": mean_scores(lgbm_scores)}


def product_series(raw: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """Dzienny szereg jednego produktu (suma po sklepach) z flagą epidemii."""
    sub = raw[raw["Product ID"] == product_id]
    return (sub.groupby("Date")
               .agg(units=("Units Sold", "sum"), epidemic=("Epidemic", "max"))
               .sort_index())


def run_bonus(daily_full: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bonus: prognozy dla top-3 produktów wg łącznej sprzedaży.

    Dla każdego produktu: ten sam model (Fourier + DOW + epidemia), walidacja
    na ostatnich `horizon` dniach i prognoza produkcyjna (scenariusz bez
    epidemii). Zwraca (forecast_products, metrics_products).
    """
    raw = load_raw()
    top = raw.groupby("Product ID")["Units Sold"].sum().nlargest(BONUS_TOP_N).index
    fc_parts, met_rows = [], []
    for pid in top:
        s = product_series(raw, pid)
        train, val = s.iloc[:-horizon], s.iloc[-horizon:]
        model = fit_model(train)
        val_ep0 = predict(model, val.index, epidemic=0.0)
        val_base = baseline_forecast(train["units"], val.index)
        model_full = fit_model(s)
        fut_idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1),
                                periods=horizon, freq="D")
        parts = [
            pd.DataFrame({"product_id": pid, "split": "val", "actual": val["units"],
                          "baseline": val_base, "model_ep0": val_ep0}),
            pd.DataFrame({"product_id": pid, "split": "future", "actual": np.nan,
                          "baseline": baseline_forecast(s["units"], fut_idx),
                          "model_ep0": predict(model_full, fut_idx, epidemic=0.0)},
                         index=fut_idx),
        ]
        fc_parts.extend(parts)
        met_rows.append({"product_id": pid, "model": "baseline", "window": "val",
                         **score(val_base, val["units"])})
        met_rows.append({"product_id": pid, "model": "model", "window": "val",
                         "scenario": "bez_epidemii", **score(val_ep0, val["units"])})
    fc = pd.concat(fc_parts)
    fc.index.name = "Date"
    return fc, pd.DataFrame(met_rows)


def main() -> None:
    daily = load_daily_enriched()
    train, val = daily.iloc[:-HORIZON], daily.iloc[-HORIZON:]

    # --- walidacja: fit na treningu, prognoza ostatnich 28 dni ---
    model = fit_model(train)
    val_base = baseline_forecast(train["units"], val.index)
    val_ep0 = predict(model, val.index, epidemic=0.0, history=train)     # scenariusz domyślny
    val_ep1 = predict(model, val.index, epidemic=1.0, history=train)     # scenariusz "trwa"
    val_ep_actual = predict(model, val.index, epidemic=val["epidemic"], history=train)

    # --- LightGBM: porównanie (te same cechy, te same scenariusze) ---
    lgbm = fit_lgbm(train)
    lgbm_ep0 = predict_lgbm(lgbm, val.index, epidemic=0.0, history=train)
    lgbm_ep1 = predict_lgbm(lgbm, val.index, epidemic=1.0, history=train)
    lgbm_ep_actual = predict_lgbm(lgbm, val.index, epidemic=val["epidemic"], history=train)

    # --- prognoza produkcyjna: 28 dni po końcu danych, fit na pełnej historii ---
    model_full = fit_model(daily)
    future_index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1),
                                 periods=HORIZON, freq="D")
    fut_base = baseline_forecast(daily["units"], future_index)
    fut_ep0 = predict(model_full, future_index, epidemic=0.0, history=daily)
    fut_ep1 = predict(model_full, future_index, epidemic=1.0, history=daily)
    lgbm_full = fit_lgbm(daily)
    fut_lgbm_ep0 = predict_lgbm(lgbm_full, future_index, epidemic=0.0, history=daily)
    fut_lgbm_ep1 = predict_lgbm(lgbm_full, future_index, epidemic=1.0, history=daily)

    # --- outputs/forecast.csv (kontrakt aplikacji) ---
    out = pd.concat([
        daily[["units", "epidemic"]].rename(
            columns={"units": "actual", "epidemic": "epidemic_actual"}),
        pd.DataFrame(index=future_index),   # przyszłość: bez actual/epidemic_actual
    ])
    out["split"] = (["train"] * len(train) + ["val"] * HORIZON + ["future"] * HORIZON)
    out["baseline"] = np.nan
    out.loc[val.index, "baseline"] = val_base.values
    out.loc[future_index, "baseline"] = fut_base.values
    out["model_ep0"] = np.nan
    out.loc[val.index, "model_ep0"] = val_ep0.values
    out.loc[future_index, "model_ep0"] = fut_ep0.values
    out["model_ep1"] = np.nan
    out.loc[val.index, "model_ep1"] = val_ep1.values
    out.loc[future_index, "model_ep1"] = fut_ep1.values
    # epidemia wg danych (tylko walidacja): linia modelu do wykresu w aplikacji
    out["model_ep_actual"] = np.nan
    out.loc[val.index, "model_ep_actual"] = val_ep_actual.values
    out["lgbm_ep0"] = np.nan
    out.loc[val.index, "lgbm_ep0"] = lgbm_ep0.values
    out.loc[future_index, "lgbm_ep0"] = fut_lgbm_ep0.values
    out["lgbm_ep1"] = np.nan
    out.loc[val.index, "lgbm_ep1"] = lgbm_ep1.values
    out.loc[future_index, "lgbm_ep1"] = fut_lgbm_ep1.values
    out["lgbm_ep_actual"] = np.nan
    out.loc[val.index, "lgbm_ep_actual"] = lgbm_ep_actual.values
    out.index.name = "Date"
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS_DIR / "forecast.csv")

    # --- outputs/metrics.csv (długi format: model × okno × scenariusz) ---
    rows = [
        {"model": "baseline", "window": "val", "scenario": "-",
         **score(val_base, val["units"])},
        {"model": "model", "window": "val", "scenario": "bez_epidemii",
         **score(val_ep0, val["units"])},
        {"model": "model", "window": "val", "scenario": "epidemia_rzeczywista",
         **score(val_ep_actual, val["units"])},
        {"model": "lgbm", "window": "val", "scenario": "bez_epidemii",
         **score(lgbm_ep0, val["units"])},
        {"model": "lgbm", "window": "val", "scenario": "epidemia_rzeczywista",
         **score(lgbm_ep_actual, val["units"])},
    ]
    cv = rolling_origin_cv(train)
    rows.append({"model": "baseline", "window": "cv5_mean", "scenario": "-", **cv["baseline"]})
    rows.append({"model": "model", "window": "cv5_mean", "scenario": "bez_epidemii",
                 **cv["model_bez_epidemii"]})
    rows.append({"model": "lgbm", "window": "cv5_mean", "scenario": "bez_epidemii",
                 **cv["lgbm_bez_epidemii"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    # --- bonus: prognozy top-3 produktów ---
    fc_products, met_products = run_bonus(daily)
    fc_products.to_csv(OUTPUTS_DIR / "forecast_products.csv", index_label="Date")
    met_products.to_csv(OUTPUTS_DIR / "metrics_products.csv", index=False)

    print(metrics.round(1).to_string(index=False))
    print("\nBonus - top-3 produkty (walidacja, MAE):")
    print(met_products.round(1).to_string(index=False))
    print(f"\nZapisano: {OUTPUTS_DIR / 'forecast.csv'}, {OUTPUTS_DIR / 'metrics.csv'}, "
          f"{OUTPUTS_DIR / 'forecast_products.csv'}, {OUTPUTS_DIR / 'metrics_products.csv'}")



if __name__ == "__main__":
    main()
