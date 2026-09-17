"""Prognoza dziennej sprzedaży sieci (Units Sold) na 28 dni w przód.

Architektura (pkt 3-4 zadania):
  * baseline - średnia krocząca z ostatnich 28 dni, prognoza płaska;
  * model główny - regresja liniowa na cechach kalendarzowych:
      - sezonowość roczna: rozwinięcie Fouriera (K=2) wg dnia roku,
      - dzień tygodnia: 6 dichotomii,
      - epidemia: flaga dnia (efekt -36% z EDA).
    Bez trendu liniowego: w rolling CV wariant z trendem nie poprawiał MAE,
    a ekstrapolacja trendu poza zakres treningu jest ryzykowna.

Epidemia jest zmienną NIEZNANĄ ex ante - dlatego prognoza powstaje w wariantach
scenariuszowych: "bez epidemii" (domyślny) i "epidemia trwa". Na oknie walidacji
dodatkowo liczymy wariant "rzeczywista epidemia" (oracle) wyłącznie do oceny,
jak wyglądałaby prognoza ze znanym harmonogramem epidemii.

Split: ostatnie 28 dni = walidacja (zgodnie z zadaniem), reszta = trening.
Prognoza produkcyjna: 28 dni po końcu danych, model dopasowany na pełnej historii.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HORIZON = 28          # horyzont prognozy [dni] - zadanie: 4 tygodnie
FOURIER_K = 2         # pary harmonicznych sezonu rocznego
ROLLING_WINDOW = 28   # okno baseline'u (średnia krocząca)
CV_FOLDS = 5          # foldy rolling-origin CV (po 28 dni) w obrębie treningu

OUTPUTS_DIR = Path("outputs")


def load_daily(path: Path = OUTPUTS_DIR / "daily_sales.csv") -> pd.DataFrame:
    """Wczytuje zagregowany szereg dzienny z outputs/daily_sales.csv."""
    daily = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    missing = {"units", "epidemic"} - set(daily.columns)
    if missing:
        raise ValueError(f"Brak kolumn w {path}: {missing} - uruchom najpierw src/data.py")
    return daily


def fourier_terms(index: pd.DatetimeIndex, k: int = FOURIER_K) -> dict:
    """Rozwinięcie Fouriera sezonu rocznego wg dnia roku (k par sin/cos)."""
    doy = index.dayofyear.to_numpy()
    terms = {}
    for j in range(1, k + 1):
        terms[f"sin{j}"] = np.sin(2 * np.pi * j * doy / 365.25)
        terms[f"cos{j}"] = np.cos(2 * np.pi * j * doy / 365.25)
    return terms


def make_features(index: pd.DatetimeIndex, epidemic) -> pd.DataFrame:
    """Macierz cech: sezon roczny (Fourier), dzień tygodnia, flaga epidemii."""
    X = pd.DataFrame(fourier_terms(index), index=index)
    for d in range(6):
        X[f"dow{d}"] = (index.dayofweek == d).astype(int)
    X["epidemic"] = np.asarray(epidemic, dtype=float)
    return X


def baseline_forecast(train_units: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline: średnia krocząca (ROLLING_WINDOW dni) z końca treningu, prognoza płaska."""
    level = train_units.rolling(ROLLING_WINDOW).mean().iloc[-1]
    return pd.Series(level, index=future_index, name="baseline")


def fit_model(daily_chunk: pd.DataFrame) -> LinearRegression:
    """Dopasowuje regresję liniową na podanym fragmencie danych."""
    X = make_features(daily_chunk.index, daily_chunk["epidemic"])
    return LinearRegression().fit(X, daily_chunk["units"])


def predict(model: LinearRegression, index: pd.DatetimeIndex, epidemic) -> pd.Series:
    """Prognoza modelu dla podanych dat i scenariusza epidemii (flaga lub seria)."""
    X = make_features(index, epidemic)
    return pd.Series(model.predict(X), index=index, name="model")


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
    base_scores, model_scores = [], []
    for cut in starts:
        tr, te = daily.iloc[:cut], daily.iloc[cut:cut + horizon]
        base_scores.append(score(baseline_forecast(tr["units"], te.index), te["units"]))
        model_scores.append(score(predict(fit_model(tr), te.index, epidemic=0.0),
                                  te["units"]))
    # średnia po foldach (RMSE: średnia z RMSE foldów, nie pooled)
    def mean_scores(scores: list) -> dict:
        return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    return {"baseline": mean_scores(base_scores),
            "model_bez_epidemii": mean_scores(model_scores)}


def main() -> None:
    daily = load_daily()
    train, val = daily.iloc[:-HORIZON], daily.iloc[-HORIZON:]

    # --- walidacja: fit na treningu, prognoza ostatnich 28 dni ---
    model = fit_model(train)
    val_base = baseline_forecast(train["units"], val.index)
    val_ep0 = predict(model, val.index, epidemic=0.0)                    # scenariusz domyślny
    val_ep1 = predict(model, val.index, epidemic=1.0)                    # scenariusz "trwa"
    val_ep_actual = predict(model, val.index, epidemic=val["epidemic"])  # oracle (tylko ocena)

    # --- prognoza produkcyjna: 28 dni po końcu danych, fit na pełnej historii ---
    model_full = fit_model(daily)
    future_index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1),
                                 periods=HORIZON, freq="D")
    fut_base = baseline_forecast(daily["units"], future_index)
    fut_ep0 = predict(model_full, future_index, epidemic=0.0)
    fut_ep1 = predict(model_full, future_index, epidemic=1.0)

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
    ]
    cv = rolling_origin_cv(train)
    rows.append({"model": "baseline", "window": "cv5_mean", "scenario": "-", **cv["baseline"]})
    rows.append({"model": "model", "window": "cv5_mean", "scenario": "bez_epidemii",
                 **cv["model_bez_epidemii"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    print(metrics.round(1).to_string(index=False))
    print(f"\nZapisano: {OUTPUTS_DIR / 'forecast.csv'} i {OUTPUTS_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
