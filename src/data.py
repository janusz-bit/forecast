"""Pobieranie, wczytanie i agregacja danych (Demand Forecasting Dataset).

Uruchomienie:  uv run python src/data.py
Wynik:         outputs/daily_sales.csv
"""
from pathlib import Path

import kagglehub
import pandas as pd

CURRENT_FILE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HORIZON_DAYS = 28  # zbiór walidacyjny: ostatnie 4 tygodnie

# Forecast do ściągnięcia
DATASET_HANDLE = "raminhuseyn/demand-forecasting-dataset"


def load_raw() -> pd.DataFrame:
    """Pobiera dane przez kagglehub, wczytuje CSV, konwertuje Date, sprawdzanie danych."""
    path = kagglehub.dataset_download(DATASET_HANDLE)
    csv_path = Path(path) / "demand_forecasting.csv"

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    # Sprawdzanie danych
    assert not df.isna().any().any(), "Dane zawierają braki!"
    key_cols = ["Date", "Store ID", "Product ID"]
    assert not df.duplicated(key_cols).any(), "Duplikaty klucza (Date, Store, Product)!"
    full_days = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    assert df["Date"].nunique() == len(full_days), "Luki w kalendarzu dat!"
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agreguje sprzedaż do dziennego szeregu sieciowego (suma po sklepach i produktach).

    Zwraca DataFrame z kolumną `units`; zmienne zewnętrzne agregowane osobno
    (epidemia: max — wystarczy 1 sklep, żeby zakłócić sieć; reszta: średnia).
    Demand: suma (cenzurowany popyt); demand - units = utracona sprzedaż
    (lost sales), rośnie przy niskim zapasie — kandydat na cechę laga (patrz EDA).
    """
    daily = df.groupby("Date").agg(
        units=("Units Sold", "sum"), # ile sprzedano
        demand=("Demand", "sum"), # popyt (cenzurowany zapasem): suma

        epidemic=("Epidemic", "max"), # czy była epidemia
        promotion=("Promotion", "mean"), # Czy była "naklejka promocja"
        discount=("Discount", "mean"), # Jaki była obniżka
        # Trzeba przemyśleć czy średnia ważona (dane historyczne, wyliczyć wagi koszyka, czy w inny sposób)
        price=("Price", "mean"), # Cena
        competitor_price=("Competitor Pricing", "mean"), # Cena u konkurencji
        inventory=("Inventory Level", "sum"), # Liczba na sklepie
    )
    daily.index.name = "Date"
    return daily


def train_val_split(series: pd.DataFrame, horizon: int = HORIZON_DAYS):
    """Podział czysto czasowy: ostatnie `horizon` dni → walidacja, reszta → trening."""
    train, val = series.iloc[:-horizon], series.iloc[-horizon:]
    return train, val


def main() -> None:
    df = load_raw()
    daily = aggregate_daily(df)
    train, val = train_val_split(daily)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    daily.to_csv(OUTPUTS_DIR / "daily_sales.csv")

    print(f"Zbiór surowy: {len(df):,} wierszy  (kagglehub: {DATASET_HANDLE})")
    print(f"Szereg dzienny: {len(daily)} dni  "
          f"({daily.index.min().date()} → {daily.index.max().date()})")
    print(f"Trening: {len(train)} dni  |  Walidacja: {len(val)} dni "
          f"({val.index.min().date()} → {val.index.max().date()})")
    print(f"Zapisano: {OUTPUTS_DIR / 'daily_sales.csv'}")


if __name__ == "__main__":
    main()
