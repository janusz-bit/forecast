"""Aplikacja Streamlit: prognoza sprzedaży sieci na 4 tygodnie.

Tylko czyta wyliczone pliki z outputs/ (kontrakt: forecast.csv, metrics.csv,
daily_sales.csv) - nic nie liczy i nie trenuje przy starcie.

Uruchomienie: uv run streamlit run src/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUTPUTS_DIR = Path("outputs")

st.set_page_config(page_title="Prognoza sprzedaży", layout="wide")
st.title("Prognoza sprzedaży sieci detalicznej")
st.caption("Dane: Kaggle *Demand forecasting dataset* | horyzont: 28 dni | "
           "prognoza na poziomie całej sieci")


@st.cache_data
def load_forecast() -> pd.DataFrame:
    """Wczytuje forecast.csv (cache - odczyt raz na uruchomienie aplikacji)."""
    df = pd.read_csv(OUTPUTS_DIR / "forecast.csv", parse_dates=["Date"], index_col="Date")
    return df.sort_index()


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Wczytuje metrics.csv (długi format: model × okno × scenariusz)."""
    return pd.read_csv(OUTPUTS_DIR / "metrics.csv")


forecast = load_forecast()
metrics = load_metrics()


@st.cache_data
def load_products() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bonus: forecast_products.csv i metrics_products.csv (top-3 produkty)."""
    fc = pd.read_csv(OUTPUTS_DIR / "forecast_products.csv",
                     parse_dates=["Date"], index_col="Date")
    met = pd.read_csv(OUTPUTS_DIR / "metrics_products.csv")
    return fc, met


products_fc, products_met = load_products()

st.sidebar.header("Fragmentator")

# --- zakres dat ---
hist_start = forecast.index.min().date()
end_date = forecast.index.max().date()
default_start = end_date - pd.Timedelta(days=90)
date_range = st.sidebar.date_input(
    "Zakres dat",
    value=(default_start, end_date),
    min_value=hist_start,
    max_value=end_date,
)

# --- dynamiczny przesuwak: okres epidemii w oknie prognozy ---
future_df = forecast[forecast["split"] == "future"]
fut_min, fut_max = future_df.index.min().date(), future_df.index.max().date()

ep_on = st.sidebar.checkbox("Epidemia w oknie prognozy", value=False,
                            help="Zaznacz, jeśli w oknie prognozy (28 dni po końcu "
                                 "danych) ma obowiązywać epidemia — zakres dni "
                                 "wybierzesz poniżej. Model reaguje natychmiast "
                                 "(prognoza jest liniowa względem flagi epidemii).")
if ep_on:
    ep_range = st.sidebar.slider(
        "Dni epidemiczne",
        min_value=fut_min, max_value=fut_max,
        value=(fut_min, min(pd.Timestamp(fut_min) + pd.Timedelta(days=5), pd.Timestamp(fut_max)).date()),
        format="DD.MM.YYYY",
    )
else:
    ep_range = None

n_ep_days = 0 if not ep_on else (ep_range[1] - ep_range[0]).days + 1
st.sidebar.caption(f"Epidemia obejmie **{n_ep_days}** z 28 dni prognozy.")

# --- drugi przesuwak: epidemia w oknie WALIDACJI (własny scenariusz "co jeśli") ---
val_df = forecast[forecast["split"] == "val"]
val_min, val_max = val_df.index.min().date(), val_df.index.max().date()

val_ep_on = st.sidebar.checkbox("Epidemia w walidacji (własny scenariusz)",
                                value=False,
                                help="Ustaw własny przebieg epidemii w 28 dniach "
                                     "walidacji i zobacz, jak zmienia się błąd "
                                     "prognozy modelu (metryki liczone na żywo "
                                     "z gotowych prognoz - model nie jest uczony "
                                     "ponownie).")
if val_ep_on:
    val_ep_range = st.sidebar.slider(
        "Dni epidemiczne (walidacja)",
        min_value=val_min, max_value=val_max,
        value=(val_min, min(pd.Timestamp(val_min) + pd.Timedelta(days=5), pd.Timestamp(val_max)).date()),
        format="DD.MM.YYYY",
    )
else:
    val_ep_range = None

val_n_ep = 0 if not val_ep_on else (val_ep_range[1] - val_ep_range[0]).days + 1
st.sidebar.caption(f"Walidacja: epidemia w **{val_n_ep}** z 28 dni (wg danych: 6).")

# --- filtr dat ---
if len(date_range) == 2:
    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    view = forecast.loc[lo:hi]
else:
    view = forecast

# --- wykres ---
fig = go.Figure()
hist = view[view["actual"].notna()]
fig.add_trace(go.Scattergl(  # type: ignore[attr-defined]
    x=hist.index, y=hist["actual"], name="Rzeczywista sprzedaż",
    line=dict(color="#38BDF8", width=1.5)))
fig.add_vrect(
    x0=hist.index.max() - pd.Timedelta(days=28), x1=hist.index.max(),
    fillcolor="rgba(241,245,249,0.10)", line_width=0,
    annotation_text="walidacja", annotation_position="top left")

fut = view[view["baseline"].notna()]
fig.add_trace(go.Scatter(
    x=fut.index, y=fut["baseline"], name="Baseline (średnia krocząca 28d)",
    line=dict(color="#FBBF24", width=1.5, dash="dot")))

# linia modelu: walidacja = epidemia wg danych; przyszłość = przesuwak
fut_all = forecast[forecast["split"] == "future"]
ep_flag = pd.Series(0.0, index=fut_all.index)
if ep_on and ep_range[0] <= ep_range[1]:
    ep_flag[(ep_flag.index >= pd.Timestamp(ep_range[0]))
            & (ep_flag.index <= pd.Timestamp(ep_range[1]))] = 1.0
# model liniowy: pred(ep) = pred(ep=0) + [pred(ep=1) - pred(ep=0)] * ep
model_dynamic = (fut_all["model_ep0"]
                 + (fut_all["model_ep1"] - fut_all["model_ep0"]) * ep_flag)
fig.add_trace(go.Scatter(
    x=fut.index, y=fut["model_ep0"], name="Model (bez epidemii)",
    line=dict(color="#94A3B8", width=1.2, dash="dash"), opacity=0.7))
fig.add_trace(go.Scatter(
    x=view[view["model_ep_actual"].notna()].index,
    y=view["model_ep_actual"].dropna(), name="Model (walidacja, epidemia wg danych)",
    line=dict(color="#F472B6", width=2)))

# walidacja wg WŁASNEGO scenariusza użytkownika (ta sama sztuczka liniowości)
val_flag = pd.Series(0.0, index=val_df.index)
if val_ep_on and val_ep_range[0] <= val_ep_range[1]:
    val_flag[(val_flag.index >= pd.Timestamp(val_ep_range[0]))
             & (val_flag.index <= pd.Timestamp(val_ep_range[1]))] = 1.0
val_model_user = (val_df["model_ep0"]
                  + (val_df["model_ep1"] - val_df["model_ep0"]) * val_flag)
fig.add_trace(go.Scatter(
    x=val_df.index, y=val_model_user,
    name="Model (walidacja: TWÓJ scenariusz epidemii)",
    line=dict(color="#FB7185", width=2, dash="dashdot")))
fig.add_trace(go.Scatter(
    x=fut_all.index, y=model_dynamic,
    name="Model (prognoza: przesuwak epidemii)",
    line=dict(color="#F472B6", width=2)))

fig.update_layout(
    hovermode="x unified",
    height=420, margin=dict(l=10, r=10, t=30, b=10),
    yaxis_title="sztuki / dzień", xaxis_title=None,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F1F5F9"),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, width="stretch")

# --- tabela metryk ---
st.subheader("Metryki jakości prognozy (walidacja: ostatnie 28 dni)")
met_view = metrics.copy()
for col in ("MAE", "RMSE", "MAPE", "MPE", "bias"):
    met_view[col] = met_view[col].round(1)
st.dataframe(met_view, width="stretch", hide_index=True)

# metryki WŁASNEGO scenariusza epidemii w walidacji (na żywo, bez uczenia modelu)
if val_ep_on:
    err = val_model_user - val_df["actual"]
    mae_u = err.abs().mean()
    rmse_u = (err ** 2).mean() ** 0.5
    mape_u = (err.abs() / val_df["actual"]).mean() * 100
    bias_u = err.mean()
    mae_actual = (val_df["model_ep_actual"] - val_df["actual"]).abs().mean()
    delta = mae_u - mae_actual
    if abs(delta) < 0.5:
        verdict = ("to dokładnie scenariusz 'epidemia wg danych' — trafiłeś w "
                   "rzeczywisty przebieg epidemii")
    else:
        verdict = (f"to {'gorzej' if delta > 0 else 'lepiej'} o {abs(delta):,.0f} "
                   f"szt./dzień niż scenariusz 'epidemia wg danych' "
                   f"(MAE {mae_actual:,.0f}). Przesuń zakres, aby błąd zminimalizować.")
    st.caption(
        f"**Twój scenariusz** ({val_n_ep} dni epidemii): MAE **{mae_u:,.0f}**, "
        f"RMSE **{rmse_u:,.0f}**, MAPE **{mape_u:.1f}%**, bias **{bias_u:+,.0f}** — "
        f"{verdict}"
    )

with st.expander("Jak czytać metryki?"):
    st.markdown(
        "- **MAE** - średni błąd bezwzględny (sztuki/dzień);\n"
        "- **RMSE** - jak MAE, ale duże pomyłki ważą bardziej (szt./dzień);\n"
        "- **MAPE** - średni błąd względny (%);\n"
        "- **MPE** - bias wyrażony w % (dodatni = przeszacowanie);\n"
        "- **bias** - średni błąd w szt./dzień (predykcja − rzeczywistość); "
        "dodatni = przeszacowanie sprzedaży.\n\n"
        "Scenariusz *epidemia_rzeczywista* (oracle) pokazuje, jak wyglądałaby "
        "prognoza ze znanym harmonogramem epidemii - w praktyce status epidemii "
        "jest ogłaszany z wyprzedzeniem. Szczegóły: README oraz rolling-origin "
        "CV w `outputs/metrics.csv` (okno cv5_mean)."
    )

# --- prognoza na przyszłość: tabela ---
st.subheader("Prognoza na najbliższe 4 tygodnie (po końcu danych)")
future = forecast[forecast["split"] == "future"][["baseline"]].copy()
future.columns = ["Baseline (28d śr.)"]
future["Model"] = model_dynamic
st.dataframe(
    future.round(0).style.format("{:,.0f}"),
    width="stretch",
)
st.caption(f"Suma prognozy modelu na 28 dni: {future['Model'].sum():,.0f} szt. "
           f"(baseline: {future['Baseline (28d śr.)'].sum():,.0f} szt.; "
           f"epidemia: {n_ep_days} dni)")

# --- bonus: top-3 produkty ---
st.subheader("Bonus: prognozy dla top-3 produktów (wg łącznej sprzedaży)")
product_id = st.selectbox("Produkt", sorted(products_fc["product_id"].unique()))

p_fc = products_fc[products_fc["product_id"] == product_id]
fig_p = go.Figure()
p_val = p_fc[p_fc["actual"].notna()]
fig_p.add_trace(go.Scatter(x=p_val.index, y=p_val["actual"],
                           name="Rzeczywista sprzedaż", line=dict(color="#38BDF8", width=1.5)))
fig_p.add_trace(go.Scatter(x=p_fc.index, y=p_fc["baseline"], name="Baseline",
                           line=dict(color="#FBBF24", width=1.5, dash="dot")))
fig_p.add_trace(go.Scatter(x=p_fc.index, y=p_fc["model_ep0"], name="Model (bez epidemii)",
                           line=dict(color="#F472B6", width=2)))
fig_p.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_title="sztuki / dzień", legend=dict(orientation="h", y=1.12, x=0))
st.plotly_chart(fig_p, width="stretch")

p_met = products_met[products_met["product_id"] == product_id].copy()
for col in ("MAE", "RMSE", "MAPE", "MPE", "bias"):
    p_met[col] = p_met[col].round(1)
st.dataframe(p_met, width="stretch", hide_index=True)
