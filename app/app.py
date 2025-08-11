# app/app.py
import io, os, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from statsmodels.tsa.statespace.sarimax import SARIMAX
import market_plotting as mp

pio.templates.default = "plotly_white"

FILE_ID  = "1EKLg7Qcn3zdBzEa5PLE-xU1aExmQxD0p"
GID_FX   = "1718643167"
GID_FOOD = "1298403133"

CITIES = {
    "Lagos":  (6.5244, 3.3792),
    "Kaduna": (10.5222, 7.4383)
}

# ---------- Data fetchers ----------------------------------------------
def _csv(fid, gid):
    u = f"https://docs.google.com/spreadsheets/d/{fid}/export?format=csv&gid={gid}"
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# Open-Meteo: daily precip & temp, past ~3 months + 16-day forecast
# (no API key, free). We'll aggregate Lagos+Kaduna as a simple national proxy.
def fetch_weather_features():
    frames = []
    for name, (lat, lon) in CITIES.items():
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=precipitation_sum,temperature_2m_mean"
            "&timezone=Africa%2FLagos&past_days=92&forecast_days=16"
        )
        j = requests.get(url, timeout=25).json()
        dates = pd.to_datetime(j["daily"]["time"])  # already local TZ dates
        df = pd.DataFrame({
            "Date": dates,
            f"precip_{name}": j["daily"]["precipitation_sum"],
            f"tmean_{name}":  j["daily"]["temperature_2m_mean"],
        })
        frames.append(df)
    w = frames[0]
    for df in frames[1:]:
        w = w.merge(df, on="Date", how="outer")
    w = w.sort_values("Date")
    w["precip"] = w.filter(like="precip_").mean(axis=1)
    w["tmean"]  = w.filter(like="tmean_").mean(axis=1)
    # Lags and smoothers
    w["rain_7"]  = w["precip"].rolling(7).sum()
    w["rain_14"] = w["precip"].rolling(14).sum()
    w["rain_30"] = w["precip"].rolling(30).sum()
    w["tmean_7"] = w["tmean"].rolling(7).mean()
    w["tmean_30"] = w["tmean"].rolling(30).mean()
    # Simple anomalies vs recent median
    w["rain_dev30"] = w["rain_30"] - w["rain_30"].rolling(60, min_periods=30).median()
    w["t_dev30"]    = w["tmean_30"] - w["tmean_30"].rolling(60, min_periods=30).median()
    return w[[
        "Date","precip","tmean","rain_7","rain_14","rain_30",
        "tmean_7","tmean_30","rain_dev30","t_dev30"
    ]]

# ---------- Initial loads ----------------------------------------------
fx_df   = mp.fx_analyse(_csv(FILE_ID, GID_FX))
food_df = mp.food_long  (_csv(FILE_ID, GID_FOOD))
wx_df   = fetch_weather_features()

# FX-derived exogenous variables
_fx = fx_df[["Date","Daily_Average"]].copy()
_fx["fx_ret_7"]  = _fx["Daily_Average"].pct_change(7) * 100
_fx["fx_ret_30"] = _fx["Daily_Average"].pct_change(30) * 100
_fx = _fx.drop(columns=["Daily_Average"])  # keep only exogs

# ---------- Forecast helper --------------------------------------------
def forecast_commodity(comm: str, horizon: int = 14):
    sub = food_df[food_df["Commodity"] == comm].copy()
    if sub.empty:
        return None, None
    s = (sub.set_index("Date")["Price"].asfreq("D")
            .interpolate(limit=7, limit_direction="both"))

    # Build aligned exog matrix
    exfx = _fx.set_index("Date").reindex(s.index).ffill()
    exwx = wx_df.set_index("Date").reindex(s.index).ffill()
    X = pd.concat([exfx, exwx], axis=1).fillna(0.0)

    # Model: SARIMAX with weekly seasonality, differenced
    try:
        mod = SARIMAX(
            s, exog=X, order=(1,1,1), seasonal_order=(0,1,1,7),
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = mod.fit(disp=False, maxiter=200)
    except Exception as e:
        # Fallback: no-seasonal if convergence issues
        mod = SARIMAX(
            s, exog=X, order=(1,1,1),
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = mod.fit(disp=False, maxiter=200)

    # Future exog
    future_idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    X_future = pd.concat([exfx, exwx], axis=1).reindex(future_idx).ffill().bfill()

    fc = res.get_forecast(steps=horizon, exog=X_future)
    yhat = fc.predicted_mean
    ci   = fc.conf_int(alpha=0.2)  # 80% CI

    # Build figure (last 120 days + forecast)
    hist = s.tail(120)
    df_plot = pd.DataFrame({"Date": hist.index, "Price": hist.values})
    fig = px.line(df_plot, x="Date", y="Price", title=f"{comm}: 14-day forecast (with 80% CI)")
    # Forecast line
    fig.add_scatter(x=yhat.index, y=yhat.values, name="Forecast", mode="lines")
    # CI band
    fig.add_scatter(x=ci.index, y=ci.iloc[:, 0], name="Lower (80%)", mode="lines", line=dict(dash="dot"))
    fig.add_scatter(x=ci.index, y=ci.iloc[:, 1], name="Upper (80%)", mode="lines", line=dict(dash="dot"), fill='tonexty')
    fig.update_layout(legend_title_text="Series")
    return fig, res.aic

# ---------- Dash UI ----------------------------------------------------
app = Dash(__name__, title="NG-FX & Food Dashboard", suppress_callback_exceptions=True)

# Prebuilt visuals reusing your logic
def fx_tabs():
    price = px.line(fx_df, x="Date", y=["Daily_Average", "MA_7", "MA_30"],
                    labels={"value":"₦ / USD"}, title="Price & moving averages")
    rsi = px.line(fx_df, x="Date", y="RSI_14", title="RSI-14")
    sigma20 = px.line(fx_df, x="Date", y=fx_df["Pct_Change"].rolling(20).std(),
                      labels={"y":"σ %"}, title="20-day realised σ (%)")
    hist = px.histogram(fx_df, x="Pct_Change", nbins=40, title="Distribution of daily Δ %")

    return [
        dcc.Tab(label="Price & MA", children=dcc.Graph(figure=price)),
        dcc.Tab(label="RSI-14",     children=dcc.Graph(figure=rsi)),
        dcc.Tab(label="20-day σ",   children=dcc.Graph(figure=sigma20)),
        dcc.Tab(label="Histogram",  children=dcc.Graph(figure=hist)),
    ]

# Simple cv-beta-mom scatter like your matplotlib version
_cv_rows = []
for comm, sub in food_df.groupby("Commodity"):
    sub = sub.sort_values("Date")
    if len(sub) < 8:
        continue
    cv = (sub["Price"].rolling(30, min_periods=10).std(ddof=0) /
          sub["Price"].rolling(30, min_periods=10).mean() * 100).iloc[-1]
    win = min(60, len(sub))
    beta = np.polyfit(sub["Date"].map(datetime.toordinal).tail(win), sub["Price"].tail(win), 1)[0]
    mom = (sub.set_index("Date")["Price"].resample("M").mean().pct_change().iloc[-1] * 100)
    _cv_rows.append({"Commodity": comm, "CV30": round(cv,2), "Beta": round(beta,1), "MoM%": round(0 if np.isnan(mom) else mom, 1)})
cv_beta = pd.DataFrame(_cv_rows)

scatter = px.scatter(
    cv_beta, x="CV30", y="Beta", text="Commodity",
    size=np.abs(cv_beta["MoM%"]) + 15, color="MoM%", color_continuous_scale="RdYlBu_r",
    labels={"CV30":"CV 30-day %", "Beta":"β  ₦/day"},
    title="30-day CV vs β (trend slope – bubble = |MoM %|)"
)
scatter.update_traces(textposition="top center", marker_line_width=.5)

# MoM heatmap
monthly = (food_df.groupby(["Commodity", food_df["Date"].dt.to_period("M")])["Price"]
           .mean().reset_index().rename(columns={"Date":"Month"}))
monthly["Month"] = monthly["Month"].dt.to_timestamp("M")
monthly["MoM%"]  = monthly.groupby("Commodity")["Price"].pct_change() * 100
heatmap = px.imshow(monthly.pivot(index="Commodity", columns="Month", values="MoM%"),
                    color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels={"color":"MoM %"}, title="MoM inflation heat-map")

app.layout = html.Div([
    html.H3("Nigeria FX & Food-Commodity Analytics", style={"textAlign":"center"}),

    html.Div([
        html.Button("🔄 Refresh data", id="refresh-btn"),
        html.Span("  (reloads FX, Food & Weather)", style={"marginLeft":"8px", "color":"#666"})
    ], style={"textAlign":"center", "marginBottom":"10px"}),

    dcc.Tabs([
        dcc.Tab(label="FX",   children=dcc.Tabs(fx_tabs())),
        dcc.Tab(label="Food", children=dcc.Tabs([
            dcc.Tab(label="CV vs β",         children=dcc.Graph(figure=scatter)),
            dcc.Tab(label="MoM heat-map",    children=dcc.Graph(figure=heatmap)),
            dcc.Tab(label="Commodity line",  children=[
                dcc.Dropdown(sorted(food_df["Commodity"].unique()),
                             value=sorted(food_df["Commodity"].unique())[0] if not food_df.empty else None,
                             id="comm-drop", style={"width":"360px"}),
                dcc.Graph(id="comm-graph")
            ]),
            dcc.Tab(label="Forecast", children=[
                html.Div("Weather-augmented 14-day forecast (SARIMAX)."),
                dcc.Dropdown(sorted(food_df["Commodity"].unique()),
                             value=sorted(food_df["Commodity"].unique())[0] if not food_df.empty else None,
                             id="comm-forecast", style={"width":"360px"}),
                html.Div(id="aic-note", style={"color":"#666", "marginTop":"4px"}),
                dcc.Graph(id="forecast-graph")
            ])
        ]))
    ])
])

# --- Callbacks ---------------------------------------------------------
@app.callback(Output("comm-graph", "figure"), Input("comm-drop", "value"))
def _update_comm(comm):
    d = food_df[food_df["Commodity"] == comm]
    return px.line(d, x="Date", y="Price", markers=True, title=f"{comm} price (₦)")

@app.callback(
    [Output("forecast-graph", "figure"), Output("aic-note", "children")],
    Input("comm-forecast", "value")
)
def _update_forecast(comm):
    fig, aic = forecast_commodity(comm)
    note = f"Model AIC: {aic:.1f}" if aic is not None else ""
    return fig, note

@app.callback(Output("refresh-btn", "children"), Input("refresh-btn", "n_clicks"), prevent_initial_call=True)
def _refresh(_):
    global fx_df, food_df, wx_df, _fx
    fx_df   = mp.fx_analyse(_csv(FILE_ID, GID_FX))
    food_df = mp.food_long  (_csv(FILE_ID, GID_FOOD))
    wx_df   = fetch_weather_features()
    _fx = fx_df[["Date","Daily_Average"]].copy()
    _fx["fx_ret_7"]  = _fx["Daily_Average"].pct_change(7) * 100
    _fx["fx_ret_30"] = _fx["Daily_Average"].pct_change(30) * 100
    _fx.drop(columns=["Daily_Average"], inplace=True)
    return "✅ Refreshed!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run_server(host="0.0.0.0", port=port, debug=False)
