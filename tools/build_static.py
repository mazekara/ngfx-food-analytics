# tools/build_static.py
import io, requests
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import market_plotting as mp

# Google Sheet IDs (same as your Dash app)
FILE_ID  = "1EKLg7Qcn3zdBzEa5PLE-xU1aExmQxD0p"
GID_FX   = "1718643167"
GID_FOOD = "1298403133"

pio.templates.default = "plotly_white"

def _csv(fid, gid):
    u = f"https://docs.google.com/spreadsheets/d/{fid}/export?format=csv&gid={gid}"
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# Load & clean
fx   = mp.fx_analyse(_csv(FILE_ID, GID_FX))
food = mp.food_long  (_csv(FILE_ID, GID_FOOD))

# --- FX figs -----------------------------------------------------------
fig_fx_price = px.line(
    fx, x="Date", y=["Daily_Average", "MA_7", "MA_30"],
    title="FX: Price & Moving Averages", labels={"value": "₦ / USD"}
)
fig_fx_rsi   = px.line(fx, x="Date", y="RSI_14", title="FX: RSI-14")
fig_fx_hist  = px.histogram(fx, x="Pct_Change", nbins=40,
                            title="FX: Distribution of Daily Δ%",
                            labels={"Pct_Change": "Δ %"})

# --- Food figs ---------------------------------------------------------
first_comm = sorted(food["Commodity"].unique())[0] if not food.empty else None
fig_food_one = px.line(
    food[food["Commodity"]==first_comm], x="Date", y="Price",
    title=f"Commodity: {first_comm} (₦)", markers=True
) if first_comm else None

# Heat-map (MoM) using monthly resample like your matplotlib impl
monthly = (food.groupby(["Commodity", food["Date"].dt.to_period("M")])["Price"]
               .mean().reset_index().rename(columns={"Date":"Month"}))
if not monthly.empty:
    monthly["Month"] = monthly["Month"].dt.to_timestamp("M")
    monthly["MoM%"]  = monthly.groupby("Commodity")["Price"].pct_change()*100
    pv = monthly.pivot(index="Commodity", columns="Month", values="MoM%")
    fig_food_heat = px.imshow(pv, color_continuous_scale="RdYlGn_r",
                              aspect="auto", title="Food: MoM Inflation Heat-map")
else:
    fig_food_heat = None

# Build the HTML page
out = Path("site"); out.mkdir(parents=True, exist_ok=True)
parts = [
    "<h2>Nigeria FX & Food Analytics (auto-updated)</h2>",
    '<p>Interactive plots update daily from Google Sheets.</p>',
]

for fig in [fig_fx_price, fig_fx_rsi, fig_fx_hist, fig_food_one, fig_food_heat]:
    if fig is not None:
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

(out/"index.html").write_text("\n".join(parts), encoding="utf-8")
print("Wrote site/index.html ✅")
