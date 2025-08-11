#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_plotting.py – Static PNG plots for ₦-FX & Food-Commodity datasets
Rev-4  · 26-Apr-2025
• Robust Nigerian DD/MM/YYYY parsing (with fallback)
• Handles sparse weekly food data (relaxed windows)
• Fixes KeyError -1 and 'M' → 'ME' FutureWarning
• Produces publication-ready PNG files in the chosen output folder
"""
from __future__ import annotations

import argparse, io, math, re, sys, requests
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.stats import linregress, zscore

# ── Matplotlib defaults ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"   : 600,
    "axes.grid"    : True,
    "grid.linestyle": "--",
    "grid.alpha"   : 0.4,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ────────────────────────────────────────────────────────────────────────
# Google-Sheet CSV helper
# ────────────────────────────────────────────────────────────────────────
def _download_csv(file_id: str, gid: str | int) -> pd.DataFrame:
    url = (
        f"https://docs.google.com/spreadsheets/d/{file_id}/export"
        f"?format=csv&gid={gid}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# ═══════════════════════════════════════════════════════════════════════
# 1 FX TECHNICALS
# ═══════════════════════════════════════════════════════════════════════
FX_COLS = [
    "Morning Price(9am)",
    "Afternoon Price(2pm)",
    "Evening Price (8pm)",
]

def fx_analyse(df: pd.DataFrame) -> pd.DataFrame:
    """Clean FX sheet and calculate MA, RSI-14, Bollinger signals."""
    df = df.copy()

    # --- dates ---------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].dt.year <= datetime.now().year + 1]
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # --- numeric cast --------------------------------------------------
    for c in FX_COLS + ["Daily Average", "Daily Std"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- fill Daily_Average / Std when blank ---------------------------
    df["Daily_Average"] = df.get("Daily Average")
    m = df["Daily_Average"].isna()
    df.loc[m, "Daily_Average"] = df.loc[m, FX_COLS].mean(axis=1, skipna=True)

    df["Daily_Std"] = df.get("Daily Std")
    m = df["Daily_Std"].isna()
    df.loc[m, "Daily_Std"] = (
        df.loc[m, FX_COLS].std(axis=1, ddof=0, skipna=True)
    )

    df.dropna(subset=["Daily_Average"], inplace=True)

    # --- indicators ----------------------------------------------------
    df["Pct_Change"] = df["Daily_Average"].pct_change() * 100
    df["MA_7"]  = df["Daily_Average"].rolling(7).mean()
    df["MA_30"] = df["Daily_Average"].rolling(30).mean()

    up   = (df["MA_7"] > df["MA_30"]) & (df["MA_7"].shift(1) <= df["MA_30"].shift(1))
    down = (df["MA_7"] < df["MA_30"]) & (df["MA_7"].shift(1) >= df["MA_30"].shift(1))
    df["MA_Signal"] = 0
    df.loc[up,   "MA_Signal"] =  1
    df.loc[down, "MA_Signal"] = -1

    delta = df["Daily_Average"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up   = gain.ewm(14, min_periods=14).mean()
    roll_down = loss.ewm(14, min_periods=14).mean()
    rs = roll_up / roll_down
    df["RSI_14"]    = 100 - 100/(1+rs)
    df["RSI_Signal"] = np.select([df["RSI_14"]>70, df["RSI_14"]<30], [-1,1], 0)

    bb_ma  = df["Daily_Average"].rolling(20).mean()
    bb_std = df["Daily_Average"].rolling(20).std(ddof=0)
    df["BB_MA20"]  = bb_ma
    df["BB_Upper"] = bb_ma + 2*bb_std
    df["BB_Lower"] = bb_ma - 2*bb_std
    df["BB_Signal"] = np.select(
        [df["Daily_Average"]>df["BB_Upper"],
         df["Daily_Average"]<df["BB_Lower"]], [-1,1], 0)

    df["Trend_Score"] = df[["MA_Signal","RSI_Signal","BB_Signal"]].sum(axis=1)
    df["Overall_Signal"] = np.select(
        [df["Trend_Score"]>0, df["Trend_Score"]<0], ["Up","Down"], "Neutral")
    return df

# --- universal format helper -------------------------------------------
def _fmt(ax, title, *, xlabel="", ylabel="", date_axis=False):
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if date_axis:
        ax.xaxis.set_major_formatter(DateFormatter("%b-%y"))

# --- FX plot functions --------------------------------------------------
def fx_plot_price(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], df["Daily_Average"], label="Daily Avg", lw=1.4)
    ax.plot(df["Date"], df["MA_7"], label="MA-7")
    ax.plot(df["Date"], df["MA_30"], label="MA-30")
    ax.scatter(df.loc[df["MA_Signal"]==1, "Date"],
               df.loc[df["MA_Signal"]==1, "Daily_Average"], marker="^", s=50)
    ax.scatter(df.loc[df["MA_Signal"]==-1, "Date"],
               df.loc[df["MA_Signal"]==-1, "Daily_Average"], marker="v", s=50)
    _fmt(ax, "NGN Daily Avg with MA Crossovers",
         ylabel="₦ / USD (parallel)", date_axis=True)
    ax.legend()
    fig.tight_layout(); fig.savefig(out/"fx_price_ma.png"); plt.close(fig)

def fx_plot_rsi(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df["Date"], df["RSI_14"], lw=1.2)
    ax.axhspan(70,100, color="red",   alpha=.1)
    ax.axhspan( 0,30, color="green", alpha=.1)
    _fmt(ax, "RSI-14", date_axis=True)
    fig.tight_layout(); fig.savefig(out/"fx_rsi.png"); plt.close(fig)

def fx_plot_vol(df: pd.DataFrame, out: Path):
    roll20 = df["Pct_Change"].rolling(20, min_periods=2).std()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df["Date"], roll20, lw=1.2)
    _fmt(ax, "20-day Realised Volatility (%)", ylabel="σ %", date_axis=True)
    fig.tight_layout(); fig.savefig(out/"fx_vol.png"); plt.close(fig)

def fx_plot_hist(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["Pct_Change"].dropna(), bins=40, density=True, color="#3182bd")
    _fmt(ax, "Distribution of Daily % Changes", xlabel="Δ %", ylabel="PDF")
    fig.tight_layout(); fig.savefig(out/"fx_return_hist.png"); plt.close(fig)

def fx_plot_timeline(df: pd.DataFrame, out: Path):
    cmap={"Up":"green","Down":"red","Neutral":"grey"}
    fig, ax = plt.subplots(figsize=(10,3))
    for sig, sub in df.groupby("Overall_Signal"):
        ax.scatter(sub["Date"], sub["Daily_Average"],
                   c=cmap[sig], s=12, label=sig)
    _fmt(ax, "Signal Timeline (colour = Overall_Signal)", date_axis=True)
    fig.tight_layout(); fig.savefig(out/"fx_signal_timeline.png"); plt.close(fig)

def run_fx(fid: str, gid: str, out: Path, *, show=False):
    out.mkdir(exist_ok=True)
    df = fx_analyse(_download_csv(fid, gid))
    fx_plot_price(df, out)
    fx_plot_rsi(df, out)
    fx_plot_vol(df, out)
    fx_plot_hist(df, out)
    fx_plot_timeline(df, out)
    if show:                     # quick preview inside Colab
        from PIL import Image
        for p in sorted(out.glob("fx_*.png")): Image.open(p).show()

# ═══════════════════════════════════════════════════════════════════════
# 2 FOOD ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
_RE_NUM = re.compile(r"[0-9.,]+")

def _to_num(x):
    if pd.isna(x): return np.nan
    m = _RE_NUM.search(str(x)); return np.nan if m is None else float(m.group(0).replace(",",""))

def food_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide sheet ⟶ long format, numeric clean-up, date fix."""
    df = df.copy(); df.columns = df.columns.str.strip()

    # parse Nigerian DD/MM/YYYY first; fall back if needed
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date")

    val_cols = [c for c in df.columns if c not in ["Date", "Notes"]]
    long = (df.melt("Date", val_cols, "Commodity", "Raw")
              .assign(Price=lambda d: d["Raw"].apply(_to_num))
              .dropna(subset=["Price"])
              .sort_values(["Commodity", "Date"]))
    return long

# --- FOOD plots ----------------------------------------------------------
def food_small_multiples(long: pd.DataFrame, out: Path):
    coms = long["Commodity"].unique()
    cols, rows = 4, math.ceil(len(coms)/4)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3.4, rows*2.4))
    axs = axs.flatten()
    for i, c in enumerate(coms):
        sub = long[long["Commodity"]==c]
        axs[i].plot(sub["Date"], sub["Price"], lw=.9)
        axs[i].set_title(c, fontsize=8)
    for ax in axs[len(coms):]: ax.remove()
    fig.suptitle("Daily Prices – all commodities", fontsize=12)
    fig.tight_layout(); fig.subplots_adjust(top=.93)
    fig.savefig(out/"food_small_multiples.png"); plt.close(fig)

def food_vol_trend_scatter(long: pd.DataFrame, out: Path):
    rec=[]
    for com, sub in long.groupby("Commodity"):
        sub=sub.sort_values("Date")
        prices=sub["Price"].to_numpy()

        # 30-day coefficient of variation (ok down to 8 obs)
        cv=np.nan
        if len(prices)>=8:
            rs = pd.Series(prices).rolling(30,min_periods=8)
            cv=float((rs.std(ddof=0)/rs.mean()*100).iloc[-1])

        # β slope over ≤60 most recent points
        beta=np.nan
        win=min(60,len(prices))
        if win>=2:
            beta=linregress(
                sub["Date"].map(datetime.toordinal).to_numpy()[-win:],
                prices[-win:]
            ).slope

        # month-over-month %
        mom=np.nan
        if len(sub)>=8:
            monthly=(sub.set_index("Date")["Price"]
                       .resample("ME").mean().pct_change())
            if not monthly.empty and pd.notna(monthly.iloc[-1]):
                mom=float(monthly.iloc[-1]*100)

        # 15-day Z-score
        z15=np.nan
        if len(prices)>=15:
            z15=float(zscore(prices[-15:])[-1])

        rec.append(dict(Commodity=com,CV=cv,Beta=beta,MoM=mom,Z15=z15))
    df=pd.DataFrame(rec)

    fig,ax=plt.subplots(figsize=(7,5))
    sc=ax.scatter(df["CV"],df["Beta"],
                  s=np.nan_to_num(np.abs(df["Z15"])*25+20,nan=20),
                  c=df["MoM"],cmap="coolwarm",alpha=.8,edgecolor="k")
    for _,r in df.iterrows():
        ax.annotate(r["Commodity"],(r["CV"],r["Beta"]),fontsize=6,alpha=.75)
    fig.colorbar(sc,ax=ax,label="Latest MoM %")
    _fmt(ax,"30-day CV vs Trend β",xlabel="CV 30-day %",ylabel="β (NGN/day)")
    fig.tight_layout(); fig.savefig(out/"food_vol_trend_scatter.png"); plt.close(fig)

def food_mom_heatmap(long: pd.DataFrame, out: Path):
    monthly=(long.groupby(["Commodity", long["Date"].dt.to_period("M")])["Price"]
               .mean().reset_index().rename(columns={"Date":"Month"}))
    monthly["Month"]=monthly["Month"].dt.to_timestamp("M")
    monthly["MoM%"]=monthly.groupby("Commodity")["Price"].pct_change()*100
    pv=monthly.pivot(index="Commodity",columns="Month",values="MoM%")
    vmax=pv.abs().max().max()
    fig,ax=plt.subplots(figsize=(10,.25*len(pv)+2))
    im=ax.imshow(pv,cmap="RdYlGn_r",vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([d.strftime("%b-%y") for d in pv.columns],
                       rotation=45,ha="right",fontsize=7)
    ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index,fontsize=7)
    _fmt(ax,"MoM Inflation Heat-map",xlabel="Month",ylabel="Commodity")
    fig.colorbar(im,ax=ax,label="MoM %")
    fig.tight_layout(); fig.savefig(out/"food_mom_heatmap.png"); plt.close(fig)

def food_corr_heatmap(long: pd.DataFrame, out: Path):
    pivot=long.pivot(index="Date",columns="Commodity",values="Price").dropna(axis=1,how="any")
    corr=pivot.pct_change().corr()
    fig,ax=plt.subplots(figsize=(10,8))
    im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
    ticks=np.arange(len(corr))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns,rotation=90,fontsize=7)
    ax.set_yticklabels(corr.index,fontsize=7)
    _fmt(ax,"Return Correlation Matrix")
    fig.colorbar(im,ax=ax,fraction=.02,pad=.01)
    fig.tight_layout(); fig.savefig(out/"food_corr_heatmap.png"); plt.close(fig)

def food_boxplot(long: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(.55*long["Commodity"].nunique()+6,4))
    long.boxplot("Price",by="Commodity",ax=ax,rot=90,
                 boxprops=dict(color="#3182bd"),
                 medianprops=dict(color="#e6550d"),
                 whiskerprops=dict(color="#3182bd"))
    _fmt(ax,"Price Distribution (box-plot)",xlabel="Commodity",ylabel="NGN")
    fig.suptitle("")
    fig.tight_layout(); fig.savefig(out/"food_boxplot.png"); plt.close(fig)

def run_food(fid: str, gid: str, out: Path, *, show=False):
    out.mkdir(exist_ok=True)
    long=food_long(_download_csv(fid,gid))
    food_small_multiples(long,out)
    food_vol_trend_scatter(long,out)
    food_mom_heatmap(long,out)
    food_corr_heatmap(long,out)
    food_boxplot(long,out)
    if show:
        from PIL import Image
        for p in sorted(out.glob("food_*.png")): Image.open(p).show()

# ═══════════════════════════════════════════════════════════════════════
# 3 CLI  (Jupyter-safe)
# ═══════════════════════════════════════════════════════════════════════
parser=argparse.ArgumentParser(description="Generate static PNG analytics")
parser.add_argument("--style",default="")

sub=parser.add_subparsers(dest="cmd",required=True)

fx=sub.add_parser("fx")
fx.add_argument("--file_id",required=True)
fx.add_argument("--gid",default="1718643167")
fx.add_argument("--out_dir",default="plots_fx")
fx.add_argument("--show",action="store_true")

food=sub.add_parser("food")
food.add_argument("--file_id",required=True)
food.add_argument("--gid",default="1298403133")
food.add_argument("--out_dir",default="plots_food")
food.add_argument("--show",action="store_true")

if __name__ == "__main__":
    # strip Jupyter kernel file arg
    argv=[a for a in sys.argv[1:] if not a.endswith(".json")]
    if not argv: parser.print_help(); sys.exit()

    # optional style sheet
    if "--style" in argv:
        i=argv.index("--style"); style=argv.pop(i+1); argv.pop(i)
        try: plt.style.use(style)
        except Exception: print(f"⚠️  Unknown style '{style}' – default used")

    args=parser.parse_args(argv)

    if args.cmd=="fx":
        run_fx(args.file_id,args.gid,Path(args.out_dir),show=args.show)
    else:
        run_food(args.file_id,args.gid,Path(args.out_dir),show=args.show)
