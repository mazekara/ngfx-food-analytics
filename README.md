# Nigeria FX & Food Analytics — Free Hosting

Two free deploy paths:

## A) GitHub Pages (static, auto-updated)
1. Create a **public** repo on GitHub. Enable **Settings → Pages → Source: GitHub Actions**.
2. Commit this repo layout. The scheduled workflow in `.github/workflows/build.yml` will run daily and publish `site/`.
3. Your site will live at: `https://<username>.github.io/<reponame>/`.
4. Make sure your Google Sheet is shared: **Anyone with the link → Viewer**.

## B) Hugging Face Spaces (full Dash app)
1. Create a (free) account at Hugging Face → New **Space** → **Docker** runtime.
2. Upload `app/app.py`, `app/requirements.txt`, `app/Dockerfile` and `market_plotting.py` to the Space (preserve paths).
3. The Space will build and serve your app at a stable URL. It may sleep after idle; it wakes on visit.

## Weather & Forecasting
- Weather via **Open-Meteo** (no key). Aggregates Lagos + Kaduna; builds rain/temperature lags and anomalies.
- Forecast: 14-day **SARIMAX** with weekly seasonality and exogenous features `[ΔFX (7/30d), rain sums (7/14/30), temp means (7/30), anomalies]`.

## Local Dev
```bash
python -m pip install -r app/requirements.txt
python app/app.py
```
