# Makeup Mastermind
AI-powered personalized makeup + skincare routine recommender.
CS 4701 AI Practicum — Meghana Kesanapalli, Anamitra Rawat, Parvi Chadha

A Flask web app that wires together a trained ranker (gradient boosting), a constraint-based routine planner (MILP), a keyword safety filter, and a Claude-powered natural-language chatbot.

---

## Supporting materials (datasets, notebooks, model files)

All the raw data, training notebooks, intermediate CSVs, validation samples, and full-size model files live in this Google Drive folder:

**https://drive.google.com/drive/folders/1Ho2et8wu3nR4I17sfj3l3UVdwjeiJ1QZ?usp=sharing**

What's in there:
- The full training and evaluation notebooks (data merge, feature engineering, model training, evaluation, ablations, planner, safety filter)
- The raw Sephora + Ulta product CSVs and the 1.1 M Sephora reviews
- All intermediate result CSVs (evaluation, ablation, fairness, learning curve, planner comparisons, etc.)
- The 302 MB `rf_model.pkl` that's too large for GitHub

The app itself doesn't need any of this — everything required to *run* the app is already in this repo. The Drive link is for graders, reviewers, or anyone who wants to inspect the underlying data, retrain models, or rerun the experiments.

---

## First-Time Setup

### Step 1 — Python 3.10+
Anaconda or python.org both work.

### Step 2 — Install dependencies
From inside the project folder (the one with `requirements.txt`):
```bash
pip install -r requirements.txt
```

This installs Flask, pandas, scikit-learn, joblib, **PuLP** (for the MILP planner), the **Anthropic SDK** (for the Claude chatbot), and python-dotenv.

### Step 3 — Add your Anthropic API key
The chatbot uses `claude-sonnet-4-6` (configurable). You need an Anthropic API key.

1. Get a key from https://console.anthropic.com/settings/keys
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env       # macOS/Linux
   copy .env.example .env     # Windows cmd
   ```
3. Open `.env` and paste your real key:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx...
   ```

`.env` is in `.gitignore` — it will **never** be committed.

### Step 4 — (One-time) Download the Random Forest model
`rf_model.pkl` is 302 MB — over GitHub's 100 MB hard file limit, so it's **not in the repo**. The app runs without it (the Random Forest model card just silently falls back to Gradient Boosting). To enable the actual RF model:

1. Open this Google Drive folder:
   **https://drive.google.com/drive/folders/1Ho2et8wu3nR4I17sfj3l3UVdwjeiJ1QZ?usp=sharing**
2. Download **`rf_model.pkl`**
3. Move it into the project's `models/` folder so the path is:
   ```
   makeup-mastermind/models/rf_model.pkl
   ```

You only need to do this once per machine. If you skip it, the backend logs `✗ random_forest → NOT FOUND` at startup and silently routes Random Forest requests to Gradient Boosting — everything else works.

### Step 5 — Start the server
```bash
python -m backend.app
```

You'll see startup logs like:
```
[startup] catalog: 5085 products, 54 columns
[startup] flags: 5085 rows
[startup] loading models...
  ✓ gradient_boosting  → GradientBoostingRegressor
  ✓ ridge              → Ridge
  ✓ random_forest      → RandomForestRegressor    (or ✗ NOT FOUND if you skipped Step 4)
[startup] ready — open http://127.0.0.1:5000
```

### Step 6 — Open the app
http://127.0.0.1:5000

---

## Every Time After That
Steps 1–4 are one-time. After that, every time you want to run the app:
```bash
python -m backend.app
```

---

## Project Structure

```
makeup-mastermind/
├── backend/
│   ├── __init__.py
│   ├── app.py              ← Flask server
│   ├── routine_planner.py  ← MILP routine planner (imports gb_model.pkl)
│   ├── safety_filter.py    ← keyword-based ingredient safety filter
│   └── claude_wrapper.py   ← natural-language chat wrapper (claude-sonnet-4-6)
├── frontend/
│   └── index.html          ← single-page UI
├── data/
│   ├── merged_products_with_personalization.csv  ← 5,085 products
│   └── safety_filter_catalog.csv                 ← precomputed sensitivity flags
├── models/
│   ├── gb_model.pkl        ← trained Gradient Boosting ranker — 12 MB, in git
│   ├── ridge_model.pkl     ← trained Ridge baseline           — 1 KB, in git
│   └── rf_model.pkl        ← trained Random Forest            — 302 MB, NOT in git
│                              (over GitHub's 100 MB file limit — download from
│                               the Drive link in Step 4 of First-Time Setup)
├── .env.example            ← copy to .env and add your API key
├── .gitignore              ← .env is excluded
├── requirements.txt
└── README.md
```

---

## How the AI components fit together

```
User survey ─▶ Safety filter (pre-excludes products with flagged ingredients)
                   │
                   ▼
              Trained ranker (Ridge / RF / GBM — scores remaining catalog per user)
                   │
                   ▼
              MILP routine planner (selects one product per slot, ≤ budget)
                   │
                   ▼
              Claude chat wrapper (translates the routine into NL — never invents)
                   │
                   ▼
              Frontend renders cards + chat
```

**Important framing:** Claude does *not* make recommendation decisions. The trained ranker + MILP planner choose the products; Claude only describes them. The system prompt in `claude_wrapper.py` enforces this — Claude is forbidden from referencing any product not in the routine output.

---

## Configuration

| Env var | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Auth for the Claude chatbot |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Override the Claude model (e.g. `claude-haiku-4-5` for cheaper testing) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
Run from the project root, not from inside `backend/`.

**`Address already in use`**
Another process is on port 5000. Kill it or change the port in `app.py`.

**`FileNotFoundError: Catalog missing`**
The `data/merged_products_with_personalization.csv` or `models/gb_model.pkl` is missing. Both ship in the git repo, so a fresh `git clone` should have them. If they're gone, re-clone or pull them from the Drive folder linked in Step 4.

**Chat returns "⚠ Claude API error: AuthenticationError"**
Your `.env` is missing the API key or the key is wrong. Check `.env` exists in the project root and has `ANTHROPIC_API_KEY=sk-ant-...`.

**Page loads but routine generation fails**
Open browser DevTools → Network tab → click "Generate". Check the `/api/generate-routine` response. If it's a 500, the Flask terminal will show the Python traceback.

**`pulp.PulpSolverError`**
PuLP couldn't find its bundled CBC solver. Re-install: `pip install --force-reinstall pulp`.
