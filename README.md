# Makeup Mastermind
AI-powered personalized makeup recommendation system.
CS 4701 AI Practicum — Meghana Kesanapalli, Anamitra Rawat, Parvi Chadha

---

## First Time Setup

### Step 1 — Make sure you have Python installed
You need Python 3.10 or higher. If you have Anaconda, you're good.

### Step 2 — Clone or download the project
If pulling from Git:
```bash
git clone <your-repo-url>
cd makeup-mastermind
```

### Step 3 — Add the dataset
Place `sephora_website_dataset.csv` inside the `data/` folder.
The folder should look like:
```
data/
└── sephora_website_dataset.csv
```

### Step 4 — Install dependencies
In your terminal, from inside the `makeup-mastermind` folder run:
```bash
pip install -r requirements.txt
```

### Step 5 — Run setup (only needed once)
This processes the data and trains the ML model. Takes about 10 seconds.
```bash
python setup.py
```
You should see it print training stats and a validation RMSE score.
After this runs, two files get saved automatically:
- `data/products.json` — cleaned product database
- `models/ranking_model.pkl` — trained GradientBoosting model

### Step 6 — Start the server
```bash
python -m backend.app
```
Wait until you see:
```
Running on http://127.0.0.1:5000
```

### Step 7 — Open the app
Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## Every Time After That

You only need to do Steps 4–7 once. After that, every time you want to run the app:

```bash
python -m backend.app
```
Then go to `http://127.0.0.1:5000` in your browser.

---

## Project Structure

```
makeup-mastermind/
├── backend/
│   ├── __init__.py          ← required, do not delete
│   ├── app.py               ← Flask server (run this)
│   ├── data_pipeline.py     ← loads & cleans Sephora CSV
│   ├── ranking_model.py     ← GradientBoosting ML model
│   ├── planner.py           ← constraint-based routine optimizer
│   └── nlp_handler.py       ← NLP chat intent parser
├── frontend/
│   └── index.html           ← web UI
├── data/
│   └── sephora_website_dataset.csv   ← add this yourself
├── models/                  ← trained model saved here by setup.py
├── setup.py                 ← run once to process data + train model
├── requirements.txt
└── README.md
```

---

## ML Architecture

### Ranking Model (`ranking_model.py`)
- **Type:** GradientBoostingRegressor (scikit-learn)
- **Input:** 18-dimensional feature vector per (user, product) pair
  - Product features: rating, review count, price, finish type (matte/dewy/longwear etc), safety score
  - User-product cross features: style × step affinity, skin type × finish compatibility, budget fit
- **Training:** Weak supervision — 20 synthetic user profiles × 600 products = 12,000 labeled pairs
- **Validation RMSE:** ~0.01

### Constraint Planner (`planner.py`)
- Selects one product per routine step
- Hard constraints: total cost ≤ budget, flagged ingredients excluded
- Picks the combination that maximizes ML compatibility scores

### Ingredient Safety (`data_pipeline.py`)
- Keyword blacklist: fragrance, parabens, sulfates, alcohol, gluten, formaldehyde releasers
- Each product gets a safety score from 0.0 (many concerns) to 1.0 (clean)

---

## Adding More Data Later

### Open Beauty Facts (expands ingredient coverage)
1. Download the export from openbeautyfacts.org/data
2. Place it in `data/open_beauty_facts.csv`
3. Re-run `python setup.py`

### Ulta Dataset
1. Download from Kaggle and place in `data/ulta_products.csv`
2. Re-run `python setup.py`

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
Make sure you are running commands from inside the `makeup-mastermind` folder, not from inside `backend/`.

**`Address already in use`**
Another process is using port 5000. Run:
```bash
lsof -i :5000
kill -9 <PID>
```
Then start the server again.

**App loads but Generate Routine shows an error**
Make sure you ran `python setup.py` first so the model file exists in `models/`.

**Page not found at localhost:5000**
Use `http://127.0.0.1:5000` instead — they are the same thing.