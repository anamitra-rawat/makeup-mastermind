"""
app.py — Flask REST API for Makeup Mastermind.

Wires together:
  - Day-3 trained Gradient Boosting ranker  (models/gb_model.pkl)
  - Day-4 MILP routine planner              (backend/routine_planner.py)
  - Day-5 keyword safety filter             (backend/safety_filter.py)
  - Day-6 Claude natural-language wrapper   (backend/claude_wrapper.py)

Catalog, flags, and model are loaded ONCE at startup.
"""
import os
import sys
import math

# Ensure backend/ is importable when running `python -m backend.app` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # picks up ANTHROPIC_API_KEY from .env

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend.routine_planner import plan_routine
from backend.safety_filter   import load_precomputed_flags, apply_precomputed_flags
from backend.claude_wrapper  import chat as claude_chat

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR  = os.path.join(ROOT, "frontend")
CATALOG_PATH  = os.path.join(ROOT, "data", "merged_products_with_personalization.csv")
FLAGS_PATH    = os.path.join(ROOT, "data", "safety_filter_catalog.csv")

# All three trained models from Day 3. The frontend's selected_model field
# routes to one of these via MODELS[name]. "our_best" is an alias for
# gradient_boosting (the production model that the Day-4 planner targets).
# rf_model.pkl is 302 MB → gitignored; load gracefully if it's missing locally.
MODEL_PATHS = {
    "gradient_boosting": os.path.join(ROOT, "models", "gb_model.pkl"),
    "ridge":             os.path.join(ROOT, "models", "ridge_model.pkl"),
    "random_forest":     os.path.join(ROOT, "models", "rf_model.pkl"),
}

# ── Startup: load catalog + safety flags + GBM once ────────────────────────
print("=" * 60)
print("  Makeup Mastermind — starting up")
print("=" * 60)

if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Catalog missing: {CATALOG_PATH}")
if not os.path.exists(FLAGS_PATH):
    raise FileNotFoundError(f"Safety flags missing: {FLAGS_PATH}")
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("[startup] WARNING: ANTHROPIC_API_KEY not set — /api/chat will fail")

print(f"[startup] loading catalog from {CATALOG_PATH}")
CATALOG = pd.read_csv(CATALOG_PATH)
print(f"[startup] catalog: {len(CATALOG)} products, {len(CATALOG.columns)} columns")

print(f"[startup] loading safety flags from {FLAGS_PATH}")
FLAGS = load_precomputed_flags(FLAGS_PATH)
print(f"[startup] flags: {len(FLAGS)} rows")

print("[startup] loading models...")
MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        print(f"  ✓ {name:<18} → {type(MODELS[name]).__name__}")
    else:
        print(f"  ✗ {name:<18} → NOT FOUND at {path} (will fall back to gradient_boosting)")

# Gradient boosting is mandatory — it's the production model the planner targets.
if "gradient_boosting" not in MODELS:
    raise FileNotFoundError(f"Required model missing: {MODEL_PATHS['gradient_boosting']}")

# Alias: "our_best" → the production GBM. Same architecture and weights as
# gradient_boosting; the UI exposes it as a separate card per the project's
# original PDF spec ("add our own best model trigger").
MODELS["our_best"] = MODELS["gradient_boosting"]

DEFAULT_MODEL_NAME = "gradient_boosting"

print("[startup] ready — open http://127.0.0.1:5000")
print("=" * 60)

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)


# ── Helpers ────────────────────────────────────────────────────────────────

def _df_to_records(df):
    """Convert DataFrame to JSON-serializable records.
    Browser JSON.parse rejects NaN and Infinity even though Python json.dumps
    allows them — so we replace both with None before converting."""
    if df is None or len(df) == 0:
        return []
    # First normalize: turn +inf / -inf into NaN, then NaN into None.
    df2 = df.replace([np.inf, -np.inf], np.nan)
    out = df2.where(pd.notna(df2), None).to_dict(orient="records")
    # Coerce numpy scalar types that pandas leaves behind.
    for row in out:
        for k, v in list(row.items()):
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                row[k] = None if (math.isnan(fv) or math.isinf(fv)) else fv
            elif isinstance(v, np.bool_):
                row[k] = bool(v)
            elif isinstance(v, float):
                row[k] = None if (math.isnan(v) or math.isinf(v)) else v
    return out


def _safe_user_payload(body):
    """Normalize the incoming JSON profile and stamp a transient user_id."""
    profile = body or {}
    sens = profile.get("sensitivities") or []
    if isinstance(sens, str):
        sens = [s.strip() for s in sens.split(",") if s.strip()]
    return {
        "user_id":      profile.get("user_id", "u_live"),
        "skin_type":    profile.get("skin_type", "normal"),
        "skin_tone":    profile.get("skin_tone", "medium"),
        "budget":       float(profile.get("budget", 100)),
        "style":        profile.get("style", "natural"),
        "sensitivities": sens,
    }


def _pick_model(body):
    """Resolve `selected_model` from the request to a loaded model object.
    Falls back to gradient_boosting if the requested model isn't loaded."""
    name = (body or {}).get("selected_model", DEFAULT_MODEL_NAME)
    if name not in MODELS:
        print(f"[warn] requested model '{name}' not loaded → falling back to {DEFAULT_MODEL_NAME}")
        name = DEFAULT_MODEL_NAME
    return name, MODELS[name]


# ── Static frontend ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── API endpoints ──────────────────────────────────────────────────────────

@app.route("/api/generate-routine", methods=["POST"])
def generate_routine():
    user = _safe_user_payload(request.json)
    model_name, model = _pick_model(request.json)
    print(f"[generate] user={user['style']}/{user['skin_type']}/${user['budget']} · model={model_name}")

    # Pre-filter via precomputed flags before scoring (Day-5 fast path).
    mask = apply_precomputed_flags(FLAGS, user["sensitivities"])
    # mask is aligned with FLAGS; FLAGS is in the same row order as CATALOG.
    # Safety: assert lengths match before applying.
    if len(mask) != len(CATALOG):
        # Fall back to letting plan_routine re-scan ingredient text.
        safe_catalog = CATALOG
    else:
        safe_catalog = CATALOG[mask.values].reset_index(drop=True)

    result = plan_routine(user, safe_catalog, model, safety_filter_df=None)

    routine_records = _df_to_records(result["routine"])
    # Add an empty matched_sensitivities for each (planner guarantees no violations
    # for the user's flagged sensitivities, so this is always []).
    for row in routine_records:
        row["matched_sensitivities"] = []

    # Force every numeric to Python primitives — the planner returns numpy
    # scalars (from .sum() on pandas Series) and Flask's JSON encoder can't
    # serialize numpy.float64 / numpy.bool_.
    total_cost = float(result["total_cost"])
    budget     = float(user["budget"])
    return jsonify({
        "routine":        routine_records,
        "status":         str(result["status"]),
        "total_cost":     total_cost,
        "budget":         budget,
        "within_budget":  bool(total_cost <= budget),
        "slot_coverage":  float(result["slot_coverage"]),
        "dropped_slots":  list(result["dropped_slots"]),
        "drop_reasons":   dict(result["drop_reasons"]),
        "n_products":     int(result["n_products"]),
        "style":          user["style"],
        "planner_used":   str(result["planner_used"]),
        "model_used":     model_name,
    })


@app.route("/api/chat", methods=["POST"])
def chat_route():
    body = request.json or {}
    user_message  = body.get("message", "").strip()
    user_profile  = _safe_user_payload(body.get("profile", {}))
    chat_history  = body.get("history", [])
    # Frontend currently sends `routine` as the list (legacy shape). Wrap it as
    # the planner-style dict the wrapper expects.
    routine_list  = body.get("routine", []) or []
    routine_dict  = {
        "routine":       routine_list,
        "status":        "Optimal" if routine_list else "None",
        "total_cost":    sum(float(p.get("price_usd", 0)) for p in routine_list),
        "slot_coverage": 1.0 if routine_list else 0.0,
        "dropped_slots": [],
    }

    if not user_message:
        return jsonify({"response": {"text": "Type a question to get started.", "action": "text_only"}})

    try:
        reply = claude_chat(
            user_message  = user_message,
            user_profile  = user_profile,
            routine_dict  = routine_dict,
            chat_history  = chat_history,
        )
    except Exception as e:
        return jsonify({"response": {
            "text": f"⚠ Claude API error: {type(e).__name__}: {e}. Check your ANTHROPIC_API_KEY in .env.",
            "action": "text_only",
        }}), 200

    return jsonify({"response": {"text": reply, "action": "text_only", "data": []}})


@app.route("/api/swap-product", methods=["POST"])
def swap_product():
    """Return top-N alternative products in the same category, excluding one."""
    body         = request.json or {}
    category     = body.get("category", "")
    exclude_id   = str(body.get("exclude_id", ""))
    profile      = _safe_user_payload(body.get("profile", {}))
    top_n        = int(body.get("top_n", 3))
    _, model     = _pick_model(body.get("profile", {}))

    # Filter to same category + apply user's safety filter + drop the excluded product.
    mask = apply_precomputed_flags(FLAGS, profile["sensitivities"])
    safe_catalog = CATALOG[mask.values] if len(mask) == len(CATALOG) else CATALOG
    cands = safe_catalog[
        (safe_catalog["category_unified"] == category) &
        (safe_catalog["source_id"].astype(str) != exclude_id) &
        (safe_catalog["price_usd"].fillna(np.inf) <= profile["budget"])
    ]

    if len(cands) == 0:
        return jsonify({"alternatives": []})

    # Score with the user's chosen model via the planner's internal scorer.
    from backend.routine_planner import _score_catalog
    scored = _score_catalog(profile, cands, model)
    top = scored.sort_values("predicted_score", ascending=False).head(top_n)

    return jsonify({"alternatives": _df_to_records(top)})


@app.route("/api/products", methods=["GET"])
def search_products():
    """Catalog search for the Explore tab."""
    q          = request.args.get("q", "").strip().lower()
    category   = request.args.get("category", "")
    max_price  = float(request.args.get("max_price", 9999))
    limit      = int(request.args.get("limit", 30))

    df = CATALOG
    if q:
        m_name  = df["product_name"].fillna("").str.lower().str.contains(q, regex=False)
        m_brand = df["brand"].fillna("").str.lower().str.contains(q, regex=False)
        df = df[m_name | m_brand]
    if category:
        df = df[df["category_unified"] == category]
    df = df[df["price_usd"].fillna(np.inf) <= max_price]
    df = df.sort_values(
        by=["rating", "num_reviews"], ascending=[False, False], na_position="last"
    ).head(limit)

    # Attach precomputed safety flags for visual badges (which sensitivities each product hits).
    flag_cols = [c for c in FLAGS.columns if c.startswith("flagged_")]
    if flag_cols:
        f = FLAGS.set_index(["source", "source_id"])[flag_cols]
        df = df.merge(f, left_on=["source", "source_id"], right_index=True, how="left")

    records = _df_to_records(df)
    # Roll the flagged_* booleans into a single matched_sensitivities list per row.
    for row in records:
        ms = []
        for c in flag_cols:
            if row.get(c):
                ms.append(c.replace("flagged_", ""))
            row.pop(c, None)
        row["matched_sensitivities"] = ms

    return jsonify({"products": records, "total": len(records)})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    loaded = {name: type(m).__name__ for name, m in MODELS.items()}
    return jsonify({
        "models_loaded":    loaded,
        "default_model":    DEFAULT_MODEL_NAME,
        "catalog_rows":     len(CATALOG),
        "flagged_rows":     len(FLAGS),
        "claude_model":     os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "has_api_key":      bool(os.environ.get("ANTHROPIC_API_KEY")),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
