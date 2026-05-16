"""
app.py  —  Flask REST API + serves frontend
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend.data_pipeline import run_pipeline
from backend.ranking_model  import train as train_model, load as load_model, predict
from backend.planner        import plan_routine, get_alternatives
from backend.nlp_handler    import handle_chat

FRONTEND_DIR  = os.path.join(os.path.dirname(__file__), "../frontend")
PRODUCTS_JSON = os.path.join(os.path.dirname(__file__), "../data/products.json")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "../models/ranking_model.pkl")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

PRODUCTS = []
MODEL    = None
MODEL_STATS = {}

def startup():
    global PRODUCTS, MODEL, MODEL_STATS
    print("=" * 50)
    print("  Makeup Mastermind — starting up")
    print("=" * 50)

    if not os.path.exists(PRODUCTS_JSON):
        PRODUCTS = run_pipeline()
    else:
        with open(PRODUCTS_JSON) as f:
            PRODUCTS = json.load(f)
        print(f"[app] Loaded {len(PRODUCTS)} products from cache")

    if not os.path.exists(MODEL_PATH):
        MODEL, rmse = train_model(PRODUCTS)
        MODEL_STATS = {"rmse": round(rmse, 4), "n_products": len(PRODUCTS)}
    else:
        MODEL = load_model()
        MODEL_STATS = {"cached": True, "n_products": len(PRODUCTS)}
        print("[app] Loaded model from cache")

    print("[app] Ready — open http://localhost:5000")


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/generate-routine", methods=["POST"])
def generate_routine():
    profile = request.json
    if not profile:
        return jsonify({"error": "No profile provided"}), 400
    result = plan_routine(profile, PRODUCTS, MODEL)
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    body     = request.json or {}
    message  = body.get("message", "")
    history  = body.get("history", [])
    profile  = body.get("profile", {})
    routine  = body.get("routine", [])
    response = handle_chat(message, history, profile, routine, PRODUCTS, MODEL)
    return jsonify({"response": response})


@app.route("/api/swap-product", methods=["POST"])
def swap_product():
    body       = request.json or {}
    step_cat   = body.get("category", "")
    exclude_id = body.get("exclude_id", "")
    profile    = body.get("profile", {})
    alts = get_alternatives(step_cat, exclude_id, profile, PRODUCTS, MODEL)
    return jsonify({"alternatives": alts})


@app.route("/api/products", methods=["GET"])
def search_products():
    q          = request.args.get("q", "").lower()
    category   = request.args.get("category", "")
    max_price  = float(request.args.get("max_price", 9999))
    min_rating = float(request.args.get("min_rating", 0))
    limit      = int(request.args.get("limit", 20))

    results = PRODUCTS
    if q:
        results = [p for p in results if q in p["name"].lower() or q in p["brand"].lower() or q in p["category"].lower()]
    if category:
        results = [p for p in results if p["category"].lower() == category.lower()]
    results = [p for p in results if p["price"] <= max_price and p["rating"] >= min_rating]
    results = sorted(results, key=lambda p: -(p["rating"] * (1 + min(p["reviews"], 1000) / 1000)))
    return jsonify({"products": results[:limit], "total": len(results)})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 300,
        "features": 18,
        "training_approach": "weak supervision (synthetic user profiles x product features)",
        **MODEL_STATS
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global PRODUCTS, MODEL, MODEL_STATS
    PRODUCTS = run_pipeline()
    MODEL, rmse = train_model(PRODUCTS)
    MODEL_STATS = {"rmse": round(rmse, 4), "n_products": len(PRODUCTS)}
    return jsonify({"status": "ok", **MODEL_STATS})


if __name__ == "__main__":
    startup()
    app.run(debug=True, port=5000)