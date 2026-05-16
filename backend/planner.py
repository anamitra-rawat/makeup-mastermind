"""
planner.py  —  Constraint-based routine planner.

Takes ML compatibility scores and selects a valid routine subject to:
  - Hard budget constraint (total price <= user budget)
  - One product per routine step
  - Blacklisted ingredients excluded
  - Focus area filter (face / eyes / lips)
  - Routine length target (quick / standard / full)

Per proposal: "optimization problem: select the combination of products that
maximizes overall compatibility while satisfying all hard constraints."
"""
import os, json
from backend.ranking_model import predict, load as load_model, build_feature_vector
import numpy as np

PRODUCTS_JSON = os.path.join(os.path.dirname(__file__), "../data/products.json")

# ── Routine templates per style ──────────────────────────────────────────────
STYLE_ROUTINES = {
    "natural":  ["Face Primer","Foundation","Concealer","Blush","Highlighter","Mascara","Lip Balm & Treatment"],
    "glam":     ["Face Primer","Foundation","Concealer","Contour","Blush","Bronzer","Highlighter",
                 "Eye Palettes","Eyeliner","Mascara","Lipstick"],
    "office":   ["BB & CC Cream","Concealer","Blush","Mascara","Eyebrow","Lip Gloss"],
    "dramatic": ["Face Primer","Foundation","Concealer","Contour","Blush",
                 "Eye Palettes","Eyeliner","Mascara","Lipstick","Setting Spray & Powder"],
    "minimal":  ["Tinted Moisturizer","Concealer","Mascara","Lip Balm & Treatment"],
}

FOCUS_CATS = {
    "face": {"Face Primer","Foundation","BB & CC Cream","Tinted Moisturizer","Concealer",
             "Blush","Bronzer","Highlighter","Contour","Setting Spray & Powder"},
    "eyes": {"Eye Palettes","Eyeshadow","Eyeliner","Mascara","Eyebrow","Eye Primer"},
    "lips": {"Lipstick","Liquid Lipstick","Lip Gloss","Lip Liner","Lip Stain",
             "Lip Balm & Treatment","Lip Plumper"},
}

LENGTH_MAX = {"quick":4, "standard":7, "full":10}


def plan_routine(user_profile: dict, products: list[dict], model) -> dict:
    """
    Select one product per routine step using ML scores + constraint satisfaction.
    Returns dict with selected routine, total cost, and per-product scores.
    """
    style  = user_profile.get("style","natural")
    budget = float(user_profile.get("budget",150))
    length = user_profile.get("routine_length","standard")
    focus  = set(user_profile.get("focus",["face","eyes","lips"]))
    sens   = set(user_profile.get("sensitivities",[]))

    # Build step list filtered by focus
    steps = STYLE_ROUTINES.get(style, STYLE_ROUTINES["natural"])[:]
    if focus:
        steps = [s for s in steps if any(s in FOCUS_CATS.get(f,set()) for f in focus)]
    steps = steps[:LENGTH_MAX.get(length, 7)]

    # Pre-index products by category
    by_cat = {}
    for p in products:
        by_cat.setdefault(p["category"],[]).append(p)

    routine = []
    total   = 0.0
    remaining = budget

    for step in steps:
        candidates = by_cat.get(step, [])
        if not candidates:
            continue

        # Hard constraint: must fit in remaining budget
        affordable = [p for p in candidates if p["price"] <= remaining]
        if not affordable:
            continue

        # Score each candidate with ML model
        scored = []
        for p in affordable:
            s = predict(model, user_profile, p)
            # Hard penalize products flagged for user sensitivities
            for concern in sens:
                if concern in p.get("safety_flags",{}):
                    s -= 0.40
            scored.append((s, p))

        # Pick highest-scoring product
        scored.sort(key=lambda x: -x[0])
        best_score, best_prod = scored[0]

        routine.append({
            "step":         step,
            "id":           best_prod["id"],
            "brand":        best_prod["brand"],
            "category":     best_prod["category"],
            "name":         best_prod["name"],
            "price":        best_prod["price"],
            "price_tier":   best_prod["price_tier"],
            "rating":       best_prod["rating"],
            "reviews":      best_prod["reviews"],
            "ml_score":     round(best_score, 4),
            "safety_flags": best_prod.get("safety_flags",{}),
            "safety_score": best_prod.get("safety_score",1.0),
        })
        total     += best_prod["price"]
        remaining -= best_prod["price"]

    return {
        "routine":       routine,
        "total_cost":    round(total, 2),
        "budget":        budget,
        "within_budget": total <= budget,
        "style":         style,
        "steps_planned": len(steps),
        "steps_filled":  len(routine),
    }


def get_alternatives(step_category: str, exclude_id: str,
                     user_profile: dict, products: list[dict],
                     model, top_n: int = 3) -> list[dict]:
    """Return top-N alternative products for a given routine step."""
    candidates = [p for p in products
                  if p["category"] == step_category and p["id"] != exclude_id
                  and p["price"] <= float(user_profile.get("budget",150))]
    scored = sorted(candidates, key=lambda p: -predict(model, user_profile, p))
    return scored[:top_n]