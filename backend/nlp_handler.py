"""
nlp_handler.py  —  NLP-driven chat handler.
"""
import re
import numpy as np
from backend.ranking_model import predict

CAT_MAP = {
    "foundation":    "Foundation",
    "lipstick":      "Lipstick",
    "mascara":       "Mascara",
    "blush":         "Blush",
    "highlighter":   "Highlighter",
    "bronzer":       "Bronzer",
    "eyeliner":      "Eyeliner",
    "concealer":     "Concealer",
    "primer":        "Face Primer",
    "eyeshadow":     "Eyeshadow",
    "lip gloss":     "Lip Gloss",
    "contour":       "Contour",
    "setting spray": "Setting Spray & Powder",
    "lip liner":     "Lip Liner",
    "eyebrow":       "Eyebrow",
}


def detect_intent(message):
    lower = message.lower()
    found = {}
    patterns = {
        "swap":       ["swap","replace","different","alternative","instead"],
        "budget":     ["budget","cheap","affordable","under","less than"],
        "ingredient": ["ingredient","paraben","fragrance","sulfate","alcohol","clean","safe"],
        "explain":    ["why","explain","what makes","tell me about"],
        "rating":     ["best rated","top rated","most popular","best"],
        "new_routine":["new routine","regenerate","start over","redo"],
        "style":      ["natural","glam","office","dramatic","minimal"],
        "skin":       ["oily","dry","sensitive","combination","normal"],
    }
    for intent, words in patterns.items():
        if any(w in lower for w in words):
            found[intent] = True
    price_match = re.search(r"\$?(\d+)", lower)
    if price_match:
        found["price_value"] = int(price_match.group(1))
    return found


def handle_chat(message, history, profile, routine, products, model):
    lower   = message.lower()
    intents = detect_intent(message)

    # Category search
    matched_cat = next((CAT_MAP[k] for k in CAT_MAP if k in lower), None)
    if matched_cat and not intents.get("swap"):
        budget = float(profile.get("budget", 999))
        candidates = [p for p in products if p["category"] == matched_cat and p["price"] <= budget]
        if not candidates:
            candidates = [p for p in products if p["category"] == matched_cat]
        top = sorted(candidates, key=lambda p: -predict(model, profile, p))[:5]
        return {
            "text": "Here are the top " + matched_cat + " picks ranked by your profile compatibility:",
            "action": "show_products",
            "data": top,
            "profile_update": None,
        }

    # Budget filter
    if intents.get("budget") and not matched_cat:
        max_p = intents.get("price_value", 30)
        budget_prods = sorted(
            [p for p in products if p["price"] <= max_p],
            key=lambda p: -predict(model, profile, p)
        )[:8]
        return {
            "text": "Here are products under $" + str(max_p) + ", ranked by compatibility with your profile:",
            "action": "show_products",
            "data": budget_prods,
            "profile_update": None,
        }

    # Style change
    for style in ["natural","glam","office","dramatic","minimal"]:
        if style in lower:
            return {
                "text": "Switching to a " + style + " style! Regenerating your routine now...",
                "action": "update_profile",
                "data": {},
                "profile_update": {"style": style},
            }

    # Skin type change
    for skin in ["oily","dry","sensitive","combination","normal"]:
        if skin in lower:
            return {
                "text": "Updated your skin type to " + skin + ". Regenerating your routine...",
                "action": "update_profile",
                "data": {},
                "profile_update": {"skin_type": skin},
            }

    # Budget change
    if intents.get("budget") and intents.get("price_value"):
        new_budget = intents["price_value"]
        return {
            "text": "Updated your budget to $" + str(new_budget) + ". Regenerating your routine...",
            "action": "update_profile",
            "data": {},
            "profile_update": {"budget": new_budget},
        }

    # Ingredient / safety question
    if intents.get("ingredient"):
        concerns = []
        if "paraben"   in lower: concerns.append("parabens")
        if "fragrance" in lower: concerns.append("fragrance")
        if "sulfate"   in lower: concerns.append("sulfates")
        if "alcohol"   in lower: concerns.append("alcohol")
        if concerns:
            safe = [p for p in products if not any(c in p.get("safety_flags",{}) for c in concerns)]
            safe_top = sorted(safe, key=lambda p: -predict(model, profile, p))[:5]
            return {
                "text": "Here are top products free from " + ", ".join(concerns) + ":",
                "action": "show_products",
                "data": safe_top,
                "profile_update": None,
            }
        return {
            "text": ("Products are checked for: fragrance/parfum, parabens, sulfates, "
                     "denatured alcohol, gluten, and formaldehyde releasers. "
                     "Select sensitivities in your profile and regenerate to filter them out."),
            "action": "text_only",
            "data": {},
            "profile_update": None,
        }

    # Explain recommendation
    if intents.get("explain") and routine:
        lines = []
        for item in routine[:3]:
            flags = item.get("safety_flags", {})
            parts = []
            if not flags:
                parts.append("no flagged ingredients")
            if item.get("rating", 0) >= 4.0:
                parts.append("highly rated (" + str(item["rating"]) + " stars)")
            score = item.get("ml_score", 0)
            line = "- " + item["brand"] + " " + item["name"] + ": ML score " + str(round(score,2))
            if parts:
                line += " — " + ", ".join(parts)
            lines.append(line)
        return {
            "text": "Here is why I recommended these products:\n" + "\n".join(lines),
            "action": "text_only",
            "data": {},
            "profile_update": None,
        }

    # Top rated
    if intents.get("rating"):
        top_rated = sorted(products, key=lambda p: -(p["rating"] * np.log10(p["reviews"]+1)))[:8]
        return {
            "text": "Here are the highest-rated products weighted by review count:",
            "action": "show_products",
            "data": top_rated,
            "profile_update": None,
        }

    # New routine
    if intents.get("new_routine"):
        return {
            "text": "Regenerating your routine with current profile settings...",
            "action": "regenerate",
            "data": {},
            "profile_update": None,
        }

    # Keyword fallback search
    words = [w for w in lower.split() if len(w) > 3]
    matches = [p for p in products if any(w in p["name"].lower() or w in p["brand"].lower() for w in words)]
    if matches:
        top = sorted(matches, key=lambda p: -predict(model, profile, p))[:5]
        return {
            "text": "Found " + str(len(matches)) + " products matching your query. Top picks:",
            "action": "show_products",
            "data": top,
            "profile_update": None,
        }

    return {
        "text": ("I can help you find products, swap routine steps, filter by ingredients, "
                 "or change your style or skin type. Try: "
                 "'Show me budget concealers', 'Switch to glam style', or 'Why did you recommend that?'"),
        "action": "text_only",
        "data": {},
        "profile_update": None,
    }