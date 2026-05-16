"""
ranking_model.py  —  Core ML component: supervised ranking model.

Trains a GradientBoostingRegressor on weakly-labeled (user, product) pairs.
The model learns to predict compatibility scores from structured features,
replacing simple hand-written rules with a learned function.
Per proposal: "learn patterns between user attributes and product features
rather than relying entirely on manually written rules."
"""
import json, os, pickle, random
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "../models/ranking_model.pkl")
PRODUCTS_JSON = os.path.join(os.path.dirname(__file__), "../data/products.json")

STYLE_STEP_AFFINITY = {
    "natural":  {"face_base":0.9,"face_color":0.5,"face_finish":0.3,"eye_color":0.4,"eye_finish":0.7,"lip":0.6},
    "glam":     {"face_base":0.9,"face_color":0.9,"face_finish":0.8,"eye_base":0.8,"eye_color":0.9,"eye_define":0.9,"eye_finish":0.9,"lip":0.9},
    "office":   {"face_base":0.8,"face_color":0.4,"face_finish":0.5,"eye_define":0.7,"eye_finish":0.8,"lip":0.5},
    "dramatic": {"face_base":0.9,"face_color":0.8,"face_finish":0.9,"eye_base":0.9,"eye_color":0.9,"eye_define":0.9,"eye_finish":0.9,"lip":0.8},
    "minimal":  {"face_base":0.6,"lip":0.5,"eye_finish":0.5},
}
SKIN_FINISH_AFFINITY = {
    "oily":        {"is_matte":1.0,"is_longwear":0.9,"is_dewy":-0.5,"is_hydrating":0.2,"is_spf":0.5},
    "dry":         {"is_dewy":1.0,"is_hydrating":1.0,"is_matte":-0.4,"is_sheer":0.6},
    "combination": {"is_matte":0.5,"is_longwear":0.7,"is_buildable":0.6},
    "sensitive":   {"is_sheer":0.8,"is_hydrating":0.7},
    "normal":      {"is_buildable":0.5,"is_sheer":0.5},
    "all":         {},
}
PRICE_TIER_ORDER = {"budget":0,"mid":1,"high":2,"luxury":3}
FINISH_FEATURES  = ["is_matte","is_dewy","is_longwear","is_hydrating","is_spf",
                    "is_buildable","is_sheer","is_volumizing","is_waterproof"]

def _budget_tier(b): 
    return 0 if b<=60 else 1 if b<=150 else 2 if b<=250 else 3

def build_feature_vector(user, product):
    """18-dim vector encoding (user profile × product attributes)."""
    style, skin, budget = user.get("style","natural"), user.get("skin_type","all"), float(user.get("budget",150))
    rating_n  = product["rating"] / 5.0
    review_n  = min(np.log10(product["reviews"]+1)/5.0, 1.0)
    price_n   = min(product["price"]/549.0, 1.0)
    safety    = float(product.get("safety_score",1.0))
    bfit      = float(np.clip(1.0 - product["price"]/max(budget,1), 0, 1))
    step_aff  = STYLE_STEP_AFFINITY.get(style,{}).get(product.get("step_group","other"), 0.3)
    skin_c    = 0.5
    for feat, w in SKIN_FINISH_AFFINITY.get(skin,{}).items():
        if product.get(feat,0): skin_c += w * 0.12
    skin_c    = float(np.clip(skin_c, 0, 1))
    tier_match= 1.0 if PRICE_TIER_ORDER.get(product.get("price_tier","mid"),1) <= _budget_tier(budget) else 0.0
    finish_v  = [float(product.get(f,0)) for f in FINISH_FEATURES]
    popularity= (product["rating"]/5.0) * min(np.log10(product["reviews"]+1)/4.0,1.0)
    return np.array([rating_n,review_n,price_n,tier_match,safety,step_aff,
                     *finish_v, skin_c,bfit,popularity], dtype=np.float32)

SYNTHETIC_USERS = [
    {"style":"natural",  "skin_type":"dry",         "budget":80,  "sensitivities":["alcohol"]},
    {"style":"natural",  "skin_type":"sensitive",   "budget":60,  "sensitivities":["fragrance","parabens"]},
    {"style":"natural",  "skin_type":"oily",        "budget":120, "sensitivities":[]},
    {"style":"natural",  "skin_type":"combination", "budget":150, "sensitivities":["parabens"]},
    {"style":"glam",     "skin_type":"oily",        "budget":300, "sensitivities":[]},
    {"style":"glam",     "skin_type":"combination", "budget":200, "sensitivities":["sulfates"]},
    {"style":"glam",     "skin_type":"dry",         "budget":350, "sensitivities":[]},
    {"style":"glam",     "skin_type":"sensitive",   "budget":220, "sensitivities":["fragrance"]},
    {"style":"office",   "skin_type":"normal",      "budget":100, "sensitivities":[]},
    {"style":"office",   "skin_type":"oily",        "budget":80,  "sensitivities":["alcohol"]},
    {"style":"office",   "skin_type":"sensitive",   "budget":130, "sensitivities":["fragrance","sulfates"]},
    {"style":"office",   "skin_type":"dry",         "budget":90,  "sensitivities":[]},
    {"style":"dramatic", "skin_type":"combination", "budget":250, "sensitivities":[]},
    {"style":"dramatic", "skin_type":"dry",         "budget":200, "sensitivities":["fragrance"]},
    {"style":"dramatic", "skin_type":"oily",        "budget":180, "sensitivities":["parabens"]},
    {"style":"dramatic", "skin_type":"normal",      "budget":280, "sensitivities":["sulfates"]},
    {"style":"minimal",  "skin_type":"sensitive",   "budget":50,  "sensitivities":["fragrance","parabens","formaldehyde_releasers"]},
    {"style":"minimal",  "skin_type":"normal",      "budget":100, "sensitivities":[]},
    {"style":"minimal",  "skin_type":"dry",         "budget":80,  "sensitivities":["alcohol"]},
    {"style":"minimal",  "skin_type":"oily",        "budget":70,  "sensitivities":[]},
]

def _rule_score(user, product):
    """Weak supervision label: domain-rule compatibility score."""
    s = 0.5
    s += (product["rating"] - 3.5) * 0.12
    s += min(np.log10(product["reviews"]+1)/15.0, 0.08)
    for c in user.get("sensitivities",[]):
        if c in product.get("safety_flags",{}): s -= 0.30
    if product["price"] > user["budget"]:
        s -= min(((product["price"]-user["budget"])/max(user["budget"],1))*0.5, 0.35)
    elif product["price"] <= user["budget"]*0.5: s += 0.04
    step_aff = STYLE_STEP_AFFINITY.get(user["style"],{}).get(product.get("step_group","other"),0.3)
    s += (step_aff - 0.5)*0.25
    for feat, w in SKIN_FINISH_AFFINITY.get(user["skin_type"],{}).items():
        if product.get(feat,0): s += w*0.06
    return float(np.clip(s, 0.0, 1.0))

def train(products):
    """Generate weak labels, train GBR, save model."""
    random.seed(42)
    sample = random.sample(products, min(600, len(products)))
    rows, labels = [], []
    for u in SYNTHETIC_USERS:
        for p in sample:
            rows.append(build_feature_vector(u, p))
            labels.append(_rule_score(u, p))
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"[ranking] Training data: {X.shape}")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.04, subsample=0.8,
                                      min_samples_leaf=5, random_state=42)
    model.fit(X_tr, y_tr)
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"[ranking] Validation RMSE: {rmse:.4f}")
    feat_names = ["rating","review_log","price_norm","tier_match","safety","style_aff",
                  *FINISH_FEATURES,"skin_compat","budget_fit","popularity"]
    top = sorted(zip(feat_names, model.feature_importances_), key=lambda x:-x[1])[:6]
    print("[ranking] Top features:", ", ".join(f"{n}={v:.3f}" for n,v in top))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH,"wb") as f: pickle.dump(model, f)
    print(f"[ranking] Saved → {MODEL_PATH}")
    return model, rmse

def load():
    with open(MODEL_PATH,"rb") as f: return pickle.load(f)

def predict(model, user_profile, product):
    vec = build_feature_vector(user_profile, product).reshape(1,-1)
    return float(np.clip(model.predict(vec)[0], 0, 1))

if __name__ == "__main__":
    with open(PRODUCTS_JSON) as f: prods = json.load(f)
    train(prods)
