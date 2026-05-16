#!/usr/bin/env python3
"""
routine_planner.py
==================
Standalone module for the Day 6 chatbot.

Public function
---------------
plan_routine(user_profile, catalog_df, gb_model,
             safety_filter_df=None, budget_aware=True)

Returns a dict:
  routine              : pd.DataFrame — selected products (one row per item)
  status               : str — "Optimal" | "Infeasible"
  total_cost           : float
  total_compatibility  : float
  slot_coverage        : float — fraction of required slots filled
  dropped_slots        : list[str]
  drop_reasons         : dict[str, str]
  n_products           : int

Usage (Day 6 chatbot)
---------------------
  from routine_planner import plan_routine
  import pandas as pd, pickle

  catalog  = pd.read_csv("merged_products_with_personalization.csv")
  gb_model = pickle.load(open("gb_model.pkl", "rb"))

  user = {
      "user_id"     : "u_live_001",
      "skin_type"   : "oily",
      "skin_tone"   : "medium",
      "budget"      : 75.0,
      "style"       : "natural",
      "sensitivities": ["fragrance"],
  }

  result = plan_routine(user, catalog, gb_model)
  print(result["routine"][["category_unified","brand","product_name","price_usd"]])
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── PuLP import with graceful fallback ──────────────────────────────────────
try:
    import pulp
    import logging
    logging.getLogger("pulp").setLevel(logging.WARNING)
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False
    warnings.warn(
        "PuLP not installed. Falling back to greedy planner. "
        "Install with: pip install pulp",
        RuntimeWarning
    )

# ── Constants ────────────────────────────────────────────────────────────────

ROUTINE_TEMPLATES = {
    "no_makeup": [
        ("Cleansers",              True),
        ("Moisturizers",           True),
        ("Face Serums",            False),
        ("Face Sunscreen",         True),
        ("Lip Balm & Treatment",   False),
    ],
    "natural": [
        ("Cleansers",              True),
        ("Moisturizers",           True),
        ("Face Sunscreen",         True),
        ("Tinted Moisturizer",     False),
        ("Concealer",              False),
        ("Blush",                  False),
        ("Lip Balm & Treatment",   False),
    ],
    "glam": [
        ("Cleansers",              True),
        ("Moisturizers",           True),
        ("Face Primer",            False),
        ("Foundation",             True),
        ("Concealer",              False),
        ("Blush",                  False),
        ("Highlighter",            False),
        ("Setting Spray & Powder", False),
        ("Mascara",                False),
        ("Lipstick",               False),
    ],
    "editorial": [
        ("Cleansers",              True),
        ("Moisturizers",           True),
        ("Face Primer",            False),
        ("Foundation",             True),
        ("Concealer",              False),
        ("Eye Palettes",           False),
        ("Eyeliner",               False),
        ("Lipstick",               False),
        ("Setting Spray & Powder", False),
    ],
    "dramatic": [
        ("Cleansers",              True),
        ("Moisturizers",           True),
        ("Face Primer",            False),
        ("Foundation",             True),
        ("Concealer",              False),
        ("Eye Palettes",           False),
        ("Eyeliner",               False),
        ("Mascara",                False),
        ("False Eyelashes",        False),
        ("Liquid Lipstick",        False),
        ("Setting Spray & Powder", False),
    ],
}

SLOT_ALTERNATIVES = {
    "Tinted Moisturizer": "BB & CC Cream",
    "Lipstick":           "Liquid Lipstick",
    "Liquid Lipstick":    "Lipstick",
}

SENSITIVITY_KEYWORDS = {
    "fragrance"  : ["fragrance", "parfum"],
    "parabens"   : ["paraben"],
    "sulfates"   : ["sulfate", "sls", "sles"],
    "phthalates" : ["phthalate"],
    "alcohol"    : ["denatured alcohol", "alcohol denat"],
    "silicones"  : ["dimethicone", "cyclomethicone", "cyclopentasiloxane"],
}

RELAXATION_PRIORITY = [
    "Setting Spray & Powder",
    "Face Sunscreen",
    "Face Serums",
    "Foundation",
    "Moisturizers",
    "Cleansers",
]

GREEDY_SLOT_PRIORITY = [
    "Cleansers", "Moisturizers", "Face Sunscreen", "Face Serums",
    "Face Primer", "Foundation", "Concealer",
    "Tinted Moisturizer", "BB & CC Cream", "Blush", "Highlighter",
    "Contour", "Eye Palettes", "Eyeliner", "Mascara", "False Eyelashes",
    "Lipstick", "Liquid Lipstick", "Lip Balm & Treatment",
    "Setting Spray & Powder",
]

# Feature columns — must match Notebook 3A exactly
SKIN_TYPE_KEYS = ["oily", "dry", "combination", "normal", "sensitive"]
SKIN_TONE_KEYS = ["fair", "light", "medium", "tan", "deep"]
STYLE_KEYS     = ["natural", "glam", "editorial", "no_makeup", "dramatic"]
SENSITIVITY_KEYS = ["fragrance", "parabens", "sulfates", "phthalates", "alcohol", "silicones"]

ACTIVE_INGREDIENTS = [
    "retinol", "niacinamide", "vitamin c", "ascorbic acid",
    "hyaluronic acid", "salicylic acid", "glycolic acid", "lactic acid",
    "ceramide", "peptide", "zinc oxide", "titanium dioxide",
    "centella asiatica", "cica", "benzoyl peroxide",
]

HIGHLIGHT_SKIN_KEYWORDS = {
    "oily"       : ["oil-free", "oil free", "mattifying", "matte", "pore"],
    "dry"        : ["hydrating", "hydration", "moisturizing", "dry skin", "rich"],
    "combination": ["balance", "combination", "normal to oily", "normal to dry"],
    "normal"     : ["all skin", "all skin types", "normal"],
    "sensitive"  : ["sensitive", "fragrance-free", "gentle", "calming", "soothing"],
}

HIGHLIGHT_STYLE_KEYWORDS = {
    "natural"   : ["tinted", "sheer", "natural", "lightweight", "bb"],
    "glam"      : ["full coverage", "shimmer", "glow", "luminous", "buildable"],
    "editorial" : ["bold", "pigmented", "intense", "vivid", "dramatic"],
    "no_makeup" : ["bare", "minimal", "transparent", "invisible", "skincare"],
    "dramatic"  : ["high coverage", "long-lasting", "waterproof", "full coverage"],
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _parse_user(user_profile):
    """Accept dict or Series; return a normalised dict."""
    if hasattr(user_profile, "to_dict"):
        d = user_profile.to_dict()
    else:
        d = dict(user_profile)
    # sensitivities: accept list or comma-separated string
    sens = d.get("sensitivities", [])
    if isinstance(sens, str):
        sens = [s.strip() for s in sens.split(",") if s.strip()]
    d["sensitivities"] = sens
    return d


def _triggers_sensitivity(ingredients_text, sensitivity):
    if not ingredients_text or pd.isna(ingredients_text):
        return False
    text = str(ingredients_text).lower()
    return any(kw in text for kw in SENSITIVITY_KEYWORDS.get(sensitivity, []))


def _featurize_pair(user, product_row):
    feats = {}
    
    # user features
    feats["user_budget"]          = float(user.get("budget", 0))
    feats["user_n_sensitivities"] = len(user.get("sensitivities", []))
    sens_list = user.get("sensitivities", [])
    for k in ["fragrance","parabens","sulfates","phthalates","alcohol","silicones"]:
        feats[f"user_sens_{k}"] = int(k in sens_list)
    for st in ["oily","dry","combination","normal","sensitive"]:
        feats[f"user_skin_{st}"] = int(user.get("skin_type","") == st)
    for st in ["fair","light","medium","tan","deep"]:
        feats[f"user_tone_{st}"] = int(user.get("skin_tone","") == st)
    for s in ["natural","glam","editorial","no_makeup","dramatic"]:
        feats[f"user_style_{s}"] = int(user.get("style","") == s)

    # product features
    price     = float(product_row.get("price_usd", 0) or 0)
    rating    = float(product_row.get("rating", 0) or 0)
    n_reviews = float(product_row.get("num_reviews", 0) or 0)
    feats["price_usd"]       = price
    feats["log_price"]       = np.log1p(price)
    feats["prod_rating"]     = rating
    feats["log_num_reviews"] = np.log1p(n_reviews)
    feats["has_match"]       = int(not pd.isna(product_row.get("match_id", np.nan)))
    feats["cross_retailer_rating_mean"] = float(product_row.get("cross_retailer_rating_mean", 0) or 0)
    feats["is_makeup"]   = int(str(product_row.get("primary_domain","")).lower() == "makeup")
    feats["is_skincare"] = int(str(product_row.get("primary_domain","")).lower() == "skincare")
    feats["has_ingredients_text"] = int(
        bool(product_row.get("ingredients_text")) and
        not pd.isna(product_row.get("ingredients_text", np.nan))
    )
    feats["src_sephora"] = int(str(product_row.get("source","")).lower() == "sephora")
    feats["src_ulta"]    = int(str(product_row.get("source","")).lower() == "ulta")

    # category one-hots
    cat = str(product_row.get("category_unified","")).lower()
    for cat_name, cat_key in [
        ("moisturizers","cat_moisturizers"), ("face serums","cat_face_serums"),
        ("face primer","cat_face_primer"),   ("foundation","cat_foundation"),
        ("setting spray & powder","cat_setting_spray_and_powder"),
        ("blush","cat_blush"),               ("cleansers","cat_cleansers"),
        ("concealer","cat_concealer"),       ("eye cream","cat_eye_cream"),
        ("highlighter","cat_highlighter"),
    ]:
        feats[cat_key] = int(cat == cat_name)
    feats["cat_other"] = int(cat not in [
        "moisturizers","face serums","face primer","foundation",
        "setting spray & powder","blush","cleansers","concealer",
        "eye cream","highlighter"
    ])

    # interaction features
    budget = feats["user_budget"]
    feats["price_within_budget"] = int(price <= budget) if budget > 0 else 0
    feats["price_budget_ratio"]  = price / budget if budget > 0 else 9.0

    skin_type = user.get("skin_type","")
    skin_tone = user.get("skin_tone","")
    def _get_demo_rating(col_suffix):
        val = product_row.get(f"rating_skin_{col_suffix}")
        if val is None or pd.isna(val):
            return float(rating), 0
        return float(val), 1

    demo_rating_skin, is_real_skin = _get_demo_rating(skin_type)
    demo_rating_tone, is_real_tone = _get_demo_rating(skin_tone)
    feats["rating_for_user_skin_type"]         = demo_rating_skin
    feats["rating_for_user_skin_type_is_real"] = is_real_skin
    feats["rating_for_user_skin_tone"]         = demo_rating_tone
    feats["rating_for_user_skin_tone_is_real"] = is_real_tone

    highlights = str(product_row.get("highlights","") or "").lower()
    skin_kws = {"oily":["oil-free","oil free","mattifying","matte","pore"],
                "dry":["hydrating","hydration","moisturizing","dry skin","rich"],
                "combination":["balance","combination","normal to oily","normal to dry"],
                "normal":["all skin","all skin types","normal"],
                "sensitive":["sensitive","fragrance-free","gentle","calming","soothing"]}
    style_kws = {"natural":["tinted","sheer","natural","lightweight","bb"],
                 "glam":["full coverage","shimmer","glow","luminous","buildable"],
                 "editorial":["bold","pigmented","intense","vivid","dramatic"],
                 "no_makeup":["bare","minimal","transparent","invisible","skincare"],
                 "dramatic":["high coverage","long-lasting","waterproof","full coverage"]}
    feats["highlights_match_skin"]  = int(any(kw in highlights for kw in skin_kws.get(skin_type,[])))
    feats["highlights_match_style"] = int(any(kw in highlights for kw in style_kws.get(user.get("style",""),[])))
    template_cats = {c for c,_ in ROUTINE_TEMPLATES.get(user.get("style",""),[])}
    feats["category_matches_style"] = int(str(product_row.get("category_unified","")) in template_cats)
    feats["has_user_sensitivity_violation"] = int(any(
        _triggers_sensitivity(product_row.get("ingredients_text"), s) for s in sens_list
    ))

    # ingredient features
    ing_text = str(product_row.get("ingredients_text","") or "").lower()
    feats["ing_retinol"]    = int("retinol" in ing_text)
    feats["ing_niacinamide"]= int("niacinamide" in ing_text)
    feats["ing_vitamin_c"]  = int("vitamin c" in ing_text or "ascorbic acid" in ing_text)
    feats["ing_hyaluronic"] = int("hyaluronic acid" in ing_text)
    feats["ing_salicylic"]  = int("salicylic acid" in ing_text)
    feats["ing_glycolic"]   = int("glycolic acid" in ing_text)
    feats["ing_lactic"]     = int("lactic acid" in ing_text)
    feats["ing_ceramides"]  = int("ceramide" in ing_text)
    feats["ing_peptides"]   = int("peptide" in ing_text)
    feats["ing_spf_actives"]= int(any(x in ing_text for x in
                                  ["zinc oxide","titanium dioxide","avobenzone",
                                   "octinoxate","octocrylene"]))
    feats["ing_centella"]   = int("centella" in ing_text or "cica" in ing_text)
    feats["ing_caffeine"]   = int("caffeine" in ing_text)
    feats["ing_vitamin_e"]  = int("tocopherol" in ing_text or "vitamin e" in ing_text)
    feats["ing_squalane"]   = int("squalane" in ing_text)
    feats["ing_shea_butter"]= int("shea butter" in ing_text or "butyrospermum parkii" in ing_text)

    return feats


def _score_catalog(user, catalog_df, gb_model):
    """
    Score every product in the catalog for this user.
    Returns catalog_df with added columns:
      predicted_score, passes_hard_constraints, over_budget, sens_violation
    """
    user_d   = _parse_user(user)
    budget   = float(user_d.get("budget", 0))
    sens     = user_d.get("sensitivities", [])

    feat_rows = []
    for _, row in catalog_df.iterrows():
        feat_rows.append(_featurize_pair(user_d, row))

    feat_df = pd.DataFrame(feat_rows)

    # align columns to what the model expects
    model_features = gb_model.feature_names_in_ if hasattr(gb_model, "feature_names_in_") else feat_df.columns
    for col in model_features:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[model_features].fillna(0.0)

    raw_scores = gb_model.predict(feat_df)

    scored = catalog_df.copy().reset_index(drop=True)
    scored["raw_score"]      = raw_scores
    scored["clipped_score"]  = np.clip(raw_scores, 0, 1)
    scored["over_budget"]    = (scored["price_usd"].fillna(np.inf) > budget).astype(int)
    scored["sens_violation"] = scored["ingredients_text"].apply(
        lambda x: int(any(_triggers_sensitivity(x, s) for s in sens))
    )
    scored["passes_hard_constraints"] = (
        (scored["over_budget"] == 0) & (scored["sens_violation"] == 0)
    ).astype(int)
    scored["predicted_score"] = np.where(
        scored["passes_hard_constraints"] == 1,
        scored["clipped_score"], 0.0
    )
    return scored


def _solve_milp(user_id, budget, style, required_cats, optional_cats, candidates_df):
    """Core PuLP MILP solve for one user. Returns (status, selected_df)."""
    if candidates_df.empty:
        return "Infeasible", pd.DataFrame()

    model = pulp.LpProblem(f"routine_{user_id}", pulp.LpMaximize)
    candidates_df = candidates_df.copy().reset_index(drop=True)
    candidates_df["_key"] = candidates_df.index.astype(str)
    x = {k: pulp.LpVariable(f"x_{k}", cat="Binary") for k in candidates_df["_key"]}

    # objective
    model += pulp.lpSum(
        row["predicted_score"] * x[row["_key"]]
        for _, row in candidates_df.iterrows()
    )

    # budget
    model += pulp.lpSum(
        row["price_usd"] * x[row["_key"]]
        for _, row in candidates_df.iterrows()
    ) <= budget

    # per-category constraints
    for cat in set(required_cats) | set(optional_cats):
        cat_keys = [
            x[row["_key"]]
            for _, row in candidates_df.iterrows()
            if row.get("category_unified") == cat
        ]
        if not cat_keys:
            continue
        if cat in required_cats:
            model += pulp.lpSum(cat_keys) == 1
        else:
            model += pulp.lpSum(cat_keys) <= 1

    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)

    if pulp.LpStatus[model.status] != "Optimal":
        return "Infeasible", pd.DataFrame()

    selected_keys = {k for k, v in x.items() if (pulp.value(v) or 0) > 0.5}
    selected = candidates_df[candidates_df["_key"].isin(selected_keys)].drop(columns=["_key"])
    return "Optimal", selected


def _get_slot_candidates(cat, candidates_df):
    c = candidates_df[candidates_df["category_unified"] == cat]
    if len(c) == 0:
        alt = SLOT_ALTERNATIVES.get(cat)
        if alt:
            c = candidates_df[candidates_df["category_unified"] == alt]
    return c


def _plan_greedy(user_id, budget, style, template, candidates_df):
    """Greedy fallback planner (used when PuLP unavailable)."""
    req_cats  = {cat for cat, req in template if req}
    all_slots = [cat for cat in GREEDY_SLOT_PRIORITY
                 if cat in {c for c, _ in template}]
    for cat, _ in template:
        if cat not in all_slots:
            all_slots.append(cat)

    remaining  = budget
    selected   = []
    dropped    = []
    drop_reasons = {}

    for cat in all_slots:
        slot_cands = _get_slot_candidates(cat, candidates_df)
        slot_cands = slot_cands[slot_cands["price_usd"] <= remaining].sort_values(
            "predicted_score", ascending=False
        )
        if slot_cands.empty:
            if cat in req_cats:
                dropped.append(cat)
                drop_reasons[cat] = "no affordable candidates"
            continue
        best = slot_cands.iloc[0]
        selected.append(best)
        remaining -= best["price_usd"]

    return (pd.DataFrame(selected).drop_duplicates(subset=["source_id"]) if selected
            else pd.DataFrame()), dropped, drop_reasons


# ── Public API ───────────────────────────────────────────────────────────────

def plan_routine(user_profile, catalog_df, gb_model,
                 safety_filter_df=None, budget_aware=True):
    """
    Plan a personalised beauty routine for one user.

    Parameters
    ----------
    user_profile : dict or pd.Series
        Must contain: user_id, skin_type, skin_tone, budget, style, sensitivities
        sensitivities may be a list or comma-separated string.

    catalog_df : pd.DataFrame
        merged_products_with_personalization.csv loaded into memory.

    gb_model : sklearn GradientBoostingRegressor
        Loaded from gb_model.pkl.

    safety_filter_df : pd.DataFrame or None
        Pre-computed safety flags (safety_filter_catalog.csv from Day 5).
        If provided, products flagged for the user's sensitivities are
        excluded before scoring (faster than ingredient-text scanning).
        If None, the module re-runs keyword matching on ingredients_text.

    budget_aware : bool
        If False, run greedy without budget constraint (for comparison only).
        Always True in production.

    Returns
    -------
    dict with keys:
        routine              pd.DataFrame  — selected products
        status               str           — "Optimal" | "Infeasible"
        total_cost           float
        total_compatibility  float         — sum of GBM predicted scores
        slot_coverage        float         — fraction of required slots filled
        dropped_slots        list[str]
        drop_reasons         dict[str, str]
        n_products           int
        planner_used         str           — "milp" | "greedy_fallback"
    """
    user   = _parse_user(user_profile)
    uid    = user.get("user_id", "live_user")
    budget = float(user.get("budget", 0))
    style  = user.get("style", "natural")
    sens   = user.get("sensitivities", [])

    if style not in ROUTINE_TEMPLATES:
        style = "natural"

    template  = ROUTINE_TEMPLATES[style]
    req_cats  = [cat for cat, req in template if req]
    opt_cats  = [cat for cat, req in template if not req]

    # ── Apply pre-computed safety filter if available ──
    working_catalog = catalog_df.copy()
    if safety_filter_df is not None and not safety_filter_df.empty:
        for s in sens:
            flag_col = f"flagged_{s}"
            if flag_col in safety_filter_df.columns:
                flagged_ids = set(
                    safety_filter_df.loc[safety_filter_df[flag_col] == True,
                                         "source_id"].astype(str)
                )
                working_catalog = working_catalog[
                    ~working_catalog["source_id"].astype(str).isin(flagged_ids)
                ]

    # ── Score the catalog ──
    scored = _score_catalog(user, working_catalog, gb_model)

    # ── Filter to passing candidates in template categories ──
    template_cats = set()
    for cat, _ in template:
        template_cats.add(cat)
        alt = SLOT_ALTERNATIVES.get(cat)
        if alt:
            template_cats.add(alt)

    candidates = scored[
        (scored["passes_hard_constraints"] == 1) &
        (scored["category_unified"].isin(template_cats))
    ].copy()

    # ── MILP solve with relaxation fallback ──
    dropped_slots = []
    drop_reasons  = {}
    planner_used  = "greedy_fallback"

    if _PULP_AVAILABLE and budget_aware:
        planner_used = "milp"
        status, selected = _solve_milp(uid, budget, style, req_cats, opt_cats, candidates)

        if status != "Optimal":
            current_req = list(req_cats)
            for slot_to_drop in RELAXATION_PRIORITY:
                if slot_to_drop not in current_req:
                    continue
                slot_cands = _get_slot_candidates(slot_to_drop, candidates)
                n_pass = len(slot_cands[slot_cands["price_usd"] <= budget])
                if len(slot_cands) == 0:
                    reason = f"no products pass sensitivity filter"
                elif n_pass == 0:
                    reason = f"all sensitivity-passing products exceed budget ${budget}"
                else:
                    reason = f"including slot makes overall budget infeasible"
                dropped_slots.append(slot_to_drop)
                drop_reasons[slot_to_drop] = reason
                current_req.remove(slot_to_drop)
                status, selected = _solve_milp(
                    uid, budget, style, current_req, opt_cats, candidates
                )
                if status == "Optimal":
                    break
    else:
        # greedy fallback
        selected, dropped_slots, drop_reasons = _plan_greedy(
            uid, budget, style, template, candidates
        )
        status = "Optimal" if not selected.empty else "Infeasible"

    # ── Compute summary metrics ──
    if not selected.empty:
        total_cost   = selected["price_usd"].sum()
        total_compat = selected["predicted_score"].sum()
        n_products   = len(selected)
        filled_req   = set(selected["category_unified"]) & set(req_cats)
        slot_coverage = len(filled_req) / len(req_cats) if req_cats else 1.0
    else:
        status        = "Infeasible"
        total_cost    = 0.0
        total_compat  = 0.0
        n_products    = 0
        slot_coverage = 0.0

    return {
        "routine"             : selected,
        "status"              : status,
        "total_cost"          : round(total_cost, 2),
        "total_compatibility" : round(total_compat, 6),
        "slot_coverage"       : round(slot_coverage, 4),
        "dropped_slots"       : dropped_slots,
        "drop_reasons"        : drop_reasons,
        "n_products"          : n_products,
        "planner_used"        : planner_used,
    }
