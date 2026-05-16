"""
data_pipeline.py  —  Step 1: Load, clean, and featurize the Sephora dataset.

Outputs data/products.json used by all other modules.
"""
import pandas as pd, numpy as np, json, re, os

RAW_CSV  = os.path.join(os.path.dirname(__file__), "../data/sephora_website_dataset.csv")
OUT_JSON = os.path.join(os.path.dirname(__file__), "../data/products.json")

MAKEUP_CATS = {
    "Lipstick","Liquid Lipstick","Lip Gloss","Lip Liner","Lip Stain","Lip Plumper",
    "Lip Balm & Treatment","Lip Balms & Treatments","Foundation","BB & CC Cream",
    "BB & CC Creams","Tinted Moisturizer","Concealer","Blush","Bronzer","Highlighter",
    "Contour","Eyeshadow","Eye Palettes","Eyeliner","Mascara","Eyebrow","Eye Primer",
    "Face Primer","Setting Spray & Powder",
}
CAT_ALIASES = {"BB & CC Creams":"BB & CC Cream","Lip Balms & Treatments":"Lip Balm & Treatment"}
CAT_TO_STEP = {
    "Face Primer":"face_base","Foundation":"face_base","BB & CC Cream":"face_base",
    "Tinted Moisturizer":"face_base","Concealer":"face_base",
    "Blush":"face_color","Bronzer":"face_color","Highlighter":"face_color","Contour":"face_color",
    "Setting Spray & Powder":"face_finish","Eye Primer":"eye_base",
    "Eye Palettes":"eye_color","Eyeshadow":"eye_color","Eyeliner":"eye_define",
    "Eyebrow":"eye_define","Mascara":"eye_finish",
    "Lipstick":"lip","Liquid Lipstick":"lip","Lip Gloss":"lip","Lip Liner":"lip",
    "Lip Stain":"lip","Lip Balm & Treatment":"lip","Lip Plumper":"lip",
}

INGREDIENT_BLACKLIST = {
    "fragrance":  ["fragrance","parfum","perfume","linalool","limonene","geraniol",
                   "citronellol","eugenol","coumarin","benzyl alcohol"],
    "parabens":   ["methylparaben","ethylparaben","propylparaben","butylparaben","isobutylparaben"],
    "sulfates":   ["sodium lauryl sulfate","sodium laureth sulfate","ammonium lauryl sulfate","sls","sles"],
    "alcohol":    ["alcohol denat","denatured alcohol","sd alcohol","isopropyl alcohol"],
    "gluten":     ["triticum vulgare","wheat","hordeum vulgare","barley","secale cereale","avena sativa"],
    "formaldehyde_releasers": ["dmdm hydantoin","imidazolidinyl urea","quaternium-15","bronopol"],
}
BLACKLIST_SEVERITY = {
    "fragrance":0.6,"parabens":0.8,"sulfates":0.5,
    "alcohol":0.4,"gluten":0.9,"formaldehyde_releasers":1.0,
}

def parse_ingredients(raw):
    if pd.isna(raw) or str(raw).strip().lower() in ("unknown",""): return []
    parts = re.split(r"[,\n]", str(raw))
    out = []
    for p in parts:
        p = re.sub(r"[•\-\*]","",p).strip()
        if ":" in p and len(p) > 50: continue
        if 3 <= len(p) <= 60: out.append(p.lower().strip())
    return out

def check_safety(ings):
    blob = " ".join(ings)
    return {c:[k for k in kws if k in blob] for c,kws in INGREDIENT_BLACKLIST.items()
            if any(k in blob for k in kws)}

def safety_score(flags):
    if not flags: return 1.0
    return max(0.0, 1.0 - min(sum(BLACKLIST_SEVERITY.get(c,.5) for c in flags), 1.0))

def price_tier(p):
    return "budget" if p<=20 else "mid" if p<=45 else "high" if p<=80 else "luxury"

def finish_features(details):
    t = (str(details) if details else "").lower()
    return {
        "is_matte":      int(any(w in t for w in ["matte","oil control","shine-free"])),
        "is_dewy":       int(any(w in t for w in ["dewy","luminous","glow","radiant"])),
        "is_longwear":   int(any(w in t for w in ["long-wear","long wear","all day","24 hour"])),
        "is_hydrating":  int(any(w in t for w in ["hydrating","moisturizing","nourishing"])),
        "is_spf":        int("spf" in t or "sunscreen" in t),
        "is_buildable":  int(any(w in t for w in ["buildable","full coverage"])),
        "is_sheer":      int(any(w in t for w in ["sheer","light coverage","lightweight"])),
        "is_volumizing": int(any(w in t for w in ["volumizing","volume","thickening"])),
        "is_waterproof": int("waterproof" in t or "water-resistant" in t),
    }

def run_pipeline(csv_path=RAW_CSV, out_path=OUT_JSON):
    print(f"[pipeline] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df["category"].isin(MAKEUP_CATS)].copy()
    df["category"] = df["category"].replace(CAT_ALIASES)
    df["price"]   = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").fillna(3.5)
    df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0).astype(int)
    products = []
    for _, row in df.iterrows():
        ings  = parse_ingredients(row.get("ingredients",""))
        flags = check_safety(ings)
        feats = finish_features(row.get("details",""))
        products.append({
            "id": str(row["id"]), "brand": str(row["brand"]),
            "category": str(row["category"]),
            "step_group": CAT_TO_STEP.get(str(row["category"]),"other"),
            "name": str(row["name"]),
            "price": float(row["price"]), "price_tier": price_tier(float(row["price"])),
            "rating": float(row["rating"]), "reviews": int(row["number_of_reviews"]),
            "ingredients": ings, "safety_flags": flags, "safety_score": safety_score(flags),
            **feats,
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w") as f: json.dump(products, f, indent=2)
    print(f"[pipeline] Saved {len(products)} products → {out_path}")
    return products

if __name__ == "__main__":
    run_pipeline()