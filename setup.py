"""
setup.py  —  One-shot: run pipeline + train model + verify.
Run this before starting the server.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*50)
    print("  Makeup Mastermind — Setup")
    print("="*50)

    print("\n[1/3] Running data pipeline...")
    from backend.data_pipeline import run_pipeline
    products = run_pipeline()
    print(f"      {len(products)} products processed")

    print("\n[2/3] Training ML ranking model...")
    from backend.ranking_model import train
    model, rmse = train(products)
    print(f"      Validation RMSE: {rmse:.4f}")

    print("\n[3/3] Verifying predictions...")
    from backend.ranking_model import predict
    test_profiles = [
        {"style":"glam",    "skin_type":"oily",      "budget":200, "sensitivities":["parabens"]},
        {"style":"minimal", "skin_type":"sensitive",  "budget":60,  "sensitivities":["fragrance"]},
        {"style":"natural", "skin_type":"dry",        "budget":100, "sensitivities":[]},
    ]
    import random; random.seed(1)
    sample = random.sample(products, min(5, len(products)))
    for user in test_profiles[:1]:
        print(f"\n  Profile: {user['style']} / {user['skin_type']} / ${user['budget']}")
        for p in sample:
            s = predict(model, user, p)
            print(f"    [{s:.3f}]  {p['brand']} — {p['name']} (${p['price']}) [{p['category']}]")

    print("\n" + "="*50)
    print("  Setup complete!")
    print("  Start server:  python -m backend.app")
    print("  Open UI:       frontend/index.html")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
