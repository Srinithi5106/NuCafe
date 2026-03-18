"""
train_ai.py — Trains AI on the same 126-item menu. Adds Apriori association rules.
Run after seed.py: python train_ai.py
"""
import pandas as pd
import numpy as np
import random
import pickle
import json
from collections import defaultdict
from itertools import combinations

random.seed(42)   # same seed = same menu names as seed.py

CATEGORIES = {
    "South Indian": ["Dosa", "Idli", "Vada", "Pongal", "Uttapam", "Bhat", "Rava Kesari"],
    "North Indian": ["Paneer Tikka", "Dal Makhani", "Naan", "Chole Bhature", "Paratha", "Butter Chicken"],
    "Biryani":      ["Hyderabadi", "Ambur", "Donne", "Lucknowi", "Egg Biryani", "Veg Biryani"],
    "Beverage":     ["Filter Coffee", "Masala Chai", "Badam Milk", "Cold Coffee", "Fruit Juice", "Lassi"],
    "Fastfood":     ["Burger", "Pizza", "Pasta", "Sandwich", "French Fries", "Momos"],
    "Snack":        ["Samosa", "Kachori", "Bhel Puri", "Pani Puri", "Vada Pav", "Pakoda"],
}

CATEGORY_IMAGES = {
    "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
    "North Indian": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60",
    "Biryani":      "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60",
    "Beverage":     "https://images.unsplash.com/photo-1609951651556-5334e2706168?w=600&auto=format&fit=crop&q=60",
    "Fastfood":     "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop&q=60",
    "Snack":        "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=600&auto=format&fit=crop&q=60",
}

# Curated pairing seeds — so Apriori learns real pairs from day 1
PAIR_SEEDS = {
    "South Indian": ["Beverage", "Snack"],
    "Biryani":      ["Beverage", "Snack"],
    "North Indian": ["Beverage"],
    "Fastfood":     ["Beverage", "Snack"],
    "Snack":        ["Beverage"],
    "Beverage":     ["Snack", "South Indian"],
}

def train_and_save():
    print("🎬 Training AI Engine...")

    # ── 1. Build menu (same as seed.py) ──────────────────────────────────────
    menu_items = []
    classics = [
        ("CTR Benne Masala Dosa",        "South Indian", 95,  10),
        ("Filter Coffee",                "Beverage",     40,   5),
        ("MTR Rava Idli",                "South Indian", 80,   8),
        ("Meghana Chicken Biryani",      "Biryani",     320,  20),
        ("Truffles All American Burger", "Fastfood",    290,  15),
        ("VV Puram Chat Basket",         "Snack",        80,   5),
    ]
    for name, cat, price, prep in classics:
        menu_items.append({"food_name": name, "category": cat, "price": price,
                           "prep_time": prep, "image": CATEGORY_IMAGES[cat], "is_new": "no"})

    for i in range(120):
        cat    = random.choice(list(CATEGORIES.keys()))
        suffix = random.choice(CATEGORIES[cat])
        name   = f"Special {cat} {suffix} {i+1}"
        menu_items.append({
            "food_name": name, "category": cat,
            "price":     random.randint(50, 450),
            "prep_time": random.randint(5, 25),
            "image":     CATEGORY_IMAGES[cat],
            "is_new":    random.choice(["yes", "no", "no", "no", "no"]),
        })

    with open('menu.json', 'w') as f:
        json.dump(menu_items, f, indent=4)
    print(f"📝 menu.json: {len(menu_items)} items")

    menu_df     = pd.DataFrame(menu_items)
    food_names  = menu_df['food_name'].tolist()
    food_to_idx = {name: i for i, name in enumerate(food_names)}
    cat_map     = dict(zip(menu_df['food_name'], menu_df['category']))
    cat_to_items = defaultdict(list)
    for m in menu_items:
        cat_to_items[m['category']].append(m['food_name'])

    # ── 2. Synthetic order history with curated-pair seeding ─────────────────
    num_users = 1000
    orders    = []           # for MF training
    baskets   = defaultdict(set)  # for Apriori

    for user_id in range(1, 101):
        pref_cat  = random.choice(list(CATEGORIES.keys()))
        cat_items = menu_df[menu_df['category'] == pref_cat]['food_name'].tolist()
        session   = []

        for _ in range(random.randint(5, 10)):
            food = random.choice(cat_items) if random.random() < 0.7 \
                   else random.choice(food_names)
            session.append(food)

            # Seed complementary companion
            for paired_cat in PAIR_SEEDS.get(cat_map[food], []):
                if random.random() < 0.7 and cat_to_items[paired_cat]:
                    companion = random.choice(cat_to_items[paired_cat])
                    session.append(companion)

        for food in session:
            orders.append({"user_id": user_id,
                           "food_idx": food_to_idx[food],
                           "rating":   random.uniform(3.5, 5.0)})
            baskets[user_id].add(food)

    # ── 3. Matrix Factorisation ───────────────────────────────────────────────
    print("🧠 Training Matrix Factorisation...")
    latent = 12
    P = np.random.normal(scale=1./latent, size=(num_users, latent))
    Q = np.random.normal(scale=1./latent, size=(len(food_names), latent))

    for epoch in range(20):
        for o in orders:
            u, f, r = o['user_id'], o['food_idx'], o['rating']
            err   = r - np.dot(P[u], Q[f])
            P[u] += 0.01 * (err * Q[f])
            Q[f] += 0.01 * (err * P[u])
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20")

    # ── 4. Apriori association rules ─────────────────────────────────────────
    print("🔗 Mining association rules...")
    baskets_list = list(baskets.values())
    N = len(baskets_list)

    item_sup = defaultdict(int)
    for b in baskets_list:
        for item in b:
            item_sup[item] += 1

    freq = {item for item, cnt in item_sup.items() if cnt / N >= 0.02}

    pair_sup = defaultdict(int)
    for b in baskets_list:
        fb = b & freq
        for a, x in combinations(sorted(fb), 2):
            pair_sup[(a, x)] += 1

    rules = defaultdict(list)
    for (a, x), cnt in pair_sup.items():
        sup = cnt / N
        if sup < 0.02:
            continue
        ca = sup / (item_sup[a] / N)
        if ca >= 0.15:
            rules[a].append((x, ca, ca / (item_sup[x] / N + 1e-9)))
        cx = sup / (item_sup[x] / N)
        if cx >= 0.15:
            rules[x].append((a, cx, cx / (item_sup[a] / N + 1e-9)))

    for k in rules:
        rules[k].sort(key=lambda r: r[2], reverse=True)

    total_rules = sum(len(v) for v in rules.values())
    print(f"  Rules mined: {total_rules}")

    # ── 5. Wait Time Model ───────────────────────────────────────────────────
    print("⏱️  Training Wait Time Model...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    CATEGORY_BASE = {
        "South Indian": 10, "North Indian": 20, "Biryani": 25,
        "Beverage": 5,      "Fastfood": 12,    "Snack": 8,
    }

    # Synthesize 5000 order records
    rng = np.random.default_rng(42)
    wt_records = []
    categories_list = list(CATEGORY_BASE.keys())
    for _ in range(5000):
        cat      = random.choice(categories_list)
        bp       = max(3, CATEGORY_BASE[cat] + random.randint(-3, 5))
        hour     = random.randint(8, 22)
        is_wknd  = random.choice([0,0,0,1])
        queue    = random.randint(0, 20)
        rush     = 1.6 if 12<=hour<=14 else (1.5 if 19<=hour<=21 else (1.1 if 9<=hour<=11 else 1.0))
        wait     = max(3, round(bp*rush + queue*1.2 + (3 if is_wknd else 0) + float(rng.normal(0,2)), 1))
        wt_records.append([bp, cat, hour, queue, is_wknd, wait])

    import pandas as _pd
    wt_df = _pd.DataFrame(wt_records, columns=['base_prep_time','category','hour_of_day','queue_length','is_weekend','actual_wait'])
    le_wt = LabelEncoder()
    wt_df['category_enc'] = le_wt.fit_transform(wt_df['category'])
    Xw = wt_df[['base_prep_time','category_enc','hour_of_day','queue_length','is_weekend']]
    yw = wt_df['actual_wait']
    wt_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    wt_model.fit(Xw, yw)
    print(f"  Wait model trained on {len(wt_df)} samples")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    with open('simple_data.pkl', 'wb') as f:
        pickle.dump({
            'P': P, 'Q': Q,
            'food_to_idx':   food_to_idx,
            'features':      latent,
            'assoc_rules':   dict(rules),
            'wait_model':    wt_model,
            'wait_encoder':  le_wt,
        }, f)

    print(f"✅ Done! simple_data.pkl saved.")
    print(f"   Items: {len(food_names)} | Users: {num_users} | Rules: {total_rules}")

if __name__ == "__main__":
    train_and_save()