"""
seed.py — Seeds the exact same 126-item menu as train_ai.py into Neon DB.
Run once: python seed.py
"""
from sqlalchemy import text, Column, Integer, String, Float
from database import Session, engine, Base

class MenuItem(Base):
    __tablename__ = 'menu_items'
    id        = Column(Integer, primary_key=True)
    food_name = Column(String, unique=True, nullable=False)
    category  = Column(String, nullable=False)
    price     = Column(Float, nullable=False)
    prep_time = Column(Integer, default=15)
    image     = Column(String, nullable=True)
    is_new    = Column(String, default="no")
    __table_args__ = {'extend_existing': True}

CATEGORY_IMAGES = {
    "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
    "North Indian": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60",
    "Biryani":      "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60",
    "Beverage":     "https://images.unsplash.com/photo-1609951651556-5334e2706168?w=600&auto=format&fit=crop&q=60",
    "Fastfood":     "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop&q=60",
    "Snack":        "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=600&auto=format&fit=crop&q=60",
}

# Exact same 126 items as train_ai.py generates — classics + 120 specials
import random
random.seed(42)   # fixed seed so names are always identical to train_ai.py

CATEGORIES = {
    "South Indian": ["Dosa", "Idli", "Vada", "Pongal", "Uttapam", "Bhat", "Rava Kesari"],
    "North Indian": ["Paneer Tikka", "Dal Makhani", "Naan", "Chole Bhature", "Paratha", "Butter Chicken"],
    "Biryani":      ["Hyderabadi", "Ambur", "Donne", "Lucknowi", "Egg Biryani", "Veg Biryani"],
    "Beverage":     ["Filter Coffee", "Masala Chai", "Badam Milk", "Cold Coffee", "Fruit Juice", "Lassi"],
    "Fastfood":     ["Burger", "Pizza", "Pasta", "Sandwich", "French Fries", "Momos"],
    "Snack":        ["Samosa", "Kachori", "Bhel Puri", "Pani Puri", "Vada Pav", "Pakoda"],
}

CLASSICS = [
    ("CTR Benne Masala Dosa",        "South Indian", 95,  10),
    ("Filter Coffee",                "Beverage",     40,   5),
    ("MTR Rava Idli",                "South Indian", 80,   8),
    ("Meghana Chicken Biryani",      "Biryani",     320,  20),
    ("Truffles All American Burger", "Fastfood",    290,  15),
    ("VV Puram Chat Basket",         "Snack",        80,   5),
]

DATA = []
for name, cat, price, prep in CLASSICS:
    DATA.append({"food_name": name, "category": cat, "price": price,
                 "prep_time": prep, "image": CATEGORY_IMAGES[cat], "is_new": "no"})

for i in range(120):
    cat    = random.choice(list(CATEGORIES.keys()))
    suffix = random.choice(CATEGORIES[cat])
    name   = f"Special {cat} {suffix} {i+1}"
    DATA.append({
        "food_name": name,
        "category":  cat,
        "price":     random.randint(50, 450),
        "prep_time": random.randint(5, 25),
        "image":     CATEGORY_IMAGES[cat],
        "is_new":    random.choice(["yes", "no", "no", "no", "no"]),
    })

def reset_and_seed():
    print("🚀 Starting reset...")
    with engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE TABLE menu_items RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE orders RESTART IDENTITY CASCADE;"))
            conn.commit()
            print("🧹 Tables truncated.")
        except Exception:
            conn.rollback()

    Base.metadata.create_all(engine)
    print("🏗️  Schema ready.")

    with Session() as db:
        inserted = skipped = 0
        for item in DATA:
            exists = db.query(MenuItem).filter_by(food_name=item["food_name"]).first()
            if exists:
                for k, v in item.items():
                    setattr(exists, k, v)
                skipped += 1
            else:
                db.add(MenuItem(**item))
                inserted += 1
        db.commit()

    cats = {}
    for d in DATA:
        cats[d["category"]] = cats.get(d["category"], 0) + 1

    print(f"\n✅ Done! Inserted: {inserted} | Updated: {skipped}")
    print("=" * 40)
    for cat, count in sorted(cats.items()):
        print(f"   {cat:<15} {count} items")
    print(f"   {'TOTAL':<15} {len(DATA)} items")
    print("=" * 40)
    print("▶  Next: python train_ai.py")

if __name__ == "__main__":
    reset_and_seed()