import pandas as pd
import numpy as np
import random
import pickle
import json
import os
from datetime import datetime, timedelta

def train_and_save():
    print("🎬 Starting Enhanced AI Training Engine...")

    # 1. GENERATE A RICH MENU (120+ Items)
    categories = {
        "South Indian": ["Dosa", "Idli", "Vada", "Pongal", "Uttapam", "Bhat", "Rava Kesari"],
        "North Indian": ["Paneer Tikka", "Dal Makhani", "Naan", "Chole Bhature", "Paratha", "Butter Chicken"],
        "Biryani": ["Hyderabadi", "Ambur", "Donne", "Lucknowi", "Egg Biryani", "Veg Biryani"],
        "Beverage": ["Filter Coffee", "Masala Chai", "Badam Milk", "Cold Coffee", "Fruit Juice", "Lassi"],
        "Fastfood": ["Burger", "Pizza", "Pasta", "Sandwich", "French Fries", "Momos"],
        "Snack": ["Samosa", "Kachori", "Bhel Puri", "Pani Puri", "Vada Pav", "Pakoda"]
    }

    # High-quality fallback image mapping - ALL CUSTOMIZED
    category_images = {
        "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZG9zYXxlbnwwfHwwfHx8MA%3D%3D",
        "North Indian": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fHBhbmVlcnxlbnwwfHwwfHx8MA%3D%3D",
        "Biryani": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyeWFuaXxlbnwwfHwwfHx8MA%3D%3D",
        "Beverage": "https://images.unsplash.com/photo-1609951651556-5334e2706168?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDN8fGZvb2R8ZW58MHx8MHx8fDA%3D",
        "Fastfood": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8Zm9vZHxlbnwwfHwwfHx8MA%3D%3D",
        "Snack": "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fGZvb2R8ZW58MHx8MHx8fDA%3D"
    }

    menu_items = []
    
    # Add the "Classics" first
    classics = [
        ("CTR Benne Masala Dosa", "South Indian", 95, 10),
        ("Filter Coffee", "Beverage", 40, 5),
        ("MTR Rava Idli", "South Indian", 80, 8),
        ("Meghana Chicken Biryani", "Biryani", 320, 20),
        ("Truffles All American Burger", "Fastfood", 290, 15),
        ("VV Puram Chat Basket", "Snack", 80, 5)
    ]
    
    for name, cat, price, prep in classics:
        menu_items.append({
            "food_name": name,
            "category": cat,
            "price": price,
            "prep_time": prep,
            "image": category_images[cat],
            "is_new": "no"
        })

    # Generate 120+ additional items
    for i in range(120):
        cat = random.choice(list(categories.keys()))
        suffix = random.choice(categories[cat])
        name = f"Special {cat} {suffix} {i+1}"
        menu_items.append({
            "food_name": name,
            "category": cat,
            "price": random.randint(50, 450),
            "prep_time": random.randint(5, 25),
            "image": category_images[cat],
            "is_new": random.choice(["yes", "no", "no", "no", "no"])
        })

    with open('menu.json', 'w') as f:
        json.dump(menu_items, f, indent=4)
    print(f"📝 Created menu.json with {len(menu_items)} items.")

    # 2. GENERATE SYNTHETIC TRAINING DATA (500 Orders)
    menu_df = pd.DataFrame(menu_items)
    food_names = menu_df['food_name'].tolist()
    food_to_idx = {name: i for i, name in enumerate(food_names)}
    
    num_users = 1000
    latent_features = 12
    
    # Create synthetic "clusters" of user behavior
    orders = []
    for user_id in range(1, 101): # Simulate 100 active users
        pref_cat = random.choice(list(categories.keys()))
        cat_items = menu_df[menu_df['category'] == pref_cat]['food_name'].tolist()
        
        for _ in range(random.randint(5, 10)):
            if random.random() < 0.7: # 70% chance to pick from preference
                food = random.choice(cat_items)
            else:
                food = random.choice(food_names)
            
            orders.append({
                "user_id": user_id,
                "food_idx": food_to_idx[food],
                "rating": random.uniform(3.5, 5.0)
            })

    # 3. TRAIN MATRIX FACTORIZATION
    print("🧠 Training Matrix Factorization Model...")
    P = np.random.normal(scale=1./latent_features, size=(num_users, latent_features))
    Q = np.random.normal(scale=1./latent_features, size=(len(food_names), latent_features))

    learning_rate = 0.01
    for epoch in range(20):
        for order in orders:
            u, f, r = order['user_id'], order['food_idx'], order['rating']
            prediction = np.dot(P[u], Q[f])
            error = r - prediction
            
            P[u] += learning_rate * (error * Q[f])
            Q[f] += learning_rate * (error * P[u])

    # 4. SAVE THE BRAIN
    ai_brain = {
        'P': P,
        'Q': Q,
        'food_to_idx': food_to_idx,
        'features': latent_features
    }

    with open('simple_data.pkl', 'wb') as f:
        pickle.dump(ai_brain, f)
    
    print("✅ Training Complete! 'simple_data.pkl' and 'menu.json' updated.")

if __name__ == "__main__":
    train_and_save()