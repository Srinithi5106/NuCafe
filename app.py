import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
import random as _random
from collections import defaultdict
from sqlalchemy import func
from database import Session, User, Order, KitchenQueue, hash_password, check_password

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="NuCafé", initial_sidebar_state="expanded")

if 'user_id'  not in st.session_state: st.session_state.user_id  = None
if 'username' not in st.session_state: st.session_state.username = None
if 'cart'     not in st.session_state: st.session_state.cart     = {}

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    if not os.path.exists('menu.json') or not os.path.exists('simple_data.pkl'):
        import train_ai
        train_ai.train_and_save()
    with open('menu.json', 'r') as f:
        menu_data = json.load(f)
    menu = pd.DataFrame(menu_data)
    if not menu.empty:
        menu['prep_norm']  = (menu['prep_time'] - menu['prep_time'].min()) / (menu['prep_time'].max() - menu['prep_time'].min() + 1e-5)
        menu['price_norm'] = (menu['price']     - menu['price'].min())     / (menu['price'].max()     - menu['price'].min()     + 1e-5)
        import hashlib
        def gen_rating(name):
            h = int(hashlib.md5(name.encode()).hexdigest(), 16)
            return round(3.5 + (h % 16) * 0.1, 1)
        menu['rating'] = menu['food_name'].apply(gen_rating)
    with open('simple_data.pkl', 'rb') as f:
        brain = pickle.load(f)
    return menu, brain

menu_df, ai = load_engine()
all_items   = menu_df['food_name'].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 2b. WAIT TIME MODEL
# ─────────────────────────────────────────────────────────────────────────────
# ── IST Timezone (UTC+5:30) ───────────────────────────────────────────────────
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
def now_ist():
    return datetime.datetime.now(IST)

def get_queue_length():
    """Count orders placed in last 30 minutes still in kitchen."""
    try:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)
        with Session() as db:
            count = db.query(KitchenQueue).filter(
                KitchenQueue.status == 'pending',
                KitchenQueue.created_at >= cutoff
            ).count()
            return min(count, 20)
    except Exception:
        return 5

def predict_wait_time(food_name, category):
    """Use Random Forest model to predict wait time in minutes."""
    if 'wait_model' not in ai or 'wait_encoder' not in ai:
        row = menu_df[menu_df['food_name'] == food_name]
        return int(row.iloc[0]['prep_time']) if not row.empty else 15
    row = menu_df[menu_df['food_name'] == food_name]
    base_prep = int(row.iloc[0]['prep_time']) if not row.empty else 15
    import pandas as _pd
    now        = now_ist()
    hour       = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0
    queue      = get_queue_length()
    try:
        cat_enc = ai['wait_encoder'].transform([category])[0]
    except Exception:
        cat_enc = 0
    X = _pd.DataFrame([[base_prep, cat_enc, hour, queue, is_weekend]],
                      columns=['base_prep_time','category_enc','hour_of_day','queue_length','is_weekend'])
    pred = ai['wait_model'].predict(X)[0]
    return max(3, int(round(pred)))

def wait_color(mins):
    if mins <= 10:  return "#46D369", "Fast"
    if mins <= 20:  return "#F5A623", "Moderate"
    return "#E50914", "Slow"


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.user_id is None:
    st.markdown("""
        <div style='text-align:center; padding:50px 0;'>
            <h1 style='color:#E50914; font-size:5rem; margin-bottom:0;'>NuCafé</h1>
            <p style='color:#aaa; font-size:1.2rem;'>Unlimited Cravings. One Subscription.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True):
            mode = st.tabs(["Sign In", "New Account"])
            with mode[0]:
                u = st.text_input("Username", key="login_u")
                p = st.text_input("Password", type="password", key="login_p")
                if st.button("Sign In", use_container_width=True, type="primary"):
                    with Session() as db:
                        user = db.query(User).filter_by(username=u).first()
                        if user and check_password(p, user.password):
                            st.session_state.user_id  = user.id
                            st.session_state.username = u
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
            with mode[1]:
                u2 = st.text_input("Choose Username")
                p2 = st.text_input("Create Password", type="password")
                if st.button("Start Membership", use_container_width=True):
                    with Session() as db:
                        if db.query(User).filter_by(username=u2).first():
                            st.error("User exists!")
                        else:
                            db.add(User(username=u2, password=hash_password(p2)))
                            db.commit()
                            st.success("Welcome! Now Sign In.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .stApp { background-color: #141414; color: white; }
    .section-title {
        font-size:1.6rem; font-weight:700; color:#E5E5E5;
        margin:40px 0 15px 0; display:flex; align-items:center; gap:12px;
    }
    .food-card {
        background:#1f1f1f; border-radius:8px; overflow:hidden;
        transition:transform 0.4s cubic-bezier(0.165,0.84,0.44,1), box-shadow 0.4s;
        padding-bottom:12px; height:100%; border:1px solid #333;
    }
    .food-card:hover {
        transform:scale(1.08); box-shadow:0 10px 20px rgba(0,0,0,0.5);
        border-color:#E50914; z-index:10;
    }
    .card-img-container { width:100%; height:160px; background:linear-gradient(45deg,#222,#333); overflow:hidden; }
    .card-img { width:100%; height:100%; object-fit:cover; display:block; background-color:#222; }
    .match-tag { color:#46D369; font-weight:800; font-size:0.9rem; padding:10px 12px 2px 12px; }
    .card-content { padding:0 12px; }
    .food-title { margin:4px 0; font-size:1.05rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .stButton > button {
        background-color:transparent !important; border:1px solid #666 !important;
        color:white !important; border-radius:4px !important; font-size:0.8rem !important; transition:0.2s !important;
    }
    .stButton > button:hover { background-color:white !important; color:black !important; border-color:white !important; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_user_order_categories(user_id):
    with Session() as db:
        orders = db.query(Order.category).filter_by(user_id=user_id).all()
        return [o[0] for o in orders]

@st.cache_data(ttl=300)
def get_global_trending_map():
    with Session() as db:
        rows = db.query(Order.food_name).order_by(Order.timestamp.desc()).limit(100).all()
        if not rows: return {}
        counts = pd.Series([o[0] for o in rows]).value_counts()
        return (counts / counts.max()).to_dict()

def get_match_score(food_name, user_categories):
    item_rows = menu_df[menu_df['food_name'] == food_name]
    if item_rows.empty: return 50
    if not st.session_state.user_id: return 50
    item = item_rows.iloc[0]
    u_idx   = st.session_state.user_id % 1000
    f_idx   = ai['food_to_idx'].get(food_name, 0)
    collab  = np.dot(ai['P'][u_idx], ai['Q'][f_idx]) / 5.0
    cat_match = user_categories.count(item['category']) / len(user_categories) if user_categories else 0.5
    prep_score = 1.0 - item.get('prep_norm', 0.5)
    content    = (0.7 * cat_match) + (0.3 * prep_score)
    trend      = get_global_trending_map().get(food_name, 0.0)
    new_bonus  = 0.15 if item.get('is_new') == 'yes' else 0.0
    final      = (0.5 * collab) + (0.3 * content) + (0.2 * trend) + new_bonus
    return min(max(int(final * 100), 12), 99)

def get_trending_items():
    t_map = get_global_trending_map()
    if not t_map: return menu_df.sample(min(8, len(menu_df)))['food_name'].tolist()
    return sorted(t_map.keys(), key=lambda x: t_map[x], reverse=True)[:12]

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPLEMENTARY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
PAIRS = {
    "dosa":          [("filter coffee", 3.0), ("masala chai", 3.0), ("lassi", 1.5)],
    "idli":          [("filter coffee", 3.0), ("masala chai", 3.0), ("vada", 2.0)],
    "vada":          [("filter coffee", 3.0), ("masala chai", 2.5), ("idli", 2.0)],
    "pongal":        [("filter coffee", 3.0), ("masala chai", 2.5)],
    "uttapam":       [("filter coffee", 2.5), ("masala chai", 2.5)],
    "rava kesari":   [("filter coffee", 2.5), ("masala chai", 2.0)],
    "biryani":       [("lassi", 3.0), ("fruit juice", 2.0), ("samosa", 1.5), ("pani puri", 1.0)],
    "paneer tikka":  [("naan", 3.0), ("lassi", 2.0), ("masala chai", 1.5)],
    "dal makhani":   [("naan", 3.0), ("lassi", 2.0)],
    "naan":          [("paneer tikka", 2.5), ("dal makhani", 2.5)],
    "chole bhature": [("lassi", 2.5), ("masala chai", 2.0)],
    "paratha":       [("lassi", 2.5), ("masala chai", 2.0)],
    "butter chicken":[("naan", 3.0), ("lassi", 2.0)],
    "burger":        [("french fries", 3.0), ("cold coffee", 2.0), ("lassi", 1.5)],
    "pizza":         [("cold coffee", 2.5), ("fruit juice", 2.0)],
    "pasta":         [("cold coffee", 2.0), ("fruit juice", 1.5)],
    "sandwich":      [("cold coffee", 2.0), ("fruit juice", 1.5)],
    "french fries":  [("burger", 2.5), ("cold coffee", 2.0)],
    "momos":         [("cold coffee", 2.0), ("fruit juice", 1.5)],
    "samosa":        [("masala chai", 3.0), ("filter coffee", 2.5)],
    "kachori":       [("masala chai", 3.0), ("filter coffee", 2.5)],
    "bhel puri":     [("masala chai", 2.0), ("fruit juice", 1.5)],
    "pani puri":     [("masala chai", 2.0), ("fruit juice", 1.5)],
    "vada pav":      [("masala chai", 3.0), ("filter coffee", 2.5)],
    "pakoda":        [("masala chai", 3.0), ("filter coffee", 2.5)],
    "filter coffee": [("dosa", 3.0), ("idli", 2.5), ("vada", 2.0), ("samosa", 2.0)],
    "masala chai":   [("samosa", 3.0), ("pakoda", 3.0), ("vada", 2.5), ("idli", 2.0)],
    "cold coffee":   [("burger", 2.0), ("french fries", 2.0), ("pizza", 1.5)],
    "lassi":         [("biryani", 2.5), ("paneer tikka", 2.0), ("naan", 1.5)],
    "badam milk":    [("samosa", 2.0), ("pakoda", 2.0)],
    "fruit juice":   [("burger", 1.5), ("pizza", 1.5), ("momos", 1.5)],
}

def get_dish_image(food_name, category):
    name = food_name.lower()
    IMAGE_MAP = {
        "dosa":          "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
        "idli":          "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=600&auto=format&fit=crop&q=60",
        "vada":          "https://images.unsplash.com/photo-1630383249896-424e482df921?w=600&auto=format&fit=crop&q=60",
        "pongal":        "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop&q=60",
        "uttapam":       "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
        "rava kesari":   "https://images.unsplash.com/photo-1606913084603-3b5c8b3b2b56?w=600&auto=format&fit=crop&q=60",
        "bhat":          "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&auto=format&fit=crop&q=60",
        "biryani":       "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop&q=60",
        "hyderabadi":    "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop&q=60",
        "ambur":         "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60",
        "lucknowi":      "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop&q=60",
        "donne":         "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60",
        "paneer tikka":  "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=600&auto=format&fit=crop&q=60",
        "butter chicken":"https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=600&auto=format&fit=crop&q=60",
        "dal makhani":   "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=600&auto=format&fit=crop&q=60",
        "naan":          "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60",
        "chole bhature": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&auto=format&fit=crop&q=60",
        "paratha":       "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60",
        "burger":        "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop&q=60",
        "pizza":         "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop&q=60",
        "pasta":         "https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=600&auto=format&fit=crop&q=60",
        "sandwich":      "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=600&auto=format&fit=crop&q=60",
        "french fries":  "https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=600&auto=format&fit=crop&q=60",
        "momos":         "https://images.unsplash.com/photo-1534422298391-e4f8c172dddb?w=600&auto=format&fit=crop&q=60",
        "samosa":        "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop&q=60",
        "kachori":       "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop&q=60",
        "bhel puri":     "https://images.unsplash.com/photo-1567337710282-00832b415979?w=600&auto=format&fit=crop&q=60",
        "pani puri":     "https://images.unsplash.com/photo-1567337710282-00832b415979?w=600&auto=format&fit=crop&q=60",
        "vada pav":      "https://images.unsplash.com/photo-1567337710282-00832b415979?w=600&auto=format&fit=crop&q=60",
        "pakoda":        "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop&q=60",
        "chat basket":   "https://images.unsplash.com/photo-1567337710282-00832b415979?w=600&auto=format&fit=crop&q=60",
        "filter coffee": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=60",
        "masala chai":   "https://images.unsplash.com/photo-1556679343-c7306c1976bc?w=600&auto=format&fit=crop&q=60",
        "cold coffee":   "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?w=600&auto=format&fit=crop&q=60",
        "lassi":         "https://media.istockphoto.com/id/2214375556/photo/lassi-in-clay-cup-topped-with-dry-fruits.webp?a=1&b=1&s=612x612&w=0&k=20&c=OsxAg9ZKpsQI-lc4bcmujNQb6kQCPVUun4e_aHeEUMY=",
        "badam milk":    "https://images.unsplash.com/photo-1550583724-b2692b85b150?w=600&auto=format&fit=crop&q=60",
        "fruit juice":   "https://images.unsplash.com/photo-1600271886742-f049cd451bba?w=600&auto=format&fit=crop&q=60",
    }
    for keyword, url in IMAGE_MAP.items():
        if keyword in name:
            return url
    CAT_IMAGES = {
        "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
        "North Indian": "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=600&auto=format&fit=crop&q=60",
        "Biryani":      "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop&q=60",
        "Beverage":     "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop&q=60",
        "Fastfood":     "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop&q=60",
        "Snack":        "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop&q=60",
    }
    return CAT_IMAGES.get(category, "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&auto=format&fit=crop&q=60")

CAT_FALLBACK = {
    "South Indian": ["Beverage", "Snack"],
    "Biryani":      ["Beverage", "Snack"],
    "North Indian": ["Beverage"],
    "Fastfood":     ["Beverage", "Snack"],
    "Snack":        ["Beverage"],
    "Beverage":     ["Snack", "South Indian"],
}

def get_complementary_items(cart_items: list, top_n: int = 3) -> list:
    if not cart_items:
        return []
    scores    = defaultdict(float)
    cat_lookup = dict(zip(menu_df['food_name'], menu_df['category']))
    for cart_item in cart_items:
        cart_lower = cart_item.lower()
        matched    = False
        for keyword, targets in PAIRS.items():
            if keyword in cart_lower:
                matched = True
                for target_kw, score in targets:
                    for name in all_items:
                        if target_kw in name.lower() and name not in cart_items:
                            scores[name] += score
        if not matched:
            item_cat  = cat_lookup.get(cart_item, "")
            cart_cats = {cat_lookup.get(i, "") for i in cart_items}
            for paired_cat in CAT_FALLBACK.get(item_cat, []):
                if paired_cat not in cart_cats:
                    for name in all_items:
                        if cat_lookup.get(name) == paired_cat and name not in cart_items:
                            scores[name] += 0.5
    if 'assoc_rules' in ai:
        for cart_item in cart_items:
            for consequent, confidence, lift in ai['assoc_rules'].get(cart_item, []):
                if consequent in scores:
                    scores[consequent] += min(confidence * lift, 2.0)
    ranked = sorted(
        [(item, s) for item, s in scores.items() if item in set(all_items) and s > 0],
        key=lambda x: x[1], reverse=True
    )
    return [item for item, _ in ranked[:top_n]]

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
user_cats = get_user_order_categories(st.session_state.user_id)

with st.sidebar:
    st.markdown(f"<h2 style='color:#E50914;margin-top:0;'><i class='fas fa-user-circle'></i> {st.session_state.username}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1rem;font-weight:700;color:#E50914;'><i class='fas fa-dna'></i> Profile DNA</p>", unsafe_allow_html=True)
    with Session() as db:
        dna_rows = db.query(Order.category, func.count(Order.id)).filter_by(
            user_id=st.session_state.user_id).group_by(Order.category).all()
        if dna_rows:
            dna_df = pd.DataFrame(dna_rows, columns=['Category', 'Orders']).set_index('Category')
            st.bar_chart(dna_df, color="#E50914")
        else:
            st.info("Start ordering to build your flavor profile!")
        st.markdown("<p style='font-size:1rem;font-weight:700;color:#E5E5E5;'><i class='fas fa-history'></i> Recent Activity</p>", unsafe_allow_html=True)
        history = db.query(Order).filter_by(
            user_id=st.session_state.user_id
        ).order_by(Order.timestamp.desc()).limit(5).all()
        for h in history:
            img_url = get_dish_image(h.food_name, h.category)
            col_img, col_info, col_btn = st.columns([1, 3, 1.2])
            with col_img:
                st.markdown(f"""<img src="{img_url}"
                    style="width:42px;height:42px;border-radius:6px;object-fit:cover;margin-top:4px;">
                """, unsafe_allow_html=True)
            with col_info:
                ist_time = h.timestamp.replace(tzinfo=datetime.timezone.utc).astimezone(IST).strftime('%d %b, %I:%M %p')
                st.markdown(f"""
                <div style='font-size:0.85rem;font-weight:600;color:#eee;margin-top:4px;'>{h.food_name}</div>
                <div style='color:#777;font-size:0.7rem;'>₹{int(h.price)} · {ist_time}</div>
                """, unsafe_allow_html=True)
            with col_btn:
                if st.button("↺", key=f"reorder_{h.id}", help="Reorder this item"):
                    row = menu_df[menu_df['food_name'] == h.food_name]
                    if not row.empty:
                        info = row.iloc[0]
                        if h.food_name in st.session_state.cart:
                            st.session_state.cart[h.food_name]['qty'] += 1
                        else:
                            st.session_state.cart[h.food_name] = {
                                'price':    float(info['price']),
                                'qty':      1,
                                'category': str(info['category']),
                            }
                        st.toast(f"{h.food_name} added to cart!")
                        st.rerun()
            st.markdown("<hr style='border:none;border-top:1px solid #2a2a2a;margin:4px 0;'>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='font-size:1rem;font-weight:700;color:#E5E5E5;'><i class='fas fa-shopping-cart'></i> Your Order</p>", unsafe_allow_html=True)
    cart = st.session_state.cart
    if not cart:
        st.caption("Cart is empty — add items below!")
    else:
        total = 0
        for name, info in list(cart.items()):
            sub   = info['price'] * info['qty']
            total += sub
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{name}** ×{info['qty']}")
            if c2.button("x", key=f"rm_{name}"):
                if info['qty'] > 1:
                    st.session_state.cart[name]['qty'] -= 1
                else:
                    del st.session_state.cart[name]
                st.rerun()
            st.caption(f"₹{int(sub)}")
        st.markdown(f"### Total: ₹{int(total)}")
        recs = get_complementary_items(list(cart.keys()))
        if recs:
            st.markdown("---")
            st.markdown("<p style='font-size:0.9rem;font-weight:600;color:#aaa;margin:8px 0 4px 0;'><i class='fas fa-utensils'></i> Pairs well with</p>", unsafe_allow_html=True)
            for rec in recs:
                row = menu_df[menu_df['food_name'] == rec]
                if row.empty: continue
                info = row.iloc[0]
                rc1, rc2 = st.columns([3, 1])
                rc1.write(f"{rec} (₹{int(info['price'])})")
                if rc2.button("Add", key=f"rec_{rec}"):
                    st.session_state.cart[rec] = {
                        'price': float(info['price']),
                        'qty':   1,
                        'category': str(info['category']),
                    }
                    st.rerun()
        if st.button("Place Order", use_container_width=True, type="primary"):
            total_wait = 0
            with Session() as db:
                for name, info in cart.items():
                    row = menu_df[menu_df['food_name'] == name]
                    img = row.iloc[0]['image'] if not row.empty else ""
                    w   = predict_wait_time(name, info['category'])
                    total_wait = max(total_wait, w)
                    for _ in range(info['qty']):
                        db.add(Order(
                            user_id=st.session_state.user_id,
                            food_name=name, category=info['category'],
                            price=info['price'], image=img,
                        ))
                        db.add(KitchenQueue(
                            food_name=name, category=info['category'],
                            status='pending'
                        ))
                db.commit()
            st.session_state.cart = {}
            st.cache_data.clear()
            st.success(f"Order placed! Ready in ~{total_wait} min")
            st.rerun()
    if st.button("Sign Out", use_container_width=True):
        st.session_state.user_id  = None
        st.session_state.username = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 8. ROW RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_row(title, foods, icon):
    if not foods: return
    st.markdown(f"<div class='section-title'><i class='{icon}'></i> {title}</div>", unsafe_allow_html=True)
    display_items = foods[:12]
    for i in range(0, len(display_items), 4):
        cols  = st.columns(4)
        chunk = display_items[i:i+4]
        for idx, name in enumerate(chunk):
            item_rows = menu_df[menu_df['food_name'] == name]
            if item_rows.empty: continue
            item      = item_rows.iloc[0]
            match     = get_match_score(name, user_cats)
            img_url   = get_dish_image(name, item['category'])
            rating    = item.get('rating', 4.0)
            stars     = '★' * int(rating) + '☆' * (5 - int(rating))
            wait_mins = predict_wait_time(name, item['category'])
            wt_color, wt_label = wait_color(wait_mins)
            with cols[idx]:
                st.markdown(f"""
                <div class="food-card">
                    <div class="card-img-container">
                        <img src="{img_url}" class="card-img" alt="{name}"
                             onerror="this.onerror=null;this.src='https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=600&q=80';">
                    </div>
                    <div class="match-tag">{match}% Match</div>
                    <div class="card-content">
                        <div class="food-title">{name}</div>
                        <div style="color:#f5a623;font-size:0.8rem;margin:2px 0;">{stars} <span style="color:#999;font-size:0.75rem;">{rating}</span></div>
                        <div style="color:#777;font-size:0.75rem;margin:2px 0;">
                            {item['category']} • ₹{item['price']}
                        </div>
                        <div style="font-size:0.75rem;margin-bottom:6px;">
                            <span style="color:{wt_color};font-weight:700;">
                                <i class="fas fa-clock"></i> {wait_mins} min
                            </span>
                            <span style="color:{wt_color};font-size:0.7rem;"> {wt_label}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                if st.button("Add to Cart", key=f"add_{title}_{i}_{idx}", use_container_width=True):
                    if name in st.session_state.cart:
                        st.session_state.cart[name]['qty'] += 1
                    else:
                        st.session_state.cart[name] = {
                            'price':    float(item['price']),
                            'qty':      1,
                            'category': str(item['category']),
                        }
                    st.toast(f"{name} added to cart!")
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='margin-bottom:20px;'>
        <h1 style='color:#E50914;font-size:4rem;font-weight:900;margin:0;line-height:1;'>NuCafé</h1>
        <p style='color:#eee;font-size:1.1rem;margin-top:10px;'>Personalized Gourmet Delivery in Bangalore</p>
    </div>
""", unsafe_allow_html=True)

top_picks = sorted(list(set(all_items)), key=lambda x: get_match_score(x, user_cats), reverse=True)
render_row("Top Picks for You", top_picks, "fas fa-magic")
render_row("Trending Now", get_trending_items(), "fas fa-fire")

bangalore_specials = ["CTR Benne Masala Dosa", "Filter Coffee", "MTR Rava Idli",
                      "VV Puram Chat Basket", "Meghana Chicken Biryani"]
available_specials = [x for x in bangalore_specials if x in all_items]
render_row("Bangalore Classics", available_specials, "fas fa-map-marker-alt")

# ── Time Filter Section ────────────────────────────────────────────────────
st.markdown("<div class='section-title'><i class='fas fa-clock'></i> Order by Available Time</div>", unsafe_allow_html=True)
time_col1, time_col2 = st.columns([1, 3])
with time_col1:
    break_time = st.slider("My break is (minutes)", min_value=10, max_value=60, value=30, step=5)
with time_col2:
    queue_now = get_queue_length()
    st.markdown(f"""
    <div style='background:#1f1f1f;border-radius:8px;padding:12px 16px;border-left:4px solid #F5A623;margin-top:8px;'>
        <span style='color:#F5A623;font-weight:700;font-size:0.95rem;'>
            <i class='fas fa-users'></i> {queue_now} orders in kitchen right now
        </span>
        <span style='color:#888;font-size:0.8rem;margin-left:12px;'>
            Showing items you can get in {break_time} min
        </span>
    </div>""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_all_wait_times():
    result = {}
    for _, row in menu_df.iterrows():
        name = row['food_name']
        cat  = row['category']
        CATEGORY_BASE = {
            "South Indian":10,"North Indian":20,"Biryani":25,
            "Beverage":5,"Fastfood":12,"Snack":8
        }
        bp   = CATEGORY_BASE.get(cat, int(row['prep_time']))
        now  = now_ist()
        hour = now.hour
        rush = 1.6 if 12<=hour<=14 else (1.5 if 19<=hour<=21 else (1.1 if 9<=hour<=11 else 1.0))
        q    = get_queue_length()
        est  = max(3, int(round(bp * rush + q * 1.2)))
        result[name] = est
    return result

wait_map = get_all_wait_times()

time_filtered = [(name, wait_map.get(name, 15)) for name in all_items
                 if wait_map.get(name, 15) <= break_time]
time_filtered.sort(key=lambda x: x[1])

if time_filtered:
    time_names = [name for name, _ in time_filtered[:12]]
    render_row(f"Ready within {break_time} min", time_names, "fas fa-bolt")
else:
    st.info("No items fit your time. Try a longer break duration.")

st.markdown("<div class='section-title'><i class='fas fa-search'></i> Explore More</div>", unsafe_allow_html=True)
cats         = sorted(menu_df['category'].unique().tolist())
selected_cat = st.selectbox("Select a Genre", cats, label_visibility="collapsed")
cat_items    = menu_df[menu_df['category'] == selected_cat]['food_name'].tolist()
render_row(f"Popular in {selected_cat}", cat_items, "fas fa-utensils")