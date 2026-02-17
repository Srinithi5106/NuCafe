import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from sqlalchemy import func
from database import Session, User, Order, hash_password, check_password

# --- 1. CONFIG & SESSION PERSISTENCE ---
st.set_page_config(layout="wide", page_title="NuCafé Netflix", initial_sidebar_state="expanded")

if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

@st.cache_resource
def load_engine():
    if not os.path.exists('simple_data.pkl') or not os.path.exists('menu.json'):
        st.error("Please run train_ai.py first!")
        st.stop()
    
    with open('menu.json', 'r') as f:
        menu_data = json.load(f)
    menu = pd.DataFrame(menu_data)
    
    # Pre-calculate norms for scoring
    if not menu.empty:
        menu['prep_norm'] = (menu['prep_time'] - menu['prep_time'].min()) / (menu['prep_time'].max() - menu['prep_time'].min() + 1e-5)
        menu['price_norm'] = (menu['price'] - menu['price'].min()) / (menu['price'].max() - menu['price'].min() + 1e-5)

    with open('simple_data.pkl', 'rb') as f:
        brain = pickle.load(f)
    return menu, brain

menu_df, ai = load_engine()

# --- 2. AUTHENTICATION ---
if st.session_state.user_id is None:
    st.markdown("""
        <div style='text-align:center; padding: 50px 0;'>
            <h1 style='color:#E50914; font-size:5rem; margin-bottom:0;'>NuCafé</h1>
            <p style='color:#aaa; font-size:1.2rem;'>Unlimited Cravings. One Subscription.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1.5,1])
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
                            st.session_state.user_id = user.id
                            st.session_state.username = u
                            st.rerun()
                        else: st.error("Invalid credentials")
            
            with mode[1]:
                u_new = st.text_input("Choose Username")
                p_new = st.text_input("Create Password", type="password")
                if st.button("Start Membership", use_container_width=True):
                    with Session() as db:
                        if db.query(User).filter_by(username=u_new).first(): st.error("User exists!")
                        else:
                            new_user = User(username=u_new, password=hash_password(p_new))
                            db.add(new_user); db.commit()
                            st.success("Welcome to the club! Now Sign In.")
    st.stop()

# --- 3. LOGIC ENGINE ---

@st.cache_data(ttl=60)
def get_user_order_categories(user_id):
    with Session() as db:
        orders = db.query(Order.category).filter_by(user_id=user_id).all()
        return [o[0] for o in orders]

@st.cache_data(ttl=300)
def get_global_trending_map():
    with Session() as db:
        recent_orders = db.query(Order.food_name).order_by(Order.timestamp.desc()).limit(100).all()
        if not recent_orders: return {}
        counts = pd.Series([o[0] for o in recent_orders]).value_counts()
        return (counts / counts.max()).to_dict()

def get_match_score(food_name, user_categories):
    item_rows = menu_df[menu_df['food_name'] == food_name]
    if item_rows.empty: return 50
    item = item_rows.iloc[0]

    # Matrix Factorization Score
    u_idx = st.session_state.user_id % 1000
    f_idx = ai['food_to_idx'].get(food_name, 0)
    collab = (np.dot(ai['P'][u_idx], ai['Q'][f_idx]) / 5.0)
    
    # Content-Based Score
    category_match = user_categories.count(item['category']) / len(user_categories) if user_categories else 0.5
    prep_score = 1.0 - item.get('prep_norm', 0.5)
    content = (0.7 * category_match) + (0.3 * prep_score)
    
    # Trending & Recency
    trend = get_global_trending_map().get(food_name, 0.0)
    new_bonus = 0.15 if item.get('is_new') == 'yes' else 0.0
    
    final_score = (0.5 * collab) + (0.3 * content) + (0.2 * trend) + new_bonus
    return min(max(int(final_score * 100), 12), 99)

def get_trending_items():
    t_map = get_global_trending_map()
    if not t_map: return menu_df.sample(min(8, len(menu_df)))['food_name'].tolist()
    return sorted(t_map.keys(), key=lambda x: t_map[x], reverse=True)[:12]

# --- 4. CSS & STYLING ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .stApp { background-color: #141414; color: white; }
    
    /* Netflix Section Title */
    .section-title { 
        font-size: 1.6rem; 
        font-weight: 700; 
        color: #E5E5E5; 
        margin: 40px 0 15px 0; 
        display: flex; 
        align-items: center; 
        gap: 12px; 
    }
    
    /* Food Card - Glassmorphism style */
    .food-card { 
        background: #1f1f1f; 
        border-radius: 8px; 
        overflow: hidden;
        transition: transform 0.4s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.4s; 
        padding-bottom: 12px;
        height: 100%;
        border: 1px solid #333;
    }
    
    .food-card:hover { 
        transform: scale(1.08); 
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border-color: #E50914;
        z-index: 10;
    }
    
    .card-img-container {
        width: 100%;
        height: 160px;
        background: linear-gradient(45deg, #222, #333);
        position: relative;
        overflow: hidden;
    }
    
    .card-img {
        width: 100%; 
        height: 100%; 
        object-fit: cover;
        display: block;
        background-color: #222;
    }

    .match-tag { 
        color: #46D369; 
        font-weight: 800; 
        font-size: 0.9rem; 
        padding: 10px 12px 2px 12px;
    }
    
    .card-content {
        padding: 0 12px;
    }
    
    .food-title {
        margin: 4px 0; 
        font-size: 1.05rem; 
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Override Streamlit Buttons for Netflix Look */
    .stButton > button {
        background-color: transparent !important;
        border: 1px solid #666 !important;
        color: white !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important;
        transition: 0.2s !important;
    }
    .stButton > button:hover {
        background-color: white !important;
        color: black !important;
        border-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

user_cats = get_user_order_categories(st.session_state.user_id)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown(f"<h2 style='color:#E50914; margin-top:0;'><i class='fas fa-user-circle'></i> {st.session_state.username}</h2>", unsafe_allow_html=True)
    
    st.markdown("### Profile DNA")
    with Session() as db:
        dna_query = db.query(Order.category, func.count(Order.id)).filter_by(user_id=st.session_state.user_id).group_by(Order.category).all()
        if dna_query:
            dna_df = pd.DataFrame(dna_query, columns=['Category', 'Orders']).set_index('Category')
            st.bar_chart(dna_df, color="#E50914")
        else:
            st.info("Start ordering to build your flavor profile!")
            
        st.markdown("### Recent Activity")
        history = db.query(Order).filter_by(user_id=st.session_state.user_id).order_by(Order.timestamp.desc()).limit(3).all()
        for h in history:
            # Using the display_image logic from the database.py models
            st.markdown(f"""
                <div style='background:#222; padding:8px; border-radius:4px; margin-bottom:8px; border-left:3px solid #E50914; display:flex; align-items:center; gap:10px;'>
                    <img src="{h.display_image}" style="width:40px; height:40px; border-radius:4px; object-fit:cover;">
                    <div>
                        <div style='font-size:0.85rem;'>{h.food_name}</div>
                        <div style='color:#777; font-size:0.7rem;'>{h.timestamp.strftime('%d %b, %H:%M')}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    if st.button("Sign Out", use_container_width=True): 
        st.session_state.user_id = None
        st.rerun()

# --- 6. ROW RENDERER ---
def render_row(title, foods, icon):
    if not foods: return
    st.markdown(f"<div class='section-title'><i class='{icon}'></i> {title}</div>", unsafe_allow_html=True)
    
    display_items = foods[:12]
    num_items = len(display_items)
    
    for i in range(0, num_items, 4):
        cols = st.columns(4)
        chunk = display_items[i:i+4]
        
        for idx, name in enumerate(chunk):
            item_rows = menu_df[menu_df['food_name'] == name]
            if item_rows.empty: continue
            item = item_rows.iloc[0]
            match = get_match_score(name, user_cats)
            
            # Create a temporary Order instance to leverage the display_image property 
            # defined in the database Canvas file
            temp_order = Order(food_name=name, category=item['category'], image=item.get('image'))
            img_url = temp_order.display_image
            
            with cols[idx]:
                st.markdown(f"""
                    <div class="food-card">
                        <div class="card-img-container">
                            <img src="{img_url}" 
                                 class="card-img" 
                                 alt="{name}"
                                 onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=600&q=80';">
                        </div>
                        <div class="match-tag">{match}% Match</div>
                        <div class="card-content">
                            <div class="food-title">{name}</div>
                            <div style="color:#777; font-size:0.75rem; margin-bottom:8px;">
                                {item['category']} • ₹{item['price']} • {item['prep_time']}m
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("Add to List", key=f"order_{title}_{i}_{idx}", use_container_width=True):
                    with Session() as db:
                        new_order = Order(
                            user_id=st.session_state.user_id, 
                            food_name=name, 
                            category=item['category'], 
                            price=item['price'],
                            image=item.get('image')
                        )
                        db.add(new_order); db.commit()
                    st.toast(f"✅ Added {name} to your history!")
                    st.rerun()

# --- 7. MAIN INTERFACE ---
st.markdown("""
    <div style='margin-bottom: 20px;'>
        <h1 style='color:#E50914; font-size:4rem; font-weight:900; margin:0; line-height:1;'>NuCafé</h1>
        <p style='color:#eee; font-size:1.1rem; margin-top:10px;'>Personalized Gourmet Delivery in Bangalore</p>
    </div>
""", unsafe_allow_html=True)

all_items = menu_df['food_name'].tolist()
top_picks = sorted(list(set(all_items)), key=lambda x: get_match_score(x, user_cats), reverse=True)
render_row("Top Picks for You", top_picks, "fas fa-magic")
render_row("Trending Now", get_trending_items(), "fas fa-fire")

bangalore_specials = ["CTR Benne Masala Dosa", "Filter Coffee", "MTR Rava Idli", "VV Puram Chat Basket", "Meghana Chicken Biryani"]
available_specials = [x for x in bangalore_specials if x in all_items]
render_row("Bangalore Classics", available_specials, "fas fa-map-marker-alt")

st.markdown("<div class='section-title'><i class='fas fa-search'></i> Explore More</div>", unsafe_allow_html=True)
cats = sorted(menu_df['category'].unique().tolist())
selected_cat = st.selectbox("Select a Genre", cats, label_visibility="collapsed")
cat_items = menu_df[menu_df['category'] == selected_cat]['food_name'].tolist()
render_row(f"Popular in {selected_cat}", cat_items, "fas fa-utensils")