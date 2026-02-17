import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# --- DATABASE SETUP ---
DATABASE_URL = "sqlite:///nucafe.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()

# --- CONSTANTS ---
# Standard high-quality placeholder for any missing assets
DEFAULT_PLACEHOLDER = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=600&q=80"

# --- MODELS ---

class User(Base):
    """User model for authentication and personalization."""
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<User(username='{self.username}')>"

class Order(Base):
    """
    Enhanced Order history model with integrated fallback image logic.
    Identical to the mapping used in the training engine for consistency.
    """
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    food_name = Column(String, nullable=False)
    category = Column(String, nullable=False) 
    price = Column(Float, nullable=False)
    quantity = Column(Integer, default=1)
    prep_time = Column(Integer, nullable=True) 
    image = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    @property
    def total_price(self):
        return self.price * self.quantity

    @property
    def display_image(self):
        """
        Logic sequence: 
        1. Specific item image (if exists)
        2. Category-specific Unsplash URL (Matched to your custom trainer URLs)
        3. Global Default Placeholder (to prevent broken UI)
        """
        if self.image and self.image.strip():
            return self.image

        # This map reflects the high-quality URLs you provided for the NuCafe brand
        category_map = {
            "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop&q=60",
            "North Indian": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600&auto=format&fit=crop&q=60",
            "Biryani": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop&q=60",
            "Beverage": "https://images.unsplash.com/photo-1609951651556-5334e2706168?w=600&auto=format&fit=crop&q=60",
            "Fastfood": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop&q=60",
            "Snack": "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=600&auto=format&fit=crop&q=60"
        }
        
        return category_map.get(self.category, DEFAULT_PLACEHOLDER)

    def __repr__(self):
        return f"<Order(user_id={self.user_id}, food='{self.food_name}', qty={self.quantity})>"

# Create tables if they don't exist
Base.metadata.create_all(engine)

# --- UTILS ---

def hash_password(password):
    """Hash a password for secure storage using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    """Check a password against a stored hash."""
    return hash_password(password) == hashed

def get_db_session():
    """Context-safe helper to get a database session."""
    return Session()