import hashlib
import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_0lSMa2wbyHVF@ep-icy-bird-a1ix5uqr-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require")

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require", "connect_timeout": 10},
    pool_pre_ping=True,
    pool_recycle=300,
)
Session = sessionmaker(bind=engine)
Base = declarative_base()

DEFAULT_PLACEHOLDER = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=600&q=80"

class User(Base):
    __tablename__ = 'users'
    id         = Column(Integer, primary_key=True)
    username   = Column(String, unique=True, nullable=False)
    password   = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Order(Base):
    __tablename__ = 'orders'
    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey('users.id'))
    food_name = Column(String, nullable=False)
    category  = Column(String, nullable=False)
    price     = Column(Float, nullable=False)
    quantity  = Column(Integer, default=1)
    prep_time = Column(Integer, nullable=True)
    image     = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    @property
    def display_image(self):
        if self.image and self.image.strip():
            return self.image
        category_map = {
            "South Indian": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600",
            "North Indian": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=600",
            "Biryani":      "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600",
            "Beverage":     "https://images.unsplash.com/photo-1609951651556-5334e2706168?w=600",
            "Fastfood":     "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600",
            "Snack":        "https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=600",
        }
        return category_map.get(self.category, DEFAULT_PLACEHOLDER)

Base.metadata.create_all(engine)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    return hash_password(password) == hashed