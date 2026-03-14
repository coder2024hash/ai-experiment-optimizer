from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./models.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModelMeta(Base):
    """Model metadata table"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    file_name = Column(String)
    model_path = Column(String)
    target_column = Column(String, nullable=True)
    problem_type = Column(String, nullable=True)
    metrics = Column(Text, nullable=True)
    feature_columns = Column(Text, nullable=True)
    feature_importance = Column(Text, nullable=True)
    n_estimators = Column(Integer, nullable=True)
    max_depth = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
