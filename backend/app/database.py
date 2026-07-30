"""
database.py
============
SQLite database (via SQLAlchemy) that stores every prediction MediPredict
makes, so a user can review their prediction history per disease over time.
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "medipredict.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    disease = Column(String(50), index=True, nullable=False)
    patient_label = Column(String(100), default="Unnamed patient")
    input_json = Column(Text, nullable=False)     # raw form inputs
    probability = Column(Float, nullable=False)   # model's predicted probability of disease
    prediction = Column(Integer, nullable=False)  # 0 = low risk, 1 = high risk
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
