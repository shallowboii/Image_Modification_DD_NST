# backend/app/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import datetime

_HERE = Path(__file__).parent
SQLITE_URL = f"sqlite:///{_HERE / 'database.db'}"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="pending")   # pending, running, done, failed
    preset = Column(String, nullable=True)
    params = Column(Text, nullable=True)         # JSON string of params
    input_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)