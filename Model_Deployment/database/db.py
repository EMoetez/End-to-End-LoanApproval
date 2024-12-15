# database/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Define the log table
class PredictionLog(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# SQLite database connection
DATABASE_URL = "sqlite:///./logs.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize the database
Base.metadata.create_all(bind=engine)
