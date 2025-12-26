from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON
from datetime import datetime

Base = declarative_base()

class History(Base):
    __tablename__ = 'history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    request = Column(String)
    response = Column(String)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Request(Base):
    __tablename__ = 'request'

    id = Column(Integer, primary_key=True, index=True)
    size = Column(Integer, index=True)
