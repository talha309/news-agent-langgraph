from database import Base
from sqlalchemy import Column, Integer, String, Boolean, Float

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key = True, index = True)
    query = Column(String)