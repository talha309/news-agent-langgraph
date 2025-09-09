from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os 

# Load env vars
load_dotenv()

# Fallback to local SQLite if DATABASE_URL is not defined
URL_DATABASE = os.getenv('DATABASE_URL') or 'sqlite:///./app.db'

# Only pass SQLite-specific connect args when using SQLite
connect_args = {"check_same_thread": False} if URL_DATABASE.startswith("sqlite") else {}

engine = create_engine(URL_DATABASE, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()