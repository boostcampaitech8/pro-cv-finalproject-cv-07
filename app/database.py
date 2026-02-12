import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import urllib

load_dotenv()

password = urllib.parse.quote(os.getenv('DB_PASS'))
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{password}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"ssl": {}}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Report DB
report_password = urllib.parse.quote(os.getenv('REPORT_DB_PASS', ''))
REPORT_DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('REPORT_DB_USER')}:{report_password}"
    f"@{os.getenv('REPORT_DB_HOST')}:{os.getenv('REPORT_DB_PORT', '3306')}/{os.getenv('REPORT_DB_NAME')}"
)

report_engine = create_engine(
    REPORT_DATABASE_URL,
    connect_args={"ssl": {}, "connect_timeout": 5},
    pool_pre_ping=True
)
ReportSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=report_engine)
ReportBase = declarative_base()

def get_report_db():
    db = ReportSessionLocal()
    try:
        yield db
    finally:
        db.close()