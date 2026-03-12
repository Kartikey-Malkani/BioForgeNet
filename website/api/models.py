from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from pathlib import Path

def _resolve_database_url() -> str:
    explicit_url = os.getenv("DEMO_DB_URL", "").strip()
    if explicit_url:
        return explicit_url

    explicit_path = os.getenv("DEMO_DB_PATH", "").strip()
    if explicit_path:
        return f"sqlite:///{Path(explicit_path)}"

    candidates = [
        Path("/tmp/bioforgenet_data"),
        Path(__file__).resolve().parents[2] / "data",
        Path.cwd() / "data",
    ]

    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{directory / 'demo_requests.db'}"
        except Exception:
            continue

    return "sqlite:///:memory:"


DATABASE_URL = _resolve_database_url()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DemoRequest(Base):
    __tablename__ = "demo_requests"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    company_size = Column(String, nullable=False)
    use_case = Column(Text, nullable=True)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    email_sent = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "company_name": self.company_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "industry": self.industry,
            "company_size": self.company_size,
            "use_case": self.use_case,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "email_sent": self.email_sent,
        }


# Create tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:
    print(f"⚠️ Demo DB table creation skipped: {exc}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
