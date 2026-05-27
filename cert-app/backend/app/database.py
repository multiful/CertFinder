"""Database connection and session management."""
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=2,       # Render 무료 플랜: 유휴 커넥션 최소화
    max_overflow=3,    # 순간 피크 허용치 (총 최대 5개)
    pool_timeout=30,
    pool_recycle=1800, # 30분마다 커넥션 재생성 (Supabase idle 끊김 방지)
    echo=settings.DEBUG,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """Check if database connection is healthy."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# Event listeners for connection debugging (only in DEBUG mode)
if settings.DEBUG:
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Database connection checked out from pool")
