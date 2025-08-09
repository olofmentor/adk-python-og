from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

import os
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///app/data/results.db")

Base = declarative_base()


class AnalysisResult(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="success")
    summary = Column(Text, nullable=True)
    details_json = Column(Text, nullable=True)


async_engine: AsyncEngine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session