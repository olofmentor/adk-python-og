from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., description="Natural language analysis instruction")


class AnalyzeResponse(BaseModel):
    result_id: int


class ResultRow(BaseModel):
    id: int
    created_at: datetime
    prompt: str
    status: str
    summary: Optional[str]
    details_json: Optional[str]

    class Config:
        from_attributes = True