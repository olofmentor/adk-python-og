from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .agent import AnalysisAgent, AgentConfig, DEFAULT_DATASET_PATH
from .db import AnalysisResult, get_session, init_db
from .schemas import AnalyzeRequest, AnalyzeResponse, ResultRow

app = FastAPI(title="Stat Analysis Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_agent() -> AnalysisAgent:
    dataset_path = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)
    return AnalysisAgent(AgentConfig(dataset_path=dataset_path))


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, session: AsyncSession = Depends(get_session)) -> AnalyzeResponse:
    agent = _get_agent()
    try:
        result = agent.run(body.prompt)
        summary = result.get("type")
        details_json = json.dumps(result)
        row = AnalysisResult(prompt=body.prompt, status="success", summary=summary, details_json=details_json)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return AnalyzeResponse(result_id=row.id)
    except Exception as exc:  # noqa: BLE001
        row = AnalysisResult(prompt=body.prompt, status="error", summary=str(exc), details_json=None)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/results/{result_id}", response_model=ResultRow)
async def get_result(result_id: int, session: AsyncSession = Depends(get_session)) -> ResultRow:
    row = await session.get(AnalysisResult, result_id)
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    return ResultRow.model_validate(row)


@app.get("/results", response_model=List[ResultRow])
async def list_results(session: AsyncSession = Depends(get_session)) -> List[ResultRow]:
    rows = (await session.execute(select(AnalysisResult).order_by(AnalysisResult.id.desc()).limit(100))).scalars().all()
    return [ResultRow.model_validate(r) for r in rows]