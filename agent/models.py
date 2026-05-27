"""Pydantic 데이터 모델 정의: Action, Observation, AgentState 및 LangGraph용 GraphState."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────── Enums ────────────────────────────


class ActionType(str, Enum):
    """Policy가 반환할 수 있는 Action 유형."""

    CALL_TOOL = "CALL_TOOL"
    WRITE_NOTE = "WRITE_NOTE"
    SYNTHESIZE = "SYNTHESIZE"
    STOP = "STOP"


class ToolName(str, Enum):
    """사용 가능한 Tool 이름."""

    REGULATION_SEARCH = "regulation_search"


class ObservationKind(str, Enum):
    """Observation 유형."""

    TOOL_RESULT = "TOOL_RESULT"
    NOTE = "NOTE"


class AgentStatus(str, Enum):
    """에이전트 실행 상태."""

    RUNNING = "RUNNING"
    DONE = "DONE"


# ──────────────────────────── Pydantic Models ────────────────────────────


class Action(BaseModel):
    """Policy가 반환하는 다음 Action."""

    type: ActionType
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    note: Optional[str] = None


class Observation(BaseModel):
    """Action 실행 결과를 기록하는 append-only 로그 엔트리."""

    seq: int
    step_id: str
    kind: ObservationKind
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    payload: Optional[dict[str, Any]] = None
    summary: str = ""


# ──────────────────────────── LangGraph TypedDict State ────────────────────────────


class GraphState(TypedDict, total=False):
    """LangGraph StateGraph에서 사용하는 상태 딕셔너리."""

    run_id: str
    goal: str
    cursor: int
    max_steps: int
    status: str  # "RUNNING" | "DONE"
    observations: list[dict[str, Any]]  # Observation.model_dump() 리스트
    final_answer: str
    current_action: dict[str, Any]  # Action.model_dump()


# ──────────────────────────── Tool 입출력 스키마 ────────────────────────────


class RegulationSearchInput(BaseModel):
    """regulation_search Tool 입력."""

    query_text: str
    market: Optional[str] = None  # 유가증권시장, 코스닥시장, 코넥스시장
    reg_type: Optional[str] = None  # 상장규정, 공시규정, 업무규정 등
    top_k: int = 5


class RegulationSearchHit(BaseModel):
    """regulation_search 결과 단건."""

    id: int
    score: float
    text: str
    regulation_name: str
    market: str
    article_title: str


class RegulationSearchOutput(BaseModel):
    """regulation_search Tool 출력."""

    hits: list[RegulationSearchHit] = Field(default_factory=list)
