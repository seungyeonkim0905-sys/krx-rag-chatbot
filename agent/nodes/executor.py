"""Executor 노드: Action을 실행하고 Observation을 누적한다.

CALL_TOOL, WRITE_NOTE 두 가지 Action 유형을 처리한다.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.models import GraphState, ObservationKind
from agent.tools.tool_runtime import execute_tool

logger = logging.getLogger(__name__)


def _make_observation(
    state: GraphState,
    kind: ObservationKind,
    summary: str,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Observation 딕셔너리를 생성한다."""
    cursor = state.get("cursor", 0)
    observations = state.get("observations", [])
    seq = len(observations)

    return {
        "seq": seq,
        "step_id": f"turn_{cursor}",
        "kind": kind.value,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "payload": payload,
        "summary": summary,
    }


def tool_executor_node(state: GraphState) -> dict[str, Any]:
    """CALL_TOOL Action을 처리하는 노드."""
    action = state.get("current_action", {})
    tool_name = action.get("tool_name", "")
    tool_args = action.get("tool_args", {})

    logger.info("Executor: Tool 실행 - %s", tool_name)

    try:
        result = execute_tool(tool_name, tool_args)
        summary = _summarize_tool_result(tool_name, result)
        observation = _make_observation(
            state,
            kind=ObservationKind.TOOL_RESULT,
            summary=summary,
            tool_name=tool_name,
            tool_args=tool_args,
            payload=result,
        )
    except Exception as e:
        logger.error("Tool 실행 오류: %s", str(e))
        observation = _make_observation(
            state,
            kind=ObservationKind.TOOL_RESULT,
            summary=f"Tool '{tool_name}' 실행 오류: {str(e)}",
            tool_name=tool_name,
            tool_args=tool_args,
        )

    observations = list(state.get("observations", []))
    observations.append(observation)
    cursor = state.get("cursor", 0) + 1

    return {"observations": observations, "cursor": cursor}


def write_note_node(state: GraphState) -> dict[str, Any]:
    """WRITE_NOTE Action을 처리하는 노드."""
    action = state.get("current_action", {})
    note = action.get("note", "")

    logger.info("Executor: 노트 작성 - %s", note[:50])

    observation = _make_observation(
        state,
        kind=ObservationKind.NOTE,
        summary=note,
    )

    observations = list(state.get("observations", []))
    observations.append(observation)
    cursor = state.get("cursor", 0) + 1

    return {"observations": observations, "cursor": cursor}


def stop_node(state: GraphState) -> dict[str, Any]:
    """STOP Action을 처리하는 노드."""
    action = state.get("current_action", {})
    note = action.get("note", "에이전트가 중단되었습니다.")

    logger.info("Executor: STOP - %s", note[:50])

    return {
        "final_answer": note,
        "status": "DONE",
    }


def _summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Tool 결과를 요약 문자열로 변환한다."""
    if tool_name == "regulation_search":
        hits = result.get("hits", [])
        if not hits:
            return "regulation_search: 결과 없음"
        lines = [f"regulation_search: {len(hits)}건 검색됨"]
        for i, hit in enumerate(hits[:5]):
            reg_name = hit.get("regulation_name", "")
            article = hit.get("article_title", "")
            text_preview = hit.get("text", "")[:500]
            score = hit.get("score", 0)
            lines.append(f"  [{i+1}] [{reg_name}] {article} (score: {score:.3f})")
            lines.append(f"      {text_preview}")
        return "\n".join(lines)
    return f"{tool_name}: 실행 완료"
