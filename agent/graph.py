"""LangGraph StateGraph 정의 및 컴파일.

플로우:
  normalize_goal → policy_node → (route_action) →
    CALL_TOOL → tool_executor → (check_budget) → policy_node 또는 force_synthesizer
    WRITE_NOTE → write_note → (check_budget) → policy_node 또는 force_synthesizer
    SYNTHESIZE → synthesizer (placeholder) → END
    STOP → stop_node → END
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from agent.config import settings
from agent.models import ActionType, AgentStatus, GraphState
from agent.nodes.executor import stop_node, tool_executor_node, write_note_node
from agent.nodes.policy import policy_node

logger = logging.getLogger(__name__)


def normalize_goal(state: GraphState) -> dict[str, Any]:
    """목표를 정규화하고 초기 상태를 설정한다."""
    run_id = state.get("run_id") or uuid.uuid4().hex[:12]
    goal = state.get("goal", "")
    max_steps = state.get("max_steps") or settings.max_steps

    logger.info("에이전트 시작: run_id=%s, goal='%s'", run_id, goal[:50])

    return {
        "run_id": run_id,
        "goal": goal,
        "cursor": 0,
        "max_steps": max_steps,
        "status": AgentStatus.RUNNING.value,
        "observations": [],
        "final_answer": "",
        "current_action": {},
    }


def route_action(state: GraphState) -> str:
    """Policy의 Action 유형에 따라 다음 노드를 결정한다."""
    action = state.get("current_action", {})
    action_type = action.get("type", "STOP")

    if action_type == ActionType.CALL_TOOL.value:
        return "tool_executor"
    elif action_type == ActionType.WRITE_NOTE.value:
        return "write_note"
    elif action_type == ActionType.SYNTHESIZE.value:
        return "synthesizer"
    else:
        return "stop"


def check_budget(state: GraphState) -> str:
    """예산(budget) 체크로 루프 계속 여부를 결정한다."""
    cursor = state.get("cursor", 0)
    max_steps = state.get("max_steps", 5)
    status = state.get("status", "RUNNING")

    if status == AgentStatus.DONE.value:
        return "force_synthesizer"

    if cursor >= max_steps:
        logger.info("예산 초과 (%d/%d) - 강제 종료", cursor, max_steps)
        return "force_synthesizer"

    return "policy"


def _synthesis_placeholder(state: GraphState) -> dict[str, Any]:
    """스트리밍 전용 placeholder 노드."""
    return {"status": AgentStatus.DONE.value}


def build_streaming_graph():
    """스트리밍 전용 LangGraph StateGraph를 구성하고 컴파일한다."""
    graph = StateGraph(GraphState)

    graph.add_node("normalize_goal", normalize_goal)
    graph.add_node("policy", policy_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("write_note", write_note_node)
    graph.add_node("synthesizer", _synthesis_placeholder)
    graph.add_node("stop", stop_node)
    graph.add_node("force_synthesizer", _synthesis_placeholder)

    graph.set_entry_point("normalize_goal")
    graph.add_edge("normalize_goal", "policy")

    graph.add_conditional_edges(
        "policy",
        route_action,
        {
            "tool_executor": "tool_executor",
            "write_note": "write_note",
            "synthesizer": "synthesizer",
            "stop": "stop",
        },
    )

    graph.add_conditional_edges(
        "tool_executor",
        check_budget,
        {
            "policy": "policy",
            "force_synthesizer": "force_synthesizer",
        },
    )

    graph.add_conditional_edges(
        "write_note",
        check_budget,
        {
            "policy": "policy",
            "force_synthesizer": "force_synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)
    graph.add_edge("stop", END)
    graph.add_edge("force_synthesizer", END)

    compiled = graph.compile()
    logger.info("LangGraph 그래프 컴파일 완료")
    return compiled


_compiled_streaming_graph = None


def get_streaming_graph():
    """스트리밍 전용 그래프 싱글턴을 반환한다."""
    global _compiled_streaming_graph
    if _compiled_streaming_graph is None:
        _compiled_streaming_graph = build_streaming_graph()
    return _compiled_streaming_graph
