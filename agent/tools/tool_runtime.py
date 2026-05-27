"""Tool 디스패치 모듈.

tool_name을 기반으로 적절한 Tool 함수를 실행한다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent.tools.regulation_search import regulation_search

logger = logging.getLogger(__name__)

# Tool 이름 → 실행 함수 매핑 레지스트리
TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "regulation_search": regulation_search,
}

# Tool 설명 (Policy 프롬프트에 제공)
TOOL_DESCRIPTIONS: dict[str, str] = {
    "regulation_search": (
        "KRX(한국거래소) 규정 데이터베이스에서 관련 규정 조문을 검색합니다. "
        "파라미터: query_text(str, 필수 - 검색할 키워드나 문장), "
        "market(str, optional - '유가증권시장'/'코스닥시장'/'코넥스시장'), "
        "reg_type(str, optional - '상장규정'/'공시규정'/'업무규정'/'시행세칙'), "
        "top_k(int, 기본 5)"
    ),
}


def get_available_tools() -> list[str]:
    """사용 가능한 Tool 이름 목록을 반환한다."""
    return list(TOOL_REGISTRY.keys())


def get_tool_descriptions() -> str:
    """Policy 프롬프트에 포함할 Tool 설명 문자열을 생성한다."""
    lines: list[str] = []
    for name, desc in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _normalize_tool_name(tool_name: str) -> str:
    """Tool 이름을 정규화한다."""
    normalized = str(tool_name)
    if "." in normalized:
        normalized = normalized.split(".")[-1].lower()
    return normalized


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """지정된 Tool을 실행하고 결과를 반환한다."""
    tool_name = _normalize_tool_name(tool_name)

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"등록되지 않은 Tool: '{tool_name}'. "
            f"사용 가능: {get_available_tools()}"
        )

    logger.info("Tool 실행: %s, args=%s", tool_name, tool_args)
    result = TOOL_REGISTRY[tool_name](tool_args)
    logger.info("Tool 실행 완료: %s", tool_name)
    return result
