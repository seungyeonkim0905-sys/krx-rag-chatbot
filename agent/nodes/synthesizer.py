"""Synthesizer 노드: 누적된 Observation을 근거로 최종 답변을 생성한다.

KRX 규정 챗봇에 특화된 답변 합성 프롬프트를 사용한다.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from agent.llm_client import get_llm_client
from agent.models import AgentStatus, GraphState

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """\
당신은 KRX(한국거래소) 규정 전문 AI 어시스턴트입니다.
사용자의 질문과 검색된 규정 조문을 바탕으로 **정확하고 신뢰할 수 있는 답변**을 작성합니다.

## 답변 규칙
1. **반드시 검색된 규정 조문에 근거하여 답변하세요.** 규정에 없는 내용은 추측하지 마세요.
2. 답변에 **근거 조문을 명시하세요** (예: "유가증권시장 상장규정 제7조에 따르면...").
3. 관련 조문이 여러 개이면 체계적으로 정리하여 설명하세요.
4. 검색 결과에 관련 정보가 없으면 솔직하게 "검색된 규정에서 해당 내용을 찾지 못했습니다"라고 답하세요.
5. 규정의 원문 표현을 최대한 살려서 정확하게 전달하세요.
6. 한국어로 답변하세요.
7. 답변 마지막에 **[참고 규정]** 섹션을 추가하여 인용한 규정명과 조문 번호를 정리하세요.
"""


def build_synthesis_prompt(state: GraphState) -> str:
    """Synthesizer에 전달할 user prompt를 구성한다."""
    observations_text = "없음"
    if state.get("observations"):
        obs_lines = []
        for obs in state["observations"]:
            kind = obs.get("kind", "?")
            summary = obs.get("summary", "")
            payload = obs.get("payload")

            line = f"[{obs.get('step_id', '?')}] ({kind}) {summary}"
            if payload and "hits" in payload:
                for hit in payload["hits"][:5]:
                    reg = hit.get("regulation_name", "")
                    article = hit.get("article_title", "")
                    text = hit.get("text", "")[:1000]
                    line += f"\n  --- [{reg}] {article} ---"
                    line += f"\n  {text}"
            obs_lines.append(line)
        observations_text = "\n".join(obs_lines)

    return f"""\
## 사용자 질문
{state.get("goal", "")}

## 검색된 규정 조문
{observations_text}

위 규정 조문을 근거로 사용자 질문에 대한 최종 답변을 작성하세요.
"""


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """Synthesizer 노드: 최종 답변을 생성한다."""
    logger.info("Synthesizer 노드 실행")

    user_prompt = build_synthesis_prompt(state)
    llm = get_llm_client()

    try:
        final_answer = llm.generate(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)
        logger.info("Synthesizer 완료: %d chars", len(final_answer))
    except Exception as e:
        logger.error("Synthesizer 에러: %s", str(e))
        final_answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    return {
        "final_answer": final_answer,
        "status": AgentStatus.DONE.value,
    }


def stream_synthesis(state: GraphState) -> Generator[str, None, None]:
    """Synthesizer를 스트리밍 방식으로 실행한다."""
    logger.info("Synthesizer 스트리밍 시작")

    user_prompt = build_synthesis_prompt(state)
    llm = get_llm_client()

    yield from llm.generate_stream(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)
