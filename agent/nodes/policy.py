"""Policy 노드: LLM을 호출하여 다음 Action을 결정한다.

KRX 규정 챗봇에 특화된 Policy 프롬프트를 사용한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.llm_client import get_llm_client
from agent.models import Action, ActionType, GraphState
from agent.tools.tool_runtime import get_tool_descriptions

logger = logging.getLogger(__name__)

POLICY_SYSTEM_PROMPT = """\
당신은 KRX(한국거래소) 규정 전문 AI 에이전트의 Policy 모듈입니다.
사용자의 질문(goal)과 지금까지의 관측 기록(observations)을 보고,
**다음에 수행할 Action 1개**를 JSON으로 반환해야 합니다.

## 사용 가능한 Action 유형
1. CALL_TOOL — Tool을 호출합니다. tool_name과 tool_args를 반드시 포함하세요.
2. WRITE_NOTE — 중간 요약/정리 메모를 작성합니다. note 필드에 내용을 포함하세요.
3. SYNTHESIZE — 충분한 정보가 모였다고 판단하면, 최종 답변 생성을 요청합니다.
4. STOP — 더 이상 진행할 수 없거나 의미가 없을 때 중단합니다. note 필드에 이유를 포함하세요.

## 사용 가능한 Tool
{tool_descriptions}

## KRX 규정 검색 전략
- 사용자 질문에서 핵심 키워드를 추출하여 regulation_search를 호출하세요.
- 시장(유가증권시장/코스닥시장/코넥스시장)이 특정되면 market 파라미터를 활용하세요.
- 규정 유형(상장규정/공시규정/업무규정/시행세칙)이 특정되면 reg_type 파라미터를 활용하세요.
- 첫 검색 결과가 부족하면 키워드를 바꿔서 재검색하세요.
- 관련 조문이 여러 규정에 걸칠 수 있으므로 필요시 다른 규정도 검색하세요.

## 판단 기준
- 규정 관련 질문: regulation_search를 우선 사용 (다양한 키워드로 검색)
- 이미 충분한 규정 조문이 관측 기록에 있다면: SYNTHESIZE
- 관측 결과를 정리할 필요가 있다면: WRITE_NOTE
- 남은 단계(budget)가 1이면 반드시 SYNTHESIZE
- KRX 규정과 무관한 질문이면: STOP (note에 이유 기재)

## 응답 형식 (반드시 아래 JSON 형식만 반환)
```json
{{
  "type": "CALL_TOOL" | "WRITE_NOTE" | "SYNTHESIZE" | "STOP",
  "tool_name": "regulation_search",
  "tool_args": {{ ... }},
  "note": "..."
}}
```
- CALL_TOOL일 때만 tool_name, tool_args 포함
- WRITE_NOTE, STOP일 때만 note 포함
- SYNTHESIZE일 때는 type만 포함
"""


def _build_user_prompt(state: GraphState) -> str:
    """Policy에 전달할 user prompt를 구성한다."""
    observations_text = "없음"
    if state.get("observations"):
        obs_lines = []
        for obs in state["observations"]:
            obs_lines.append(
                f"[{obs.get('step_id', '?')}] ({obs.get('kind', '?')}) "
                f"{obs.get('summary', '')}"
            )
        observations_text = "\n".join(obs_lines)

    remaining = state.get("max_steps", 5) - state.get("cursor", 0)

    return f"""\
## 현재 상태
- 사용자 질문: {state.get("goal", "")}
- 진행 단계: {state.get("cursor", 0)} / {state.get("max_steps", 5)} (남은 단계: {remaining})
- 상태: {state.get("status", "RUNNING")}

## 관측 기록
{observations_text}

위 정보를 바탕으로 다음 Action을 JSON으로 반환하세요.
"""


def policy_node(state: GraphState) -> dict[str, Any]:
    """Policy 노드: LLM을 호출하여 다음 Action을 결정한다."""
    logger.info("Policy 노드 실행 (cursor=%d)", state.get("cursor", 0))

    system_prompt = POLICY_SYSTEM_PROMPT.format(
        tool_descriptions=get_tool_descriptions()
    )
    user_prompt = _build_user_prompt(state)

    llm = get_llm_client()

    try:
        action_dict = llm.generate_json(system_prompt, user_prompt)
        action = Action(**action_dict)
        logger.info("Policy 결정: %s", action.type.value)
        return {"current_action": action.model_dump(mode="json")}
    except Exception as e:
        logger.error("Policy 에러, STOP 반환: %s", str(e))
        fallback = Action(
            type=ActionType.STOP,
            note=f"Policy 결정 중 오류 발생: {str(e)}",
        )
        return {"current_action": fallback.model_dump(mode="json")}
