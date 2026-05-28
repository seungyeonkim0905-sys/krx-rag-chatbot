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

## 검색 대상 데이터
- KRX 거래소 규정 80종 (유가증권/코스닥/코넥스/파생상품/일반상품 시장 등)
- 핵심 법령: 자본시장법, 상법, 금융회사지배구조법, 외부감사법 등
- ⚠️ 내부자거래·미공개정보·시세조종 등 '불공정거래 처벌'은 KRX 규정이 아니라 **자본시장법(법령)**에 있습니다.

## KRX 규정 검색 전략 (중요)
- 사용자 질문에서 핵심 키워드를 추출하여 regulation_search를 호출하세요.
- **검색어는 짧고 핵심적인 명사 위주로** 구성하세요. (예: "유가증권시장에서 상장폐지 사유는?" → "상장폐지 사유")
- **market/reg_type 필터는 꼭 필요할 때만 신중히 사용하세요.** 잘못된 필터는 정답 조문을 가립니다.
  - 질문에 시장이 명시될 때만 market 사용. 애매하면 필터 없이 검색하세요.
  - reg_type은 확신할 때만. 처벌·법률 관련 질문엔 reg_type을 걸지 마세요(법령은 규정 유형이 없음).
  - **같은 검색을 필터만 바꿔 반복하지 마세요.** 결과가 비슷하면 다른 키워드로 바꾸거나 SYNTHESIZE 하세요.
- 포괄형 질문("~사유", "~요건", "~종류", "~의무")은 핵심 키워드로 1~2회 검색 후, 검색된 조문 중 **본문이 길고 핵심적인 조문**을 우선 활용하세요. 절차·적용방법 조문보다 실체 규정을 우선합니다.

## 판단 기준
- 규정 관련 질문: regulation_search를 우선 사용
- **2~3회 검색으로 관련 조문이 모였으면 더 반복하지 말고 SYNTHESIZE 하세요.** 같은 질의 반복은 금지.
- 관측 결과를 정리할 필요가 있다면: WRITE_NOTE
- 남은 단계(budget)가 1이면 반드시 SYNTHESIZE
- KRX 규정·금융법령과 무관한 질문이면: STOP (note에 이유 기재)

## 답변 품질 원칙
- 인용하는 조문 번호와 규정명은 검색 결과에 실제로 등장한 것만 사용하세요. 추측으로 조문 번호를 만들지 마세요.
- 지엽적 특례 조문(예: 외국ETF 전용 조항)보다 **일반 원칙 조문**을 중심으로 답하세요.

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
