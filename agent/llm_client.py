"""LLM 클라이언트 추상 인터페이스 및 Google Generative AI 구현체."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import google.generativeai as genai

from agent.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM 클라이언트 추상 인터페이스."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """텍스트 생성."""

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """JSON 형식으로 텍스트 생성."""

    @abstractmethod
    def generate_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """텍스트를 스트리밍 방식으로 생성한다."""


class GoogleGenAIClient(LLMClient):
    """Google Generative AI 기반 LLM 클라이언트."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key or settings.google_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(self._model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Google Generative AI를 사용하여 텍스트를 생성한다."""
        logger.debug("LLM generate 호출: model=%s", self._model)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._client.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self._max_tokens,
            ),
        )
        text = response.text
        logger.debug("LLM 응답 길이: %d chars", len(text))
        return text

    def generate_stream(
        self, system_prompt: str, user_prompt: str
    ) -> Generator[str, None, None]:
        """Google Generative AI의 stream()을 사용하여 텍스트를 스트리밍한다."""
        logger.debug("LLM generate_stream 호출: model=%s", self._model)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._client.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self._max_tokens,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Google Generative AI를 사용하여 JSON 응답을 생성한다."""
        raw = self.generate(system_prompt, user_prompt)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """텍스트에서 JSON 객체를 추출한다."""
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            first_brace = text.index("{")
            last_brace = text.rindex("}") + 1
            json_str = text[first_brace:last_brace]

        return json.loads(json_str)


def get_llm_client() -> LLMClient:
    """기본 LLM 클라이언트 인스턴스를 반환한다."""
    return GoogleGenAIClient()
