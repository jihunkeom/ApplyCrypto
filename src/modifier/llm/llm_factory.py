"""
LLM Provider Factory

설정에 따라 적절한 LLM 프로바이더를 생성하는 팩토리 클래스입니다.
"""

import logging
import os
from typing import Optional

from .claude_ai_provider import ClaudeAIProvider
from .llm_provider import LLMProvider
from .mock_llm_provider import MockLLMProvider
from .openai_provider import OpenAIProvider
from .watsonx_provider import WatsonXAIProvider

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """LLM Provider 관련 오류"""

    pass


def create_llm_provider(provider_name: str) -> LLMProvider:
    """
    설정에 따라 적절한 LLM 프로바이더를 생성합니다.
    필요한 credential 정보는 환경변수에서 가져옵니다.

    Args:
        provider_name: 프로바이더 이름 ("watsonx_ai", "openai", "claude_ai", "mock")

    Returns:
        LLMProvider: 생성된 LLM 프로바이더 인스턴스

    Raises:
        LLMProviderError: 지원하지 않는 프로바이더이거나 생성 실패 시
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower == "watsonx_ai" or provider_name_lower == "watsonx":
        # 환경변수에서 credential 정보 가져오기
        api_key = os.getenv("WATSONX_API_KEY")
        api_url = os.getenv("WATSONX_API_URL")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = os.getenv("WATSONX_MODEL_ID")

        if not api_key:
            raise LLMProviderError(
                "WatsonX.AI 프로바이더는 WATSONX_API_KEY 환경변수가 필요합니다."
            )

        return WatsonXAIProvider(
            api_key=api_key, api_url=api_url, model_id=model_id, project_id=project_id
        )

    elif provider_name_lower == "openai":
        # 환경변수에서 credential 정보 가져오기
        api_key = os.getenv("OPENAI_API_KEY")
        model_id = os.getenv("OPENAI_MODEL_ID")

        if not api_key:
            raise LLMProviderError(
                "OpenAI 프로바이더는 OPENAI_API_KEY 환경변수가 필요합니다."
            )

        return OpenAIProvider(api_key=api_key, model_id=model_id)

    elif provider_name_lower == "claude_ai":
        # 환경변수에서 credential 정보 가져오기
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model_id = os.getenv("ANTHROPIC_MODEL_ID", "claude-sonnet-4-20250514")

        if not api_key:
            raise LLMProviderError(
                "Claude AI 프로바이더는 ANTHROPIC_API_KEY 환경변수가 필요합니다."
            )

        return ClaudeAIProvider(api_key=api_key, model_id=model_id)

    elif provider_name_lower == "mock":
        return MockLLMProvider()

    else:
        raise LLMProviderError(
            f"지원하지 않는 LLM 프로바이더: {provider_name}. "
            f"지원하는 프로바이더: watsonx_ai, openai, claude_ai, mock"
        )
