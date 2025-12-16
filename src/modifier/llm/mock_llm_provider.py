"""
Mock LLM Provider 구현

테스트 목적으로 사용할 수 있는 Mock LLM 프로바이더입니다.
실제 API 호출 없이 미리 정의된 응답을 반환합니다.
"""

import logging
import os
from typing import Any, Dict, Optional

from .llm_provider import LLMProvider

logger = logging.getLogger("applycrypto.code_patcher")


class MockLLMProvider(LLMProvider):
    """
    Mock LLM Provider 구현 클래스

    실제 LLM 호출 없이 테스트를 위한 Mock 응답을 반환합니다.
    """

    def __init__(self, mock_response: Optional[str] = None, **kwargs):
        """
        Mock LLM Provider 초기화

        Args:
            mock_response: 반환할 Mock 응답 내용 (선택적)
                          지정하지 않으면 MOCK_LLM_RESPONSE 환경변수나 기본값을 사용합니다.
            **kwargs: 추가 설정 파라미터
        """
        self.mock_response = (
            mock_response
            or os.getenv("MOCK_LLM_RESPONSE")
            or "This is a mock response from MockLLMProvider."
        )
        logger.info(f"Mock LLM Provider 초기화 완료 (응답 길이: {len(self.mock_response)})")

    def call(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Mock 응답을 반환합니다.

        Args:
            prompt: LLM에 전달할 프롬프트 (로그에만 기록됨)
            max_tokens: 최대 토큰 수 (무시됨)
            temperature: 온도 파라미터 (무시됨)

        Returns:
            Dict[str, Any]: Mock 응답 딕셔너리
        """
        logger.debug(f"Mock LLM 호출됨. 프롬프트 길이: {len(prompt)}")
        
        # 간단한 토큰 수 계산 (단어 수 * 1.3 정도)
        tokens_used = int(len(self.mock_response.split()) * 1.3)
        
        result = {
            "content": self.mock_response,
            "tokens_used": tokens_used,
            "model": "mock-model-v1",
        }
        
        return result

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Mock 응답의 유효성을 검증합니다.

        Args:
            response: 검증할 응답 딕셔너리

        Returns:
            bool: 항상 True (Mock은 항상 유효하다고 가정)
        """
        required_fields = ["content", "tokens_used", "model"]
        return all(field in response for field in required_fields)

    def get_provider_name(self) -> str:
        """
        프로바이더 이름을 반환합니다.

        Returns:
            str: "mock"
        """
        return "mock"
