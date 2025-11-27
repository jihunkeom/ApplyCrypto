"""
Claude AI Provider 구현

Anthropic Claude AI LLM API를 호출하는 프로바이더입니다.
"""

import logging
from typing import Dict, Any, Optional

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .llm_provider import LLMProvider


logger = logging.getLogger("applycrypto.code_patcher")


class ClaudeAIProvider(LLMProvider):
    """
    Claude AI Provider 구현 클래스
    
    Anthropic Claude AI API를 사용하여 LLM 호출을 수행합니다.
    """
    
    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        **kwargs
    ):
        """
        Claude AI Provider 초기화
        
        Args:
            api_key: Anthropic API 키
            model_id: 모델 ID (선택적, 기본값: "claude-sonnet-4-20250514")
            **kwargs: 추가 설정 파라미터
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic 라이브러리가 설치되지 않았습니다. "
                "다음 명령어로 설치하세요: pip install anthropic"
            )
        
        self.api_key = api_key
        self.model_id = model_id or "claude-sonnet-4-20250514"
        
        # Anthropic 클라이언트 초기화
        try:
            self.client = Anthropic(api_key=self.api_key)
            
            logger.info(f"Claude AI Provider 초기화 완료: {self.model_id}")
        except Exception as e:
            logger.error(f"Claude AI Provider 초기화 실패: {e}")
            raise
    
    def call(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Claude AI API를 호출하여 응답을 받습니다.
        스트리밍 방식을 사용하여 응답을 수집합니다.
        
        Args:
            prompt: LLM에 전달할 프롬프트
            max_tokens: 최대 토큰 수 (선택적)
            temperature: 온도 파라미터 (선택적)
            
        Returns:
            Dict[str, Any]: LLM 응답 딕셔너리
                - content: 응답 내용
                - tokens_used: 사용된 토큰 수
                - model: 사용된 모델명
        """
        try:
            # 파라미터 설정
            params = {
                "model": self.model_id,
                "max_tokens": max_tokens or 64000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            if temperature is not None:
                params["temperature"] = temperature
            
            # 스트리밍 방식으로 API 호출
            content_parts = []
            tokens_used = 0
            
            with self.client.messages.stream(**params) as stream:
                # 스트리밍된 텍스트 수집
                for text in stream.text_stream:
                    content_parts.append(text)
                
                # 최종 메시지에서 사용량 정보 가져오기
                final_message = stream.get_final_message()
                if final_message.usage:
                    tokens_used = final_message.usage.input_tokens + final_message.usage.output_tokens
            
            # 수집한 텍스트를 하나의 문자열로 결합
            content = "".join(content_parts)
            
            result = {
                "content": content,
                "tokens_used": tokens_used,
                "model": self.model_id
            }
            
            logger.debug(f"Claude AI 응답 수신: {len(content)} 문자, {tokens_used} 토큰")
            return result
            
        except Exception as e:
            logger.error(f"Claude AI API 호출 실패: {e}")
            raise
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Claude AI 응답의 유효성을 검증합니다.
        
        Args:
            response: 검증할 응답 딕셔너리
            
        Returns:
            bool: 응답이 유효하면 True, 그렇지 않으면 False
        """
        required_fields = ["content", "tokens_used", "model"]
        return all(field in response for field in required_fields) and bool(response.get("content"))
    
    def get_provider_name(self) -> str:
        """
        프로바이더 이름을 반환합니다.
        
        Returns:
            str: "claude_ai"
        """
        return "claude_ai"

