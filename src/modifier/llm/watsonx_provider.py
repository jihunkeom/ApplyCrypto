"""
WatsonX.AI Provider 구현

IBM WatsonX.AI LLM API를 호출하는 프로바이더입니다.
"""

import logging
import os
from typing import Dict, Any, Optional, List

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

from .llm_provider import LLMProvider


logger = logging.getLogger("applycrypto.code_patcher")


class WatsonXAIProvider(LLMProvider):
    """
    WatsonX.AI Provider 구현 클래스
    
    IBM WatsonX.AI API를 사용하여 LLM 호출을 수행합니다.
    """
    
    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs
    ):
        """
        WatsonX.AI Provider 초기화
        
        Args:
            api_key: WatsonX.AI API 키
            api_url: API URL (선택적, 기본값: 환경변수에서 가져옴)
            model_id: 모델 ID (선택적, 기본값: "ibm/granite-13b-chat-v2")
            project_id: 프로젝트 ID (선택적, 환경변수에서 가져옴)
            **kwargs: 추가 설정 파라미터
        """
        if not WATSONX_AVAILABLE:
            raise ImportError(
                "WatsonX.AI 라이브러리가 설치되지 않았습니다. "
                "다음 명령어로 설치하세요: pip install ibm-watsonx-ai"
            )
        
        self.api_key = api_key
        self.api_url = api_url or os.getenv("WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id = model_id or os.getenv("WATSONX_MODEL_ID", "mistralai/mistral-medium-2505")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        
        # Credentials 초기화
        try:
            credentials = Credentials(
                url=self.api_url,
                api_key=self.api_key
            )
            
            # ModelInference 초기화
            self.model_inference = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
                params=kwargs
            )
            
            logger.info(f"WatsonX.AI Provider 초기화 완료: {self.model_id}")
        except Exception as e:
            logger.error(f"WatsonX.AI Provider 초기화 실패: {e}")
            raise
    
    def call(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        WatsonX.AI API를 호출하여 응답을 받습니다.
        
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
            # 메시지 형식으로 변환
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 파라미터 설정
            params = {}
            if max_tokens:
                params["max_tokens"] = max_tokens
            else:
                params["max_tokens"] = 100000

            if temperature is not None:
                params["temperature"] = temperature
            
            # API 호출 (ModelInference의 chat 메서드 사용)
            response = self.model_inference.chat(messages=messages, params=params)
            
            # 응답 파싱 (최신 형식: response["choices"][0]["message"]["content"])
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 토큰 사용량 추출 (응답에 있는 경우)
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0) if usage else 0
            
            result = {
                "content": content,
                "tokens_used": tokens_used,
                "model": self.model_id
            }
            
            logger.debug(f"WatsonX.AI 응답 수신: {len(content)} 문자, {tokens_used} 토큰")
            return result
            
        except Exception as e:
            logger.error(f"WatsonX.AI API 호출 실패: {e}")
            raise
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        WatsonX.AI 응답의 유효성을 검증합니다.
        
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
            str: "watsonx_ai"
        """
        return "watsonx_ai"

