"""
WatsonX.AI OnPremise Provider 구현

IBM WatsonX.AI OnPremise LLM API를 호출하는 프로바이더입니다.
"""

import json
import logging
import os
import warnings
from typing import Any, Dict, Optional

import requests

from .llm_provider import LLMProvider

warnings.filterwarnings("ignore")

logger = logging.getLogger("applycrypto.code_patcher")


class WatsonXAIOnPremiseProvider(LLMProvider):
    """
    WatsonX.AI OnPremise Provider 구현 클래스

    IBM WatsonX.AI OnPremise API를 사용하여 LLM 호출을 수행합니다.
    """

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_name: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        WatsonX.AI OnPremise Provider 초기화

        Args:
            api_key: WatsonX.AI API 키
            api_url: API URL (선택적, 기본값: 환경변수에서 가져옴)
            model_id: 모델 ID (선택적, 기본값: "ibm/granite-13b-chat-v2")
            project_id: 프로젝트 ID (선택적, 환경변수에서 가져옴)
            user_name: 사용자 이름
            password: 비밀번호
            **kwargs: 추가 설정 파라미터
        """
        self.user_name = user_name
        self.password = password
        self.api_key = api_key
        self.api_url = api_url or os.getenv(
            "WATSONX_API_URL", "https://cpd-zen.apps.wca.samsunglife.kr"
        )
        self.model_id = model_id or os.getenv(
            "WATSONX_ON_PREMISE_MODEL_ID", "ibm/granite-3-3-8b-instruct"
        )
        self.project_id = project_id or os.getenv("WATSONX_ON_PREMISE_PROJECT_ID")

    def _get_credentials(self) -> Dict[str, Any]:
        """
        WatsonX.AI OnPremise Bearer Token를 가져옵니다.
        """
        try:
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "username": self.user_name,
                "password": self.password,
                "api_key": self.api_key,
            }

            response = requests.post(
                f"{self.api_url}/icp4d-api/v1/authorize",
                headers=headers,
                data=json.dumps(data),
                verify=False
            )
            response.raise_for_status()
            return response.json().get("token")
        except Exception as e:
            logger.error(f"WatsonX.AI OnPremise Bearer Token 가져오기 실패: {e}")
            raise

    def call(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
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
            iam_token = self._get_credentials()
            url = f"{self.api_url}/m1/v1/text/chat?version=2023-05-29"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {iam_token}",
            }
            body = {
                "messages": [{"role":"user","text":prompt}],
                "project_id": self.project_id,
                "model_id": self.model_id,
                "frequency_penalty": 0,
                "max_tokens": max_tokens or 100000,
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1,
            }

            response = requests.post(
                url,
                headers=headers,
                json=body,
                verify=False,
            )
            response = response.json()

            # 응답 파싱 (최신 형식: response["choices"][0]["message"]["content"])
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # 토큰 사용량 추출 (응답에 있는 경우)
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0) if usage else 0

            result = {
                "content": content,
                "tokens_used": tokens_used,
                "model": self.model_id,
            }

            logger.debug(
                f"WatsonX.AI 응답 수신: {len(content)} 문자, {tokens_used} 토큰"
            )
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
        return all(field in response for field in required_fields) and bool(
            response.get("content")
        )

    def get_provider_name(self) -> str:
        """
        프로바이더 이름을 반환합니다.

        Returns:
            str: "watsonx_ai"
        """
        return "watsonx_ai_on_prem"
