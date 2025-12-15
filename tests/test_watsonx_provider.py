"""
WatsonXAIProvider 통합 테스트

IBM WatsonX.AI Provider의 실제 API 호출을 테스트합니다.
환경변수에서 credential 정보를 로딩합니다.
"""

import pytest
import os
from typing import Dict, Any, Optional
from pathlib import Path

# 테스트 전에 모듈 import
try:
    from modifier.llm.watsonx_provider import WatsonXAIProvider, WATSONX_AVAILABLE
except ImportError:
    WatsonXAIProvider = None
    WATSONX_AVAILABLE = False


def load_env():
    """
    .env 파일에서 환경변수를 로딩합니다.
    python-dotenv를 사용하여 .env 파일을 읽습니다.
    """
    try:
        from dotenv import load_dotenv
        # 프로젝트 루트 디렉토리 찾기
        project_root = Path(__file__).parent.parent
        print(f"project_root: {project_root}")
        env_file = project_root / ".env"
        
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✓ .env 파일 로딩 완료: {env_file}")
        else:
            print(f"⚠ .env 파일을 찾을 수 없습니다: {env_file}")
            print("  환경변수를 직접 설정하거나 .env 파일을 생성하세요.")
    except ImportError:
        print("⚠ python-dotenv가 설치되지 않았습니다. 환경변수를 직접 설정하세요.")
        print("  설치: pip install python-dotenv")


def get_watsonx_credentials() -> Optional[Dict[str, str]]:
    """
    환경변수에서 WatsonX.AI credential 정보를 가져옵니다.
    
    Returns:
        Dict[str, str]: credential 정보 (api_key, api_url, project_id, model_id)
                       또는 None (필수 정보가 없을 경우)
    """
    api_key = os.getenv("WATSONX_API_KEY")
    api_url = os.getenv("WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("WATSONX_MODEL_ID", "mistralai/mistral-medium-2505")
    
    if not api_key:
        return None
    
    return {
        "api_key": api_key,
        "api_url": api_url,
        "project_id": project_id,
        "model_id": model_id
    }


def has_watsonx_credentials() -> bool:
    """
    WatsonX.AI credential이 환경변수에 설정되어 있는지 확인합니다.
    
    Returns:
        bool: credential이 설정되어 있으면 True
    """
    return get_watsonx_credentials() is not None


class TestWatsonXAIProvider:
    """WatsonXAIProvider 통합 테스트 클래스"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_env(self):
        pass
        # 테스트 후 정리 작업이 필요한 경우 여기에 추가
    
    @pytest.fixture
    def provider_config(self):
        """테스트용 프로바이더 설정 (환경변수에서 로딩)"""
        creds = get_watsonx_credentials()
        if not creds:
            pytest.skip("WatsonX.AI credential이 환경변수에 설정되지 않았습니다. "
                       "WATSONX_API_KEY, WATSONX_PROJECT_ID를 설정하세요.")
        return creds
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_provider_initialization_success(self, provider_config):
        """프로바이더 초기화 성공 테스트 (실제 API 연결)"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        assert provider is not None
        assert provider.api_key == provider_config["api_key"]
        assert provider.api_url == provider_config.get("api_url", "https://us-south.ml.cloud.ibm.com")
        assert provider.model_id == provider_config.get("model_id", "ibm/granite-13b-chat-v2")
        if provider_config.get("project_id"):
            assert provider.project_id == provider_config["project_id"]
        assert provider.get_provider_name() == "watsonx_ai"
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_call_success(self, provider_config):
        """실제 API 호출 성공 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        result = provider.call("안녕하세요. 간단히 자기소개를 해주세요.")
        
        assert result is not None
        assert "content" in result
        assert "tokens_used" in result
        assert "model" in result
        assert len(result["content"]) > 0, "응답 내용이 비어있습니다"
        assert result["tokens_used"] >= 0, "토큰 사용량이 음수입니다"
        assert result["model"] == provider_config.get("model_id", "ibm/granite-13b-chat-v2")
        
        print(f"\n✓ API 호출 성공:")
        print(f"  - 응답 길이: {len(result['content'])} 문자")
        print(f"  - 토큰 사용량: {result['tokens_used']}")
        print(f"  - 모델: {result['model']}")
        print(f"  - 응답 미리보기: {result['content'][:100]}...")
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_call_with_parameters(self, provider_config):
        """파라미터를 포함한 실제 API 호출 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        result = provider.call(
            prompt="한 문장으로 Python 프로그래밍 언어에 대해 설명해주세요.",
            max_tokens=100,
            temperature=0.7
        )
        
        assert result is not None
        assert "content" in result
        assert len(result["content"]) > 0
        
        print(f"\n✓ 파라미터 포함 API 호출 성공:")
        print(f"  - 응답: {result['content']}")
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_call_multiple_requests(self, provider_config):
        """여러 번의 API 호출 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        prompts = [
            "1+1은?",
            "2+2는?",
            "3+3은?"
        ]
        
        results = []
        for prompt in prompts:
            result = provider.call(prompt)
            assert result is not None
            assert "content" in result
            assert len(result["content"]) > 0
            results.append(result)
        
        assert len(results) == len(prompts)
        print(f"\n✓ {len(prompts)}번의 연속 API 호출 성공")
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_validate_response_valid(self, provider_config):
        """실제 응답 검증 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        result = provider.call("테스트")
        
        # 실제 응답 검증
        assert provider.validate_response(result) is True
        
        # 유효하지 않은 응답 검증
        invalid_response1 = {
            "content": "테스트 응답"
            # tokens_used, model 누락
        }
        assert provider.validate_response(invalid_response1) is False
        
        invalid_response2 = {
            "content": "",
            "tokens_used": 100,
            "model": "test-model"
        }
        assert provider.validate_response(invalid_response2) is False
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_get_provider_name(self, provider_config):
        """프로바이더 이름 반환 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        assert provider.get_provider_name() == "watsonx_ai"
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    def test_provider_initialization_without_credentials(self):
        """credential이 없을 때 초기화 실패 테스트"""
        # 잘못된 API 키로 초기화 시도
        with pytest.raises(Exception):
            WatsonXAIProvider(
                api_key="invalid-api-key",
                project_id="invalid-project-id"
            )
    
    @pytest.mark.skipif(not WATSONX_AVAILABLE, reason="ibm-watsonx-ai 라이브러리가 설치되지 않았습니다")
    @pytest.mark.skipif(not has_watsonx_credentials(), reason="WatsonX.AI credential이 환경변수에 설정되지 않았습니다")
    def test_call_with_long_prompt(self, provider_config):
        """긴 프롬프트를 사용한 API 호출 테스트"""
        provider = WatsonXAIProvider(
            api_key=provider_config["api_key"],
            api_url=provider_config.get("api_url"),
            model_id=provider_config.get("model_id"),
            project_id=provider_config.get("project_id")
        )
        
        long_prompt = "Python 프로그래밍 언어에 대해 설명해주세요. " * 10
        result = provider.call(long_prompt)
        
        assert result is not None
        assert "content" in result
        assert len(result["content"]) > 0
        
        print(f"\n✓ 긴 프롬프트 API 호출 성공:")
        print(f"  - 프롬프트 길이: {len(long_prompt)} 문자")
        print(f"  - 응답 길이: {len(result['content'])} 문자")


if __name__ == "__main__":
    # main에서 환경변수 로딩
    """테스트 클래스 시작 시 환경변수 로딩"""
    load_env()

    # 테스트 실행
    pytest.main([__file__, "-v", "-s"])
