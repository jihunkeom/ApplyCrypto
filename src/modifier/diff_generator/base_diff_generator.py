import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import tiktoken
from jinja2 import Template

from models.diff_generator import DiffGeneratorInput, DiffGeneratorOutput
from modifier.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def render_template(template_str: str, variables: Dict[str, Any]) -> str:
    """
    Jinja2를 사용하여 템플릿을 렌더링합니다.

    Args:
        template_str: 템플릿 문자열
        variables: 치환할 변수 딕셔너리

    Returns:
        str: 렌더링된 문자열
    """

    template = Template(template_str)
    return template.render(**variables)


class BaseDiffGenerator:
    """Diff 생성기 베이스 클래스"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_cache: Dict[str, Dict[str, Any]] = None,
        template_path: Optional[Path] = None,
    ):
        """
        BaseDiffGenerator 초기화

        Args:
            llm_provider: LLM 프로바이더
            prompt_cache: 프롬프트 캐시 저장소 (선택적)
            template_path: 템플릿 파일 경로
        """
        self.llm_provider = llm_provider
        self._prompt_cache = prompt_cache if prompt_cache is not None else {}

        if template_path:
            self.template_path = Path(template_path)
        else:
            # 클래스가 정의된 모듈의 경로를 찾음 (상속 시 해당 클래스 위치 기준)
            module = sys.modules[self.__class__.__module__]
            if hasattr(module, "__file__") and module.__file__:
                self.template_path = Path(module.__file__).parent / "template.md"
            else:
                self.template_path = Path(__file__).parent / "template.md"

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found at: {self.template_path}")

        # 토큰 인코더 초기화 (GPT-4용)
        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # tiktoken이 없거나 모델을 찾을 수 없는 경우 간단한 추정 사용
            logger.warning(
                "tiktoken을 사용할 수 없습니다. 간단한 토큰 추정을 사용합니다."
            )
            self.token_encoder = None

    def calculate_token_size(self, text: str) -> int:
        """
        텍스트의 토큰 크기를 계산합니다.

        Args:
            text: 토큰 크기를 계산할 텍스트

        Returns:
            int: 토큰 크기
        """
        if self.token_encoder:
            try:
                tokens = self.token_encoder.encode(text)
                return len(tokens)
            except Exception as e:
                logger.warning(f"토큰 인코딩 실패, 추정값 사용: {e}")

        # 간단한 추정: 대략 1 토큰 = 4 문자
        return len(text) // 4

    def create_prompt(self, input_data: DiffGeneratorInput) -> str:
        """
        입력 데이터를 사용하여 프롬프트를 생성합니다.

        Args:
            input_data: Diff 생성 입력

        Returns:
            str: 생성된 프롬프트
        """
        # 배치 프롬프트 생성 (절대 경로 포함)
        batch_variables = {
            "table_info": input_data.table_info,
            "layer_name": input_data.layer_name,
            "source_files": "\n\n".join(
                [
                    f"=== File Path (Absolute): {snippet.path} ===\n{snippet.content}"
                    for snippet in input_data.code_snippets
                ]
            ),
            "file_count": len(input_data.code_snippets),
            **(input_data.extra_variables or {}),
        }

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        return render_template(template_str, batch_variables)

    def generate(self, input_data: DiffGeneratorInput) -> DiffGeneratorOutput:
        """
        입력 데이터를 바탕으로 Diff를 생성합니다.

        Args:
            input_data: Diff 생성 입력

        Returns:
            DiffGeneratorOutput: LLM 응답 (Diff 포함)
        """
        prompt = self.create_prompt(input_data)

        # 캐시 확인
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._prompt_cache:
            logger.debug(f"캐시에서 응답을 가져왔습니다: {cache_key[:50]}...")
            return self._prompt_cache[cache_key]

        # LLM 호출
        try:
            response = self.llm_provider.call(prompt)

            # DiffGeneratorOutput 객체 생성
            output = DiffGeneratorOutput(
                content=response.get("content", ""),
                tokens_used=response.get("tokens_used", 0),
            )

            # 캐시에 저장
            self._prompt_cache[cache_key] = output

            return output
        except Exception as e:
            logger.error(f"Diff 생성 중 오류 발생: {e}")
            raise

    def _get_cache_key(self, prompt: str) -> str:
        """프롬프트의 캐시 키를 생성합니다."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def clear_cache(self):
        """캐시를 비웁니다."""
        self._prompt_cache.clear()
