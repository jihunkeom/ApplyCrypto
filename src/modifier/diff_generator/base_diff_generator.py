import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tiktoken
from jinja2 import Template

from config.config_manager import Configuration
from models.diff_generator import DiffGeneratorInput, DiffGeneratorOutput
from modifier.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class DiffGeneratorError(Exception):
    """Diff Generator 관련 오류"""

    pass


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
        config: Optional[Configuration] = None,
    ):
        """
        BaseDiffGenerator 초기화

        Args:
            llm_provider: LLM 프로바이더
            prompt_cache: 프롬프트 캐시 저장소 (선택적)
            template_path: 템플릿 파일 경로
            config: 설정 객체 (선택적)
        """
        self.llm_provider = llm_provider
        self._prompt_cache = prompt_cache if prompt_cache is not None else {}
        self.config = config

        if template_path:
            self.template_path = Path(template_path)
        else:
            # 클래스가 정의된 모듈의 경로를 찾음 (상속 시 해당 클래스 위치 기준)
            module = sys.modules[self.__class__.__module__]
            if hasattr(module, "__file__") and module.__file__:
                template_dir = Path(module.__file__).parent
            else:
                template_dir = Path(__file__).parent

            # generate_full_source 설정에 따라 템플릿 파일 선택
            if config and config.generate_full_source:
                template_filename = "template_full.md"
            else:
                template_filename = "template.md"

            self.template_path = template_dir / template_filename

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

    def _parse_llm_response(
        self, response: Union[Dict[str, Any], DiffGeneratorOutput]
    ) -> List[Dict[str, Any]]:
        """
        LLM 응답을 파싱하여 수정 정보를 추출합니다.

        Args:
            response: LLM 응답 (Dictionary or DiffGeneratorOutput)

        Returns:
            List[Dict[str, Any]]: 수정 정보 리스트
                - file_path: 파일 경로
                - unified_diff: Unified Diff 형식의 수정 내용

        Raises:
            DiffGeneratorError: 파싱 실패 시
        """
        try:
            # 응답에서 content 추출
            if isinstance(response, DiffGeneratorOutput):
                content = response.content
            else:
                content = response.get("content", "")

            if not content:
                raise DiffGeneratorError("LLM 응답에 content가 없습니다.")

            # JSON 파싱 시도
            # content가 JSON 코드 블록으로 감싸져 있을 수 있음
            content = content.strip()
            if content.startswith("```"):
                # 코드 블록 제거
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            elif content.startswith("```json"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            # JSON 파싱
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시, modifications 키워드로 찾기 시도
                if "modifications" in content:
                    # modifications 부분만 추출
                    start_idx = content.find('"modifications"')
                    if start_idx != -1:
                        # JSON 객체 시작 찾기
                        brace_start = content.rfind("{", 0, start_idx)
                        if brace_start != -1:
                            # JSON 객체 끝 찾기
                            brace_count = 0
                            for i in range(brace_start, len(content)):
                                if content[i] == "{":
                                    brace_count += 1
                                elif content[i] == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = content[brace_start : i + 1]
                                        data = json.loads(json_str)
                                        break
                            else:
                                raise DiffGeneratorError(
                                    "JSON 파싱 실패: 올바른 JSON 형식이 아닙니다."
                                )
                        else:
                            raise DiffGeneratorError(
                                "JSON 파싱 실패: modifications를 찾을 수 없습니다."
                            )
                    else:
                        raise DiffGeneratorError(
                            "JSON 파싱 실패: modifications 키를 찾을 수 없습니다."
                        )
                else:
                    raise DiffGeneratorError(
                        "JSON 파싱 실패: 올바른 JSON 형식이 아닙니다."
                    )

            # modifications 추출
            modifications = data.get("modifications", [])
            if not modifications:
                raise DiffGeneratorError("LLM 응답에 modifications가 없습니다.")

            # 검증
            for mod in modifications:
                if "file_path" not in mod:
                    raise DiffGeneratorError("수정 정보에 file_path가 없습니다.")
                if "reason" not in mod:
                    raise DiffGeneratorError("수정 정보에 reason가 없습니다.")
                if "unified_diff" not in mod:
                    raise DiffGeneratorError("수정 정보에 unified_diff가 없습니다.")

            logger.info(f"{len(modifications)}개 파일 수정 정보를 파싱했습니다.")
            return modifications

        except Exception as e:
            # logger.error(f"LLM 응답 파싱 실패: {e}") # generate 에서 호출될 때 잡음 발생 가능
            raise DiffGeneratorError(f"LLM 응답 파싱 실패: {e}")

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

            # 파싱 시도 (실패 시 무시 - CallChain 등 다른 포맷일 수 있음)
            try:
                output.parsed_out = self._parse_llm_response(output)
            except Exception as e:
                logger.debug(f"자동 Diff 파싱 실패 (포맷 불일치 가능성): {e}")
                output.parsed_out = None

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
