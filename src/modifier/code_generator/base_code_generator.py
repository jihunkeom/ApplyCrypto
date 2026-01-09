import hashlib
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tiktoken
from jinja2 import Template

from config.config_manager import Configuration
from models.code_generator import CodeGeneratorInput, CodeGeneratorOutput
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo
from modifier.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class CodeGeneratorError(Exception):
    """Code Generator 관련 오류"""

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


class BaseCodeGenerator(ABC):
    """Code 생성기 베이스 클래스"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_cache: Dict[str, Dict[str, Any]] = None,
        template_path: Optional[Path] = None,
        config: Optional[Configuration] = None,
    ):
        """
        BaseCodeGenerator 초기화

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

    def create_prompt(self, input_data: CodeGeneratorInput) -> str:
        """
        입력 데이터를 사용하여 프롬프트를 생성합니다.

        Args:
            input_data: Code 생성 입력

        Returns:
            str: 생성된 프롬프트
        """
        source_files_str = "\n\n".join(
            [
                f"=== File: {Path(snippet.path).name} ===\n{snippet.content}"
                for snippet in input_data.code_snippets
            ]
        )

        # 배치 프롬프트 생성
        batch_variables = {
            "table_info": input_data.table_info,
            "layer_name": input_data.layer_name,
            "source_files": source_files_str,
            "file_count": len(input_data.code_snippets),
            **(input_data.extra_variables or {}),
        }

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        return render_template(template_str, batch_variables)

    def parse_llm_response(
        self, response: Union[Dict[str, Any], CodeGeneratorOutput]
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
            if isinstance(response, CodeGeneratorOutput):
                content = response.content
                file_mapping = response.file_mapping or {}
            else:
                content = response.get("content", "")
                file_mapping = response.get("file_mapping", {})

            if not content:
                raise Exception("LLM응답에 content가 없습니다.")
            
            content = content.strip()

            # 구분자 기반 형식 시도
            if "=====FILE=====" in content:
                return self._parse_delimited_format(content, file_mapping)
             
        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}")
            raise Exception(f"LLM 응답 파싱 실패: {e}")

    def _parse_delimited_format(self, content:str, file_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        구분자 기반 형식의 LLM 응답을 파싱합니다.

        형식:
            ======FILE======
            EmployeeService.java
            ======REASON======
            Added encryption for name field
            ======MODIFIED_CODE======
            package com.example;
            ...
            ======END======

        Args:
            content: LLM 응답 내용
            file_mapping: 파일명 -> 절대 경로 매핑

        Returns:
            List[Dict[str, Any]]: 수정 정보 리스트
        """
        modifications = []

        # ======END====== 기준으로 블록 분리
        blocks = content.split("======END======")

        for block in blocks:
            block = block.strip()
            if not block or "======FILE======" not in block:
                continue

            try:
                # 각 섹션 추출
                file_name = self._extract_section(
                    block, "======FILE======", "======REASON======"
                )
                reason = self._extract_section(
                    block, "======REASON======", "======MODIFIED_CODE======"
                )
                modified_code = self._extract_section(
                    block, "======MODIFIED_CODE======", "======END======"
                )

                # 파일명 정리
                file_name = file_name.strip()

                # 파일명을 절대 경로로 변환
                if file_name in file_mapping:
                    file_path = file_mapping[file_name]
                else:
                    # 매핑에 없으면 파일명 그대로 사용 (절대 경로일 수도 있음)
                    file_path = file_name
                    logger.warning(
                        f"파일 매핑에서 찾을 수 없음: {file_name}. 원본 값 사용."
                    )

                modifications.append({
                    "file_path": file_path,
                    "reason": reason.strip(),
                    "unified_diff": modified_code.strip(),
                })

            except Exception as e:
                logger.warning(f"블록 파싱 중 오류 (건너뜀): {e}")
                continue

        if not modifications:
            raise Exception(
                "구분자 형식 파싱 실패: 유효한 수정 블록을 찾을 수 없습니다."
            )

        logger.info(f"{len(modifications)}개 파일 수정 정보를 파싱했습니다 (구분자 형식).")
        return modifications

    def _extract_section(
        self, block: str, start_delimiter: str, end_delimiter: str
    ) -> str:
        """
        블록에서 특정 섹션을 추출합니다.

        Args:
            block: 전체 블록 문자열
            start_delimiter: 시작 구분자
            end_delimiter: 종료 구분자

        Returns:
            str: 추출된 섹션 내용
        """
        start_idx = block.find(start_delimiter)
        if start_idx == -1:
            return ""

        start_idx += len(start_delimiter)

        # end_delimiter가 블록에 없으면 끝까지
        end_idx = block.find(end_delimiter, start_idx)
        if end_idx == -1:
            return block[start_idx:].strip()

        return block[start_idx:end_idx].strip()

    @abstractmethod
    def generate(self, input_data: CodeGeneratorInput) -> CodeGeneratorOutput:
        """
        입력 데이터를 바탕으로 Code를 생성합니다.

        Args:
            input_data: Code 생성 입력

        Returns:
            CodeGeneratorOutput: LLM 응답 (Code 포함)
        """
        pass

    @abstractmethod
    def generate_modification_plans(
        self, table_access_info: TableAccessInfo
    ) -> List[ModificationPlan]:
        """
        수정 계획을 생성합니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        pass

    def _get_cache_key(self, prompt: str) -> str:
        """프롬프트의 캐시 키를 생성합니다."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def clear_cache(self):
        """캐시를 비웁니다."""
        self._prompt_cache.clear()

