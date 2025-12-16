"""
Prompt Template Manager

프롬프트 템플릿을 로드하고 변수를 치환하며 토큰 크기를 계산하는 클래스입니다.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import tiktoken
import yaml

logger = logging.getLogger(__name__)


class PromptTemplateError(Exception):
    """프롬프트 템플릿 관련 오류"""

    pass


class PromptTemplateManager:
    """
    프롬프트 템플릿 관리 클래스

    템플릿 파일을 로드하고, 변수를 치환하며, 토큰 크기를 계산합니다.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        PromptTemplateManager 초기화

        Args:
            template_dir: 템플릿 파일이 있는 디렉토리 (선택적)
        """
        if template_dir is None:
            # 기본 템플릿 디렉토리 (현재 파일 기준)
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)

        if not self.template_dir.exists():
            raise PromptTemplateError(
                f"템플릿 디렉토리가 존재하지 않습니다: {self.template_dir}"
            )

        # 템플릿 캐시
        self._template_cache: Dict[str, Dict[str, Any]] = {}

        # 토큰 인코더 초기화 (GPT-4용)
        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # tiktoken이 없거나 모델을 찾을 수 없는 경우 간단한 추정 사용
            logger.warning(
                "tiktoken을 사용할 수 없습니다. 간단한 토큰 추정을 사용합니다."
            )
            self.token_encoder = None

    def load_template(self, template_type: str = "default") -> Dict[str, Any]:
        """
        템플릿 파일을 로드합니다.

        Args:
            template_type: 템플릿 타입 ("default" 또는 기존 타입들도 지원)

        Returns:
            Dict[str, Any]: 로드된 템플릿 딕셔너리

        Raises:
            PromptTemplateError: 템플릿 파일을 찾을 수 없거나 로드 실패 시
        """
        # 캐시 확인
        if template_type in self._template_cache:
            return self._template_cache[template_type]

        # 템플릿 파일 경로 (통합 템플릿 사용)
        if template_type == "default":
            template_file = self.template_dir / "prompt_template.yaml"
        else:
            # 하위 호환성을 위해 기존 파일명도 지원
            template_file = self.template_dir / f"prompt_template_{template_type}.yaml"

        if not template_file.exists():
            raise PromptTemplateError(
                f"템플릿 파일을 찾을 수 없습니다: {template_file}"
            )

        try:
            with open(template_file, "r", encoding="utf-8") as f:
                template = yaml.safe_load(f)

            # 캐시에 저장
            self._template_cache[template_type] = template

            logger.debug(f"템플릿 로드 완료: {template_type}")
            return template

        except Exception as e:
            raise PromptTemplateError(f"템플릿 파일 로드 실패: {e}")

    def render_template(
        self, template: Dict[str, Any], variables: Dict[str, Any]
    ) -> str:
        """
        템플릿에 변수를 치환하여 최종 프롬프트를 생성합니다.

        Args:
            template: 템플릿 딕셔너리
            variables: 치환할 변수 딕셔너리
                - table_info: 테이블/칼럼 정보 (JSON 문자열)
                - source_files: 소스 파일 목록 및 내용 (문자열)
                - layer_name: 레이어명
                - file_count: 파일 개수

        Returns:
            str: 최종 프롬프트 문자열
        """
        # 템플릿을 문자열로 변환
        prompt_parts = []

        # System Instruction
        if "system_instruction" in template:
            prompt_parts.append("## System Instruction\n")
            prompt_parts.append(template["system_instruction"])
            prompt_parts.append("\n")

        # Coding Rules
        if "coding_rules" in template:
            prompt_parts.append("\n## Coding Rules\n")
            prompt_parts.append(template["coding_rules"])
            prompt_parts.append("\n")

        # Few-shot Examples
        if "few_shot_examples" in template:
            prompt_parts.append("\n## Few-shot Examples\n")
            for i, example in enumerate(template["few_shot_examples"], 1):
                prompt_parts.append(
                    f"\n### Example {i}: {example.get('example_type', '')}\n"
                )
                prompt_parts.append(
                    f"**Before:**\n```java\n{example.get('before', '')}\n```\n"
                )
                prompt_parts.append(
                    f"**After:**\n```java\n{example.get('after', '')}\n```\n"
                )
                if "explanation" in example:
                    prompt_parts.append(f"**Explanation:** {example['explanation']}\n")

        # Table Column Info
        if "table_column_info" in template:
            prompt_parts.append("\n## Table Column Information\n")
            if variables.get("table_column_info", None) is not None:
                table_info = template["table_column_info"].format(**variables)
                prompt_parts.append(table_info)
            else:
                prompt_parts.append(template["table_column_info"])
                prompt_parts.append("\n")

        # Source Files
        if "source_files" in template:
            prompt_parts.append("\n## Source Files to Modify\n")
            if variables.get("source_files", None) is not None:
                source_files = template["source_files"].format(**variables)
                prompt_parts.append(source_files)
            else:
                prompt_parts.append(template["source_files"])
                prompt_parts.append("\n")

        # Layer Name
        if "layer_name" in template:
            prompt_parts.append("\n## Current Layer\n")
            if variables.get("layer_name", None) is not None:
                layer_name = template["layer_name"].format(**variables)
                prompt_parts.append(f"\n## Current Layer: {layer_name}\n")
            else:
                prompt_parts.append(template["layer_name"])
                prompt_parts.append("\n")

        # File Count
        if "file_count" in template:
            prompt_parts.append("\n## File Count\n")
            if variables.get("file_count", None) is not None:
                file_count = template["file_count"].format(**variables)
                prompt_parts.append(f"\n## File Count: {file_count}\n")
            else:
                prompt_parts.append(template["file_count"])
                prompt_parts.append("\n")

        # Output Format
        if "output_format" in template:
            prompt_parts.append("\n## Output Format\n")
            prompt_parts.append(template["output_format"])
            prompt_parts.append("\n")

        # Warnings
        if "warnings" in template:
            prompt_parts.append("\n## Warnings\n")
            prompt_parts.append(template["warnings"])
            prompt_parts.append("\n")

        # 최종 프롬프트 생성
        final_prompt = "\n".join(prompt_parts)

        # # 변수 치환 (남은 변수들)
        # try:
        #     final_prompt = final_prompt.format(**variables)
        # except KeyError as e:
        #     logger.warning(f"템플릿 변수 치환 중 누락된 변수: {e}")

        return final_prompt

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

    def clear_cache(self):
        """템플릿 캐시를 비웁니다."""
        self._template_cache.clear()
        logger.debug("템플릿 캐시가 비워졌습니다.")
