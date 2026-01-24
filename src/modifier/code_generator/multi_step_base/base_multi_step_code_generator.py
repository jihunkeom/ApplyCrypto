"""
Base Multi-Step Code Generator

다단계 LLM 협업 전략을 사용하는 CodeGenerator의 공통 기반 클래스입니다.
TwoStepCodeGenerator와 ThreeStepCodeGenerator가 이 클래스를 상속합니다.

공통 기능:
- 템플릿 로딩 및 렌더링
- 파일 읽기 (일반/인덱스 형식)
- JSON 추출 및 복구
- 파일 경로 해결 (인덱스, 정확, 대소문자무시, 시그니처, Fuzzy 매칭)
- 세션 디렉토리 및 결과 저장
- Execution phase 파싱
"""

import json
import logging
import re
from abc import abstractmethod
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from jinja2 import Template

from config.config_manager import (
    Configuration,
    MultiStepExecutionConfig,
    ThreeStepConfig,
    TwoStepConfig,
)
from models.code_generator import CodeGeneratorInput, CodeGeneratorOutput
from models.modification_context import ModificationContext
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo
from modifier.llm.llm_provider import LLMProvider

from ..base_code_generator import BaseCodeGenerator

logger = logging.getLogger("applycrypto")


# 토큰 계산을 위한 기본 템플릿 (BaseContextGenerator.create_batches()에서 사용)
_TOKEN_CALCULATION_TEMPLATE = """
## Table Info
{{ table_info }}

## Source Files
{{ source_files }}
"""


class BaseMultiStepCodeGenerator(BaseCodeGenerator):
    """
    다단계 LLM 협업 전략을 사용하는 CodeGenerator의 공통 기반 클래스.

    TwoStepCodeGenerator와 ThreeStepCodeGenerator의 공통 로직을 제공합니다.

    상속 클래스에서 구현해야 할 추상 메서드:
    - _get_output_subdir_name(): 출력 디렉토리 하위 폴더명
    - _get_step_config(): TwoStepConfig 또는 ThreeStepConfig 반환
    - _get_execution_provider(): Execution LLM provider 반환
    - _get_execution_template_path(): Execution 템플릿 경로 반환
    - _execute_planning_phases(): Planning 단계들 실행
    - _get_planning_reasons(): Planning 결과에서 reason 추출
    """

    def __init__(self, config: Configuration):
        """
        BaseMultiStepCodeGenerator 초기화

        Args:
            config: 설정 객체
        """
        self.config = config
        self._prompt_cache: Dict[str, Any] = {}
        self._session_timestamp: Optional[str] = None  # 세션 공유 timestamp

        # 토큰 인코더 초기화
        self._init_token_encoder()

    def _init_token_encoder(self) -> None:
        """토큰 인코더를 초기화합니다."""
        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            logger.warning(
                "tiktoken을 사용할 수 없습니다. 간단한 토큰 추정을 사용합니다."
            )
            self.token_encoder = None

    def _init_output_directory(self) -> None:
        """
        출력 디렉토리를 초기화합니다.

        Note: 이 메서드는 서브클래스의 __init__에서 _get_output_subdir_name() 구현 후 호출해야 합니다.
        """
        self.output_dir = (
            Path(self.config.target_project)
            / ".applycrypto"
            / self._get_output_subdir_name()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 프롬프트 저장 디렉토리 초기화 (ApplyCrypto 실행 디렉토리 기준)
        self.prompt_results_dir = Path("./prompt_results")
        self.prompt_results_dir.mkdir(parents=True, exist_ok=True)

    # ========== 추상 메서드 (서브클래스에서 구현) ==========

    @abstractmethod
    def _get_output_subdir_name(self) -> str:
        """
        출력 디렉토리 하위 폴더명을 반환합니다.

        Returns:
            str: 출력 디렉토리명 (예: 'two_step_results', 'three_step_results')
        """
        pass

    @abstractmethod
    def _get_step_config(self) -> Union[TwoStepConfig, ThreeStepConfig]:
        """
        현재 Step 설정을 반환합니다.

        Returns:
            Union[TwoStepConfig, ThreeStepConfig]: Step 설정 객체
        """
        pass

    @abstractmethod
    def _get_execution_provider(self) -> LLMProvider:
        """
        Execution phase에서 사용할 LLM provider를 반환합니다.

        Returns:
            LLMProvider: Execution LLM provider
        """
        pass

    @abstractmethod
    def _get_execution_template_path(self) -> Path:
        """
        Execution 템플릿 파일 경로를 반환합니다.

        Returns:
            Path: Execution 템플릿 경로
        """
        pass

    @abstractmethod
    def _execute_planning_phases(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Planning 단계들을 실행합니다.

        TwoStep: Phase 1 (Planning)
        ThreeStep: Phase 1 (DataMapping) + Phase 2 (Planning)

        Args:
            session_dir: 세션 디렉토리 경로
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보

        Returns:
            Tuple[Dict[str, Any], int]: (planning_result, total_tokens_used)
        """
        pass

    @abstractmethod
    def _get_planning_reasons(
        self, planning_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Planning 결과에서 파일명 -> reason 매핑을 추출합니다.

        Args:
            planning_result: Planning 단계의 결과

        Returns:
            Dict[str, str]: 파일명 -> reason 매핑
        """
        pass

    @abstractmethod
    def _get_step_name(self) -> str:
        """
        현재 Step 이름을 반환합니다 (로깅용).

        Returns:
            str: Step 이름 (예: '2-Step', '3-Step')
        """
        pass

    @abstractmethod
    def _get_last_planning_step_number(self) -> int:
        """
        마지막 Planning 단계의 번호를 반환합니다.

        resume_from_plan 시 이 번호의 결과 파일을 로드합니다.

        Returns:
            int: 마지막 Planning 단계 번호 (TwoStep: 1, ThreeStep: 2)
        """
        pass

    @abstractmethod
    def _get_last_planning_phase_name(self) -> str:
        """
        마지막 Planning 단계의 이름을 반환합니다.

        Returns:
            str: 마지막 Planning 단계 이름 (TwoStep: 'planning', ThreeStep: 'planning')
        """
        pass

    # ========== Execution Options 헬퍼 메서드 ==========

    def _get_execution_options(self) -> Optional[MultiStepExecutionConfig]:
        """
        현재 Step 설정에서 execution_options를 가져옵니다.

        Returns:
            Optional[MultiStepExecutionConfig]: 실행 옵션 또는 None
        """
        step_config = self._get_step_config()
        return getattr(step_config, "execution_options", None)

    def _get_mode(self) -> str:
        """
        실행 모드를 가져옵니다.

        Returns:
            str: 실행 모드 ("full", "plan_only", "execution_only")
        """
        options = self._get_execution_options()
        if options is None:
            return "full"
        return options.mode

    def _is_plan_only_mode(self) -> bool:
        """
        plan_only 모드인지 확인합니다.

        Returns:
            bool: plan_only 모드이면 True
        """
        return self._get_mode() == "plan_only"

    def _is_execution_only_mode(self) -> bool:
        """
        execution_only 모드인지 확인합니다.

        Returns:
            bool: execution_only 모드이면 True
        """
        return self._get_mode() == "execution_only"

    def _get_plan_timestamp(self) -> Optional[str]:
        """
        execution_only 모드에서 사용할 plan_timestamp를 가져옵니다.

        Returns:
            Optional[str]: 이전 Planning 결과의 timestamp 또는 None
        """
        options = self._get_execution_options()
        if options is None:
            return None
        return options.plan_timestamp

    def _get_plan_session_dir(
        self, modification_context: ModificationContext
    ) -> Optional[Path]:
        """
        execution_only 모드에서 현재 컨텍스트에 맞는 plan 세션 디렉토리를 찾습니다.

        디렉토리 구조: {output_dir}/{timestamp}/{table_name}/{first_file}/

        Args:
            modification_context: 수정 컨텍스트

        Returns:
            Optional[Path]: 찾은 세션 디렉토리 경로, 없으면 None
        """
        plan_timestamp = self._get_plan_timestamp()
        if not plan_timestamp:
            return None

        # 컨텍스트 정보 추출
        table_name = modification_context.table_name
        first_file = Path(modification_context.file_paths[0]).stem if modification_context.file_paths else None

        if not first_file:
            return None

        # 파일 시스템에 안전한 이름으로 변환
        safe_table_name = re.sub(r"[^\w\-]", "_", table_name)
        safe_first_file = re.sub(r"[^\w\-]", "_", first_file)

        # 세션 디렉토리 경로 구성: {output_dir}/{timestamp}/{table_name}/{first_file}/
        session_dir = self.output_dir / plan_timestamp / safe_table_name / safe_first_file

        if session_dir.exists():
            return session_dir

        logger.warning(
            f"Plan 세션 디렉토리를 찾을 수 없습니다: {session_dir}\n"
            f"  - timestamp: {plan_timestamp}\n"
            f"  - table_name: {table_name}\n"
            f"  - first_file: {first_file}"
        )
        return None

    def _load_previous_planning_result(
        self, session_dir: Path
    ) -> Dict[str, Any]:
        """
        이전 Planning 결과를 로드합니다.

        Args:
            session_dir: 세션 디렉토리 경로

        Returns:
            Dict[str, Any]: Planning 결과

        Raises:
            FileNotFoundError: Planning 결과 파일을 찾을 수 없는 경우
            ValueError: Planning 결과 파일 형식이 올바르지 않은 경우
        """
        if not session_dir.exists():
            raise FileNotFoundError(f"세션 디렉토리를 찾을 수 없습니다: {session_dir}")

        # 마지막 Planning 단계의 결과 파일 찾기
        step_number = self._get_last_planning_step_number()
        phase_name = self._get_last_planning_phase_name()
        result_file = session_dir / f"step{step_number}_{phase_name}.json"

        if not result_file.exists():
            raise FileNotFoundError(
                f"Planning 결과 파일을 찾을 수 없습니다: {result_file}\n"
                f"execution_only 모드는 이전 plan_only 실행의 결과가 필요합니다."
            )

        with open(result_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        planning_result = saved_data.get("result", {})
        if not planning_result:
            raise ValueError(
                f"Planning 결과 파일에 'result' 필드가 없습니다: {result_file}"
            )

        logger.info(f"이전 Planning 결과 로드됨: {result_file}")
        return planning_result

    def _create_plan_only_results(
        self,
        modification_context: ModificationContext,
        planning_result: Dict[str, Any],
        session_dir: Path,
        tokens_used: int,
    ) -> List[ModificationPlan]:
        """
        plan_only 모드일 때 결과를 생성합니다.

        Execution을 건너뛰고 Planning 결과만 포함한 ModificationPlan 리스트를 반환합니다.

        Args:
            modification_context: 수정 컨텍스트
            planning_result: Planning 결과
            session_dir: 세션 디렉토리 경로
            tokens_used: 사용된 토큰 수

        Returns:
            List[ModificationPlan]: plan_only 상태의 수정 계획 리스트
        """
        plans = []
        layer_name = modification_context.layer
        planning_reasons = self._get_planning_reasons(planning_result)

        for file_path in modification_context.file_paths:
            file_name = Path(file_path).name
            reason = planning_reasons.get(file_name, "")

            plan = ModificationPlan(
                file_path=file_path,
                layer_name=layer_name,
                modification_type="encryption",
                modified_code="",  # plan_only에서는 코드 생성 안함
                reason=reason if reason else "plan_only 모드 - Execution 대기 중",
                tokens_used=tokens_used,
                status="plan_only",  # 특별 상태: Planning 완료, Execution 대기
            )
            plans.append(plan)

        # timestamp 추출 (session_dir에서 상위 2단계 폴더명)
        # 구조: {output_dir}/{timestamp}/{table_name}/{first_file}/
        timestamp = session_dir.parent.parent.name

        logger.info("=" * 60)
        logger.info(f"{self._get_step_name()} Plan-Only 모드 완료")
        logger.info(f"Planning 결과가 저장되었습니다: {session_dir}")
        logger.info(f"총 토큰 사용량: {tokens_used}")
        logger.info(f"대상 파일 수: {len(plans)}")
        logger.info("")
        logger.info("Execution을 실행하려면 다음 설정을 사용하세요:")
        logger.info('  "mode": "execution_only"')
        logger.info(f'  "plan_timestamp": "{timestamp}"')
        logger.info("=" * 60)

        return plans

    # ========== 템플릿 및 렌더링 유틸리티 ==========

    def _load_template(self, template_path: Path) -> str:
        """템플릿 파일을 로드합니다."""
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    def _render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """Jinja2 템플릿을 렌더링합니다."""
        template = Template(template_str)
        return template.render(**variables)

    # ========== 파일 읽기 유틸리티 ==========

    def _read_file_contents(self, file_paths: List[str]) -> str:
        """파일들의 내용을 읽어서 문자열로 반환합니다."""
        snippets = []
        for file_path in file_paths:
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    with open(path_obj, "r", encoding="utf-8") as f:
                        content = f.read()
                    snippets.append(f"=== File: {path_obj.name} ===\n{content}")
                else:
                    logger.warning(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Failed to read file: {file_path} - {e}")
        return "\n\n".join(snippets)

    def _read_file_contents_indexed(
        self, file_paths: List[str]
    ) -> Tuple[str, Dict[int, str], Dict[str, str]]:
        """
        파일들의 내용을 인덱스와 함께 읽어서 반환합니다.

        Args:
            file_paths: 파일 경로 리스트

        Returns:
            Tuple[str, Dict[int, str], Dict[str, str]]:
                - 인덱스가 포함된 파일 내용 문자열
                - 인덱스 -> 파일 경로 매핑
                - 파일 경로 -> 파일 내용 매핑 (시그니처 매칭용)
        """
        snippets = []
        index_to_path: Dict[int, str] = {}
        path_to_content: Dict[str, str] = {}

        for idx, file_path in enumerate(file_paths, start=1):
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    with open(path_obj, "r", encoding="utf-8") as f:
                        content = f.read()
                    # 인덱스 형식으로 파일 헤더 생성
                    snippets.append(
                        f"[FILE_{idx}] {path_obj.name}\n"
                        f"=== Content ===\n{content}"
                    )
                    index_to_path[idx] = file_path
                    path_to_content[file_path] = content
                else:
                    logger.warning(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Failed to read file: {file_path} - {e}")

        return "\n\n".join(snippets), index_to_path, path_to_content

    # ========== 코드 시그니처 및 파일 매칭 유틸리티 ==========

    def _extract_code_signature(
        self, code: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Java 코드에서 패키지명과 클래스명을 추출합니다.

        Args:
            code: Java 소스 코드

        Returns:
            Tuple[Optional[str], Optional[str]]: (패키지명, 클래스명)
        """
        package_name = None
        class_name = None

        # 패키지명 추출
        package_match = re.search(r"package\s+([\w.]+)\s*;", code)
        if package_match:
            package_name = package_match.group(1)

        # 클래스명 추출 (class, interface, enum)
        class_match = re.search(
            r"(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
            code,
        )
        if class_match:
            class_name = class_match.group(1)

        return package_name, class_name

    def _match_by_code_signature(
        self, modified_code: str, path_to_content: Dict[str, str]
    ) -> Optional[str]:
        """
        수정된 코드의 시그니처로 원본 파일을 찾습니다.

        Args:
            modified_code: LLM이 생성한 수정된 코드
            path_to_content: 파일 경로 -> 원본 내용 매핑

        Returns:
            Optional[str]: 매칭된 파일 경로, 없으면 None
        """
        mod_package, mod_class = self._extract_code_signature(modified_code)

        if not mod_class:
            # 클래스명을 추출할 수 없으면 매칭 불가
            return None

        for file_path, original_content in path_to_content.items():
            orig_package, orig_class = self._extract_code_signature(original_content)

            # 클래스명이 일치하고, 패키지명도 일치하거나 둘 다 없는 경우
            if orig_class == mod_class:
                if mod_package == orig_package:
                    logger.info(
                        f"코드 시그니처 매칭 성공: {mod_package}.{mod_class} -> {file_path}"
                    )
                    return file_path
                elif mod_package is None or orig_package is None:
                    # 패키지가 없는 경우에도 클래스명만으로 매칭 시도
                    logger.info(
                        f"코드 시그니처 매칭 (클래스명만): {mod_class} -> {file_path}"
                    )
                    return file_path

        return None

    def _fuzzy_match_filename(
        self, llm_filename: str, file_mapping: Dict[str, str]
    ) -> Optional[str]:
        """
        Fuzzy matching으로 유사한 파일명을 찾습니다.

        Args:
            llm_filename: LLM이 반환한 파일명
            file_mapping: 파일명 -> 파일 경로 매핑

        Returns:
            Optional[str]: 매칭된 파일 경로, 없으면 None
        """
        best_match = None
        best_ratio = 0.0
        threshold = 0.7  # 70% 이상 유사해야 매칭

        llm_filename_lower = llm_filename.lower()

        for filename, filepath in file_mapping.items():
            ratio = SequenceMatcher(
                None, llm_filename_lower, filename.lower()
            ).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = filepath

        if best_match:
            logger.info(
                f"Fuzzy 매칭 성공: {llm_filename} -> {Path(best_match).name} "
                f"(유사도: {best_ratio:.1%})"
            )

        return best_match

    def _resolve_file_path(
        self,
        llm_file_identifier: str,
        modified_code: str,
        index_to_path: Dict[int, str],
        file_mapping: Dict[str, str],
        path_to_content: Dict[str, str],
    ) -> Tuple[Optional[str], str]:
        """
        여러 전략을 순차적으로 시도하여 파일 경로를 해결합니다.

        Args:
            llm_file_identifier: LLM이 반환한 파일 식별자 (인덱스 또는 파일명)
            modified_code: LLM이 생성한 수정된 코드
            index_to_path: 인덱스 -> 파일 경로 매핑
            file_mapping: 파일명 -> 파일 경로 매핑
            path_to_content: 파일 경로 -> 원본 내용 매핑

        Returns:
            Tuple[Optional[str], str]: (해결된 파일 경로, 매칭 방법)
        """
        # 1. 인덱스 매칭 시도 (FILE_1, FILE_2, ...)
        index_match = re.match(
            r"FILE_(\d+)", llm_file_identifier.strip(), re.IGNORECASE
        )
        if index_match:
            idx = int(index_match.group(1))
            if idx in index_to_path:
                logger.debug(f"인덱스 매칭 성공: FILE_{idx} -> {index_to_path[idx]}")
                return index_to_path[idx], "index"

        # 2. 정확한 파일명 매칭 시도
        clean_name = llm_file_identifier.strip()
        # 경로가 포함된 경우 파일명만 추출
        if "/" in clean_name or "\\" in clean_name:
            clean_name = Path(clean_name).name

        if clean_name in file_mapping:
            logger.debug(f"정확한 파일명 매칭 성공: {clean_name}")
            return file_mapping[clean_name], "exact"

        # 3. 대소문자 무시 매칭
        lower_mapping = {k.lower(): v for k, v in file_mapping.items()}
        if clean_name.lower() in lower_mapping:
            logger.debug(f"대소문자 무시 매칭 성공: {clean_name}")
            return lower_mapping[clean_name.lower()], "case_insensitive"

        # 4. 코드 시그니처 매칭 (가장 신뢰도 높음)
        if modified_code and modified_code.strip():
            signature_match = self._match_by_code_signature(
                modified_code, path_to_content
            )
            if signature_match:
                return signature_match, "signature"

        # 5. Fuzzy 매칭 (마지막 수단)
        fuzzy_match = self._fuzzy_match_filename(clean_name, file_mapping)
        if fuzzy_match:
            return fuzzy_match, "fuzzy"

        # 매칭 실패
        logger.warning(
            f"파일 경로 해결 실패: {llm_file_identifier}. "
            f"시도한 방법: index, exact, case_insensitive, signature, fuzzy"
        )
        return None, "failed"

    # ========== 섹션 추출 유틸리티 ==========

    def _extract_section(
        self, content: str, start_marker: str, end_marker: str
    ) -> str:
        """
        콘텐츠에서 두 마커 사이의 섹션을 추출합니다.

        Args:
            content: 전체 콘텐츠
            start_marker: 시작 마커
            end_marker: 종료 마커

        Returns:
            str: 추출된 섹션 (마커 제외)
        """
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return ""

        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)

        if end_idx == -1:
            # 종료 마커가 없으면 끝까지 추출
            return content[start_idx:].strip()

        return content[start_idx:end_idx].strip()

    # ========== JSON 추출 및 복구 유틸리티 ==========

    def _extract_json_from_content(self, content: str) -> str:
        """
        LLM 응답에서 JSON 부분만 추출합니다.

        다양한 형식을 처리:
        1. ```json ... ``` 코드 블록
        2. ``` ... ``` 일반 코드 블록
        3. 순수 JSON (앞뒤 텍스트 제거)
        """
        content = content.strip()

        # 1. ```json ... ``` 형식 처리
        if "```json" in content:
            start_idx = content.find("```json") + len("```json")
            end_idx = content.find("```", start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()

        # 2. ``` ... ``` 형식 처리
        if "```" in content:
            start_idx = content.find("```") + len("```")
            end_idx = content.find("```", start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()

        # 3. 첫 번째 '{' 부터 마지막 '}' 까지 추출
        first_brace = content.find("{")
        last_brace = content.rfind("}")

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return content[first_brace : last_brace + 1]

        return content

    def _repair_json(self, json_str: str) -> str:
        """
        일반적인 JSON 오류를 자동으로 수정합니다.

        수정하는 오류:
        1. Trailing comma: {"a": 1,} → {"a": 1}
        2. JavaScript 주석: // comment, /* comment */
        3. 작은따옴표: {'a': 1} → {"a": 1}
        4. 불완전한 JSON (괄호 불균형)
        """
        original = json_str
        repaired = json_str

        # 1. JavaScript 한 줄 주석 제거 (문자열 내부 제외를 위해 간단한 패턴 사용)
        # 줄 시작이나 공백 뒤의 // 주석만 제거
        repaired = re.sub(r"^\s*//.*$", "", repaired, flags=re.MULTILINE)

        # 2. JavaScript 블록 주석 제거
        repaired = re.sub(r"/\*[\s\S]*?\*/", "", repaired)

        # 3. Trailing comma 제거 (객체와 배열 모두)
        # },] 또는 },} 패턴 처리
        repaired = re.sub(r",(\s*[\]}])", r"\1", repaired)

        # 4. 작은따옴표를 큰따옴표로 변환 (JSON 키/값에서)
        # 주의: 문자열 내부의 작은따옴표는 유지해야 하므로 신중하게 처리
        # 간단한 케이스만 처리: {'key': 'value'} 형태
        repaired = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', repaired)  # 키
        repaired = re.sub(r":\s*'([^']*)'(\s*[,}\]])", r': "\1"\2', repaired)  # 값

        # 5. 불완전한 JSON 수정 시도 (괄호 균형)
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")

        # 닫는 괄호가 부족한 경우 추가
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)
            logger.warning(f"JSON 복구: {open_braces - close_braces}개의 '}}' 추가")

        if open_brackets > close_brackets:
            # 배열이 먼저 닫히고 객체가 닫혀야 함 - 끝에서 역순으로 추가
            repaired = repaired.rstrip()
            if repaired.endswith("}"):
                # 마지막 } 앞에 ] 추가
                repaired = (
                    repaired[:-1] + "]" * (open_brackets - close_brackets) + "}"
                )
            else:
                repaired += "]" * (open_brackets - close_brackets)
            logger.warning(f"JSON 복구: {open_brackets - close_brackets}개의 ']' 추가")

        if repaired != original:
            logger.info("JSON 자동 복구가 적용되었습니다.")

        return repaired

    def _parse_json_response(
        self, response: Dict[str, Any], phase_name: str
    ) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON을 파싱합니다.

        파싱 전략:
        1. JSON 추출 (코드 블록, 앞뒤 텍스트 제거)
        2. 1차 파싱 시도
        3. 실패 시 JSON 복구 후 재시도

        Args:
            response: LLM 응답
            phase_name: 단계 이름 (로깅용)

        Returns:
            Dict[str, Any]: 파싱된 JSON

        Raises:
            ValueError: 파싱 실패 시
        """
        content = response.get("content", "")
        if not content:
            raise ValueError(f"{phase_name} LLM 응답에 content가 없습니다.")

        # Step 1: JSON 추출
        json_str = self._extract_json_from_content(content)

        # Step 2: 1차 파싱 시도
        try:
            result = json.loads(json_str)
            logger.info(f"{phase_name} 응답 파싱 성공")
            return result
        except json.JSONDecodeError as first_error:
            logger.warning(
                f"{phase_name} 1차 JSON 파싱 실패: {first_error}. 자동 복구 시도 중..."
            )

        # Step 3: JSON 복구 후 재시도
        try:
            repaired_json = self._repair_json(json_str)
            result = json.loads(repaired_json)
            logger.info(f"{phase_name} 응답 파싱 성공 (자동 복구 적용)")
            return result
        except json.JSONDecodeError as second_error:
            logger.error(f"{phase_name} JSON 파싱 최종 실패: {second_error}")
            logger.debug(f"원본 응답 (처음 500자):\n{content[:500]}...")
            logger.debug(f"추출된 JSON (처음 500자):\n{json_str[:500]}...")
            raise ValueError(
                f"{phase_name} JSON 파싱 실패: {second_error}\n"
                f"자동 복구를 시도했으나 실패했습니다."
            )

    # ========== 프롬프트 저장 ==========

    def _save_prompt_to_file(
        self,
        prompt: str,
        modification_context: ModificationContext,
        phase_name: str,
    ) -> Path:
        """
        LLM에 전달되는 프롬프트를 .md 파일로 저장합니다.

        파일명 형식: {first_file_name}_{timestamp}_{phase_name}.md

        Args:
            prompt: LLM에 전달되는 프롬프트
            modification_context: 수정 컨텍스트
            phase_name: 단계 이름 (data_mapping, planning, execution)

        Returns:
            Path: 저장된 파일 경로
        """
        # 첫 번째 파일 이름 추출
        first_file_name = "unknown"
        if modification_context.file_paths:
            first_file_path = Path(modification_context.file_paths[0])
            first_file_name = first_file_path.stem
            first_file_name = re.sub(r"[^\w\-]", "_", first_file_name)

        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명 생성
        filename = f"{first_file_name}_{timestamp}_{phase_name}.md"
        output_path = self.prompt_results_dir / filename

        # 프롬프트 저장
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        logger.info(f"프롬프트 저장됨: {output_path}")
        return output_path

    # ========== 세션 및 결과 관리 ==========

    def _create_session_dir(
        self,
        modification_context: ModificationContext,
    ) -> Path:
        """
        컨텍스트별 세션 디렉토리를 생성합니다.

        디렉토리 구조: {output_dir}/{timestamp}/{table_name}/{first_file_name}/

        같은 실행 세션의 모든 컨텍스트는 동일한 timestamp를 공유합니다.

        Args:
            modification_context: 수정 컨텍스트

        Returns:
            Path: 생성된 세션 디렉토리 경로
        """
        # 세션 timestamp 생성 (첫 번째 컨텍스트 처리 시에만)
        if self._session_timestamp is None:
            self._session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        table_name = modification_context.table_name

        # 파일 시스템에 안전한 이름으로 변환 (특수문자 제거)
        safe_table_name = re.sub(r"[^\w\-]", "_", table_name)

        # 컨텍스트의 첫 번째 파일 이름 추출 (확장자 제외)
        first_file_name = "unknown"
        if modification_context.file_paths:
            first_file_path = Path(modification_context.file_paths[0])
            first_file_name = first_file_path.stem  # 확장자 제외한 파일명
            first_file_name = re.sub(r"[^\w\-]", "_", first_file_name)

        # 구조: {timestamp}/{table_name}/{first_file_name}/
        session_dir = (
            self.output_dir / self._session_timestamp / safe_table_name / first_file_name
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"세션 디렉토리 생성됨: {session_dir}")
        return session_dir

    def _save_phase_result(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        step_number: int,
        phase_name: str,
        result: Dict[str, Any],
        tokens_used: int,
    ) -> Path:
        """
        각 Phase의 결과를 JSON 파일로 저장합니다.

        파일명 형식: step{N}_{phase_name}.json

        Args:
            session_dir: 세션 디렉토리 경로
            modification_context: 수정 컨텍스트
            step_number: 단계 번호 (1, 2, 3)
            phase_name: 단계 이름 (planning, execution, data_mapping)
            result: 저장할 결과 데이터
            tokens_used: 사용된 토큰 수

        Returns:
            Path: 저장된 파일 경로
        """
        filename = f"step{step_number}_{phase_name}.json"
        output_path = session_dir / filename

        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "step_number": step_number,
                "phase": phase_name,
                "table_name": modification_context.table_name,
                "layer": modification_context.layer,
                "tokens_used": tokens_used,
                "file_paths": modification_context.file_paths,
            },
            "result": result,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Step {step_number} ({phase_name}) 결과 저장됨: {output_path}")
        return output_path

    # ========== Execution Phase 공통 로직 ==========

    def _create_execution_prompt(
        self,
        modification_context: ModificationContext,
        modification_instructions: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[int, str], Dict[str, str], Dict[str, str]]:
        """
        Execution 프롬프트를 생성합니다.

        Args:
            modification_context: 수정 컨텍스트
            modification_instructions: 수정 지침 리스트

        Returns:
            Tuple[str, Dict[int, str], Dict[str, str], Dict[str, str]]:
                - Execution 프롬프트
                - 인덱스 -> 파일 경로 매핑
                - 파일명 -> 파일 경로 매핑
                - 파일 경로 -> 파일 내용 매핑
        """
        # 소스 파일 내용 (인덱스 형식)
        source_files_str, index_to_path, path_to_content = (
            self._read_file_contents_indexed(modification_context.file_paths)
        )

        # 파일명 -> 파일 경로 매핑 생성
        file_mapping = {Path(fp).name: fp for fp in modification_context.file_paths}

        # 수정 지침을 JSON 문자열로 변환
        instructions_str = json.dumps(
            modification_instructions, indent=2, ensure_ascii=False
        )

        # 템플릿 변수 준비
        variables = {
            "source_files": source_files_str,
            "modification_instructions": instructions_str,
        }

        # 템플릿 렌더링
        template_str = self._load_template(self._get_execution_template_path())
        prompt = self._render_template(template_str, variables)

        return prompt, index_to_path, file_mapping, path_to_content

    def _parse_execution_response(
        self,
        response: Dict[str, Any],
        index_to_path: Dict[int, str],
        file_mapping: Dict[str, str],
        path_to_content: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Execution LLM 응답을 파싱합니다.

        복합 전략을 사용하여 파일 경로를 해결합니다:
        1. 인덱스 매칭 (FILE_1, FILE_2, ...)
        2. 정확한 파일명 매칭
        3. 대소문자 무시 매칭
        4. 코드 시그니처 매칭 (package + class name)
        5. Fuzzy 매칭

        형식:
            ======FILE_1======
            (수정된 코드)
            ======END======

        Args:
            response: LLM 응답
            index_to_path: 인덱스 -> 파일 경로 매핑
            file_mapping: 파일명 -> 파일 경로 매핑
            path_to_content: 파일 경로 -> 원본 내용 매핑

        Returns:
            List[Dict[str, Any]]: 파싱된 수정 정보 리스트
        """
        content = response.get("content", "")
        if not content:
            raise ValueError("Execution LLM 응답에 content가 없습니다.")

        modifications = []
        content = content.strip()

        # ======END====== 기준으로 블록 분리
        blocks = content.split("======END======")

        # 매칭 통계
        match_stats = {
            "index": 0,
            "exact": 0,
            "case_insensitive": 0,
            "signature": 0,
            "fuzzy": 0,
            "failed": 0,
        }

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # FILE 마커 찾기 (======FILE_1====== 또는 ======FILE====== 형식 모두 지원)
            file_marker_match = re.search(r"======FILE(?:_(\d+))?======", block)
            if not file_marker_match:
                continue

            try:
                # 파일 식별자 추출
                if file_marker_match.group(1):
                    # 인덱스 형식: ======FILE_1======
                    file_identifier = f"FILE_{file_marker_match.group(1)}"
                else:
                    # 기존 형식: ======FILE======\nFileName.java
                    file_identifier = self._extract_section(
                        block, "======FILE======", "======MODIFIED_CODE======"
                    ).strip()

                # MODIFIED_CODE 섹션 추출
                modified_code = self._extract_section(
                    block, "======MODIFIED_CODE======", "======END======"
                ).strip()

                # 복합 전략으로 파일 경로 해결
                resolved_path, match_method = self._resolve_file_path(
                    llm_file_identifier=file_identifier,
                    modified_code=modified_code,
                    index_to_path=index_to_path,
                    file_mapping=file_mapping,
                    path_to_content=path_to_content,
                )

                match_stats[match_method] += 1

                if resolved_path:
                    modifications.append(
                        {
                            "file_path": resolved_path,
                            "modified_code": modified_code,
                            "match_method": match_method,
                        }
                    )
                else:
                    logger.error(
                        f"파일 경로 해결 실패: {file_identifier}. "
                        f"이 파일의 수정 사항은 건너뜁니다."
                    )

            except Exception as e:
                logger.warning(f"Execution 블록 파싱 중 오류 (건너뜀): {e}")
                continue

        # 매칭 통계 로깅
        logger.info(
            f"파일 매칭 통계: 인덱스={match_stats['index']}, "
            f"정확={match_stats['exact']}, 대소문자무시={match_stats['case_insensitive']}, "
            f"시그니처={match_stats['signature']}, Fuzzy={match_stats['fuzzy']}, "
            f"실패={match_stats['failed']}"
        )

        if not modifications:
            raise Exception(
                "Execution 응답 파싱 실패: 유효한 수정 블록을 찾을 수 없습니다."
            )

        logger.info(
            f"{len(modifications)}개 파일 수정 정보를 파싱했습니다 (복합 매칭 전략 사용)."
        )
        return modifications

    def _execute_execution_phase(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        modification_instructions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Execution phase를 실행합니다.

        Args:
            session_dir: 세션 디렉토리 경로
            modification_context: 수정 컨텍스트
            modification_instructions: 수정 지침 리스트

        Returns:
            Tuple[List[Dict[str, Any]], int]: (파싱된 수정 정보 리스트, 사용된 토큰 수)
        """
        step_config = self._get_step_config()
        logger.info("-" * 40)
        logger.info(f"[Execution Phase] 시작...")
        logger.info(f"Provider: {step_config.execution_provider}")
        logger.info(f"Model: {step_config.execution_model}")

        # 프롬프트 생성
        prompt, index_to_path, file_mapping, path_to_content = (
            self._create_execution_prompt(modification_context, modification_instructions)
        )
        logger.debug(f"Execution 프롬프트 길이: {len(prompt)} chars")
        logger.debug(f"파일 인덱스 매핑: {list(index_to_path.keys())}")

        # 프롬프트 저장 (LLM 호출 직전)
        self._save_prompt_to_file(prompt, modification_context, "execution")

        # LLM 호출
        response = self._get_execution_provider().call(prompt)
        tokens_used = response.get("tokens_used", 0)
        logger.info(f"Execution LLM 응답 완료 (토큰: {tokens_used})")

        # 응답 파싱
        parsed_modifications = self._parse_execution_response(
            response=response,
            index_to_path=index_to_path,
            file_mapping=file_mapping,
            path_to_content=path_to_content,
        )

        # Execution 결과 저장 (step_number는 서브클래스에서 결정)
        execution_step_number = self._get_execution_step_number()
        execution_result = {
            "modifications": [
                {
                    "file_path": mod.get("file_path"),
                    "match_method": mod.get("match_method"),
                    "code_length": len(mod.get("modified_code", "")),
                }
                for mod in parsed_modifications
            ],
            "raw_response_length": len(response.get("content", "")),
        }
        self._save_phase_result(
            session_dir=session_dir,
            modification_context=modification_context,
            step_number=execution_step_number,
            phase_name="execution",
            result=execution_result,
            tokens_used=tokens_used,
        )

        return parsed_modifications, tokens_used

    @abstractmethod
    def _get_execution_step_number(self) -> int:
        """
        Execution phase의 단계 번호를 반환합니다.

        Returns:
            int: 단계 번호 (TwoStep: 2, ThreeStep: 3)
        """
        pass

    # ========== 메인 인터페이스 ==========

    def create_prompt(self, input_data: CodeGeneratorInput) -> str:
        """
        토큰 계산을 위한 프롬프트를 생성합니다.

        Note: 이 메서드는 BaseContextGenerator.create_batches()에서
        토큰 크기 계산 목적으로만 사용됩니다.

        Args:
            input_data: Code 생성 입력

        Returns:
            str: 토큰 계산용 프롬프트
        """
        # 소스 파일 내용 읽기
        source_files_str = self._read_file_contents(input_data.file_paths)

        # 간단한 템플릿으로 토큰 계산
        variables = {
            "table_info": input_data.table_info,
            "source_files": source_files_str,
        }

        return self._render_template(_TOKEN_CALCULATION_TEMPLATE, variables)

    def calculate_token_size(self, text: str) -> int:
        """텍스트의 토큰 크기를 계산합니다."""
        if self.token_encoder:
            return len(self.token_encoder.encode(text))
        else:
            # 간단한 추정: 문자 4개당 1토큰
            return len(text) // 4

    def generate(self, input_data: CodeGeneratorInput) -> CodeGeneratorOutput:
        """
        BaseCodeGenerator 인터페이스 준수를 위한 메서드.

        Note: MultiStep CodeGenerator는 generate() 대신
        generate_modification_plan()을 사용합니다.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}는 generate() 대신 "
            "generate_modification_plan()을 사용하세요."
        )

    def generate_modification_plan(
        self,
        modification_context: ModificationContext,
        table_access_info: Optional[TableAccessInfo] = None,
    ) -> List[ModificationPlan]:
        """
        다단계 LLM 협업을 통해 수정 계획을 생성합니다.

        Template Method 패턴:
        1. execution_only 모드면 이전 Planning 결과 로드
        2. 아니면 Planning phases 실행 (서브클래스에서 구현)
        3. plan_only 모드면 여기서 종료
        4. Execution phase 실행 (공통)
        5. ModificationPlan 리스트 생성 (공통)

        Args:
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보 (execution_only 모드가 아닐 때 필수)

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        if not modification_context.file_paths:
            return []

        # 모드 확인
        mode = self._get_mode()
        is_execution_only = mode == "execution_only"
        is_plan_only = mode == "plan_only"

        # execution_only 모드가 아닐 때만 table_access_info 필수
        if not is_execution_only and not table_access_info:
            raise ValueError(
                f"{self.__class__.__name__}는 table_access_info가 필수입니다. "
                "(execution_only 모드를 사용하는 경우 제외)"
            )

        plans = []
        layer_name = modification_context.layer
        total_tokens_used = 0
        step_name = self._get_step_name()

        logger.info("=" * 60)
        logger.info(f"{step_name} Code Generation 시작")
        logger.info(f"테이블: {modification_context.table_name}")
        logger.info(f"레이어: {layer_name}")
        logger.info(f"소스 파일 수: {len(modification_context.file_paths)}")
        if modification_context.context_files:
            logger.info(f"컨텍스트 파일 수: {len(modification_context.context_files)}")
        for fp in modification_context.file_paths:
            logger.info(f"  - {fp}")

        # 모드 표시
        if is_execution_only:
            plan_timestamp = self._get_plan_timestamp()
            logger.info("모드: EXECUTION_ONLY (이전 Planning 결과에서 Execution만 실행)")
            logger.info(f"  - plan_timestamp: {plan_timestamp}")
        elif is_plan_only:
            logger.info("모드: PLAN_ONLY (Planning까지만 실행)")
        else:
            logger.info("모드: FULL (Planning + Execution)")
        logger.info("=" * 60)

        try:
            # ===== Step 1: Planning 결과 획득 =====
            if is_execution_only:
                # Execution Only 모드: 이전 Planning 결과 로드
                session_dir = self._get_plan_session_dir(modification_context)
                if session_dir is None:
                    # Plan이 없는 컨텍스트는 건너뜀
                    logger.warning(
                        f"컨텍스트에 해당하는 Plan을 찾을 수 없습니다. 건너뜁니다: "
                        f"table={modification_context.table_name}, "
                        f"first_file={Path(modification_context.file_paths[0]).stem}"
                    )
                    return []
                planning_result = self._load_previous_planning_result(session_dir)
                logger.info(f"이전 Planning 결과를 사용합니다: {session_dir}")
            else:
                # Full/Plan-Only 모드: 새로운 세션 디렉토리 생성 및 Planning 실행
                session_dir = self._create_session_dir(modification_context)
                planning_result, planning_tokens = self._execute_planning_phases(
                    session_dir, modification_context, table_access_info
                )
                total_tokens_used += planning_tokens

            # ===== Step 2: Plan-Only 모드 체크 =====
            if is_plan_only:
                return self._create_plan_only_results(
                    modification_context, planning_result, session_dir, total_tokens_used
                )

            # ===== Step 3: Execution Phase 실행 (공통) =====
            modification_instructions = planning_result.get(
                "modification_instructions", []
            )
            parsed_modifications, execution_tokens = self._execute_execution_phase(
                session_dir, modification_context, modification_instructions
            )
            total_tokens_used += execution_tokens

            # Planning 지침에서 파일명 -> reason 매핑 생성
            planning_reasons = self._get_planning_reasons(planning_result)

            # 각 수정 사항에 대해 계획 생성
            for mod in parsed_modifications:
                file_path_str = mod.get("file_path", "")
                modified_code = mod.get("modified_code", "")

                # reason은 Planning 지침에서 가져옴
                file_name = Path(file_path_str).name
                reason = planning_reasons.get(file_name, "")

                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    file_path = Path(self.config.target_project) / file_path

                file_path = file_path.resolve()

                plan = ModificationPlan(
                    file_path=str(file_path),
                    layer_name=layer_name,
                    modification_type="encryption",
                    modified_code=modified_code,
                    reason=reason,
                    tokens_used=total_tokens_used,
                    status="pending",
                )

                if not modified_code or modified_code.strip() == "":
                    logger.info(f"파일 수정 건너뜀: {file_path} (이유: {reason})")
                    plan.status = "skipped"

                plans.append(plan)

            logger.info("=" * 60)
            logger.info(f"{step_name} Code Generation 완료")
            logger.info(f"총 토큰 사용량: {total_tokens_used}")
            logger.info(f"생성된 계획 수: {len(plans)}")
            logger.info(f"결과 저장 경로: {session_dir}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"{step_name} Code Generation 실패: {e}")
            # 모든 파일에 대해 실패 계획 기록
            for file_path in modification_context.file_paths:
                plans.append(
                    ModificationPlan(
                        file_path=file_path,
                        layer_name=layer_name,
                        modification_type="encryption",
                        modified_code="",
                        reason=f"{step_name} 생성 실패: {str(e)}",
                        tokens_used=total_tokens_used,
                        status="failed",
                    )
                )

        return plans
