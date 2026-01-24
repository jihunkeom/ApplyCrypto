"""
Two-Step Code Generator

2단계 LLM 협업 전략 (Planning + Execution)을 사용하는 CodeGenerator입니다.

- Step 1 (Planning): 논리적 분석 능력이 뛰어난 모델 (예: GPT-OSS-120B)이
  Data Flow를 분석하고 구체적인 수정 지침을 생성합니다.
- Step 2 (Execution): 코드 생성 안정성이 높은 모델 (예: Codestral-2508)이
  수정 지침에 따라 실제 코드를 작성합니다.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from config.config_manager import Configuration, TwoStepConfig
from models.modification_context import ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.llm.llm_factory import create_llm_provider
from modifier.llm.llm_provider import LLMProvider

from ..multi_step_base import BaseMultiStepCodeGenerator

logger = logging.getLogger("applycrypto")


class TwoStepCodeGenerator(BaseMultiStepCodeGenerator):
    """2단계 LLM 협업 Code 생성기 (Planning + Execution)"""

    def __init__(self, config: Configuration):
        """
        TwoStepCodeGenerator 초기화

        Args:
            config: 설정 객체 (two_step_config 필수)

        Raises:
            ValueError: two_step_config가 설정되지 않은 경우
        """
        # 부모 클래스 초기화 (토큰 인코더 등)
        super().__init__(config)

        # TwoStepConfig 검증
        if not config.two_step_config:
            raise ValueError(
                "modification_type이 'TwoStep'일 때는 two_step_config가 필수입니다."
            )

        self.two_step_config: TwoStepConfig = config.two_step_config

        # 두 개의 LLM Provider 초기화
        logger.info(
            f"Planning LLM 초기화: {self.two_step_config.planning_provider} "
            f"(model: {self.two_step_config.planning_model})"
        )
        self.planning_provider: LLMProvider = create_llm_provider(
            provider_name=self.two_step_config.planning_provider,
            model_id=self.two_step_config.planning_model,
        )

        logger.info(
            f"Execution LLM 초기화: {self.two_step_config.execution_provider} "
            f"(model: {self.two_step_config.execution_model})"
        )
        self.execution_provider: LLMProvider = create_llm_provider(
            provider_name=self.two_step_config.execution_provider,
            model_id=self.two_step_config.execution_model,
        )

        # 템플릿 로드
        template_dir = Path(__file__).parent
        self.planning_template_path = template_dir / "planning_template.md"
        self.execution_template_path = template_dir / "execution_template.md"

        if not self.planning_template_path.exists():
            raise FileNotFoundError(
                f"Planning 템플릿을 찾을 수 없습니다: {self.planning_template_path}"
            )
        if not self.execution_template_path.exists():
            raise FileNotFoundError(
                f"Execution 템플릿을 찾을 수 없습니다: {self.execution_template_path}"
            )

        # BaseContextGenerator.create_batches()에서 토큰 계산을 위해 사용하는 속성
        self.template_path = self.planning_template_path

        # 출력 디렉토리 초기화 (부모 클래스 메서드 사용)
        self._init_output_directory()

    # ========== 추상 메서드 구현 ==========

    def _get_output_subdir_name(self) -> str:
        """출력 디렉토리 하위 폴더명 반환"""
        return "two_step_results"

    def _get_step_config(self) -> TwoStepConfig:
        """TwoStepConfig 반환"""
        return self.two_step_config

    def _get_execution_provider(self) -> LLMProvider:
        """Execution LLM provider 반환"""
        return self.execution_provider

    def _get_execution_template_path(self) -> Path:
        """Execution 템플릿 경로 반환"""
        return self.execution_template_path

    def _get_step_name(self) -> str:
        """Step 이름 반환 (로깅용)"""
        return "2-Step"

    def _get_execution_step_number(self) -> int:
        """Execution phase의 단계 번호 반환"""
        return 2

    def _get_last_planning_step_number(self) -> int:
        """마지막 Planning 단계의 번호 반환 (TwoStep은 Step 1)"""
        return 1

    def _get_last_planning_phase_name(self) -> str:
        """마지막 Planning 단계의 이름 반환"""
        return "planning"

    def _get_planning_reasons(
        self, planning_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Planning 결과에서 파일명 -> reason 매핑 추출"""
        planning_reasons = {}
        for instr in planning_result.get("modification_instructions", []):
            file_name = instr.get("file_name", "")
            reason = instr.get("reason", "")
            if file_name:
                planning_reasons[file_name] = reason
        return planning_reasons

    def _execute_planning_phases(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Planning phase를 실행합니다.

        TwoStep은 단일 Planning phase만 실행합니다.

        Args:
            session_dir: 세션 디렉토리 경로
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보

        Returns:
            Tuple[Dict[str, Any], int]: (planning_result, tokens_used)
        """
        logger.info("-" * 40)
        logger.info("[Step 1] Planning LLM 호출 중...")
        logger.info(f"Provider: {self.two_step_config.planning_provider}")
        logger.info(f"Model: {self.two_step_config.planning_model}")

        # Planning 프롬프트 생성
        planning_prompt = self._create_planning_prompt(
            modification_context, table_access_info
        )
        logger.debug(f"Planning 프롬프트 길이: {len(planning_prompt)} chars")

        # 프롬프트 저장 (LLM 호출 직전)
        self._save_prompt_to_file(planning_prompt, modification_context, "planning")

        # LLM 호출
        planning_response = self.planning_provider.call(planning_prompt)
        tokens_used = planning_response.get("tokens_used", 0)
        logger.info(f"Planning LLM 응답 완료 (토큰: {tokens_used})")

        # Planning 응답 파싱 (부모 클래스의 범용 메서드 사용)
        modification_instructions = self._parse_json_response(
            planning_response, "Planning"
        )

        # Planning 결과 저장
        self._save_phase_result(
            session_dir=session_dir,
            modification_context=modification_context,
            step_number=1,
            phase_name="planning",
            result=modification_instructions,
            tokens_used=tokens_used,
        )

        # Planning 결과 로깅
        if "data_flow_analysis" in modification_instructions:
            analysis = modification_instructions["data_flow_analysis"]
            logger.info(f"Data Flow 분석 결과: {analysis.get('overview', 'N/A')}")

        instruction_count = len(
            modification_instructions.get("modification_instructions", [])
        )
        logger.info(f"생성된 수정 지침 수: {instruction_count}")

        return modification_instructions, tokens_used

    # ========== TwoStep 고유 메서드 ==========

    def _create_planning_prompt(
        self,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> str:
        """
        Step 1 (Planning) 프롬프트를 생성합니다.

        Args:
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보

        Returns:
            str: Planning 프롬프트
        """
        # 테이블/칼럼 정보
        table_info = {
            "table_name": modification_context.table_name,
            "columns": modification_context.columns,
        }
        table_info_str = json.dumps(table_info, indent=2, ensure_ascii=False)

        # 소스 파일 내용
        source_files_str = self._read_file_contents(modification_context.file_paths)

        # 컨텍스트 파일 (VO) 내용
        context_files_str = self._read_file_contents(
            modification_context.context_files or []
        )

        # SQL 쿼리 정보 (핵심 Data Flow 정보)
        sql_queries_str = self._get_sql_queries_for_prompt(
            table_access_info, modification_context.file_paths
        )

        # Call Stacks 정보
        call_stacks_str = self._get_callstacks_from_table_access_info(
            modification_context.file_paths, table_access_info
        )

        # 템플릿 변수 준비
        variables = {
            "table_info": table_info_str,
            "source_files": source_files_str,
            "context_files": context_files_str,
            "sql_queries": sql_queries_str,
            "call_stacks": call_stacks_str,
        }

        # 템플릿 렌더링
        template_str = self._load_template(self.planning_template_path)
        return self._render_template(template_str, variables)
