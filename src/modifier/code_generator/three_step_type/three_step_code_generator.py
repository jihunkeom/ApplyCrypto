"""
Three-Step Code Generator

3단계 LLM 협업 전략 (VO Extraction + Planning + Execution)을 사용하는 CodeGenerator입니다.

- Phase 1 (VO Extraction): VO 파일과 SQL 쿼리에서 필드 매핑 정보를 추출합니다.
- Phase 2 (Planning): vo_info를 기반으로 Data Flow를 분석하고 수정 지침을 생성합니다.
- Phase 3 (Execution): 수정 지침에 따라 실제 코드를 작성합니다.

Phase 1, 2는 분석 능력이 뛰어난 모델 (예: GPT-OSS-120B)이 수행하고,
Phase 3는 코드 생성 안정성이 높은 모델 (예: Codestral-2508)이 수행합니다.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from config.config_manager import Configuration, ThreeStepConfig
from models.modification_context import ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.llm.llm_factory import create_llm_provider
from modifier.llm.llm_provider import LLMProvider

from ..multi_step_base import BaseMultiStepCodeGenerator

logger = logging.getLogger("applycrypto")


class ThreeStepCodeGenerator(BaseMultiStepCodeGenerator):
    """3단계 LLM 협업 Code 생성기 (VO Extraction + Planning + Execution)"""

    def __init__(self, config: Configuration):
        """
        ThreeStepCodeGenerator 초기화

        Args:
            config: 설정 객체 (three_step_config 필수)

        Raises:
            ValueError: three_step_config가 설정되지 않은 경우
        """
        # 부모 클래스 초기화 (토큰 인코더 등)
        super().__init__(config)

        # ThreeStepConfig 검증
        if not config.three_step_config:
            raise ValueError(
                "modification_type이 'ThreeStep'일 때는 three_step_config가 필수입니다."
            )

        self.three_step_config: ThreeStepConfig = config.three_step_config

        # LLM Provider 초기화
        # Phase 1, 2: 분석용 (analysis_provider)
        logger.info(
            f"Analysis LLM 초기화: {self.three_step_config.analysis_provider} "
            f"(model: {self.three_step_config.analysis_model})"
        )
        self.analysis_provider: LLMProvider = create_llm_provider(
            provider_name=self.three_step_config.analysis_provider,
            model_id=self.three_step_config.analysis_model,
        )

        # Phase 3: 코드 생성용 (execution_provider)
        logger.info(
            f"Execution LLM 초기화: {self.three_step_config.execution_provider} "
            f"(model: {self.three_step_config.execution_model})"
        )
        self.execution_provider: LLMProvider = create_llm_provider(
            provider_name=self.three_step_config.execution_provider,
            model_id=self.three_step_config.execution_model,
        )

        # 템플릿 로드
        template_dir = Path(__file__).parent
        self.data_mapping_template_path = template_dir / "data_mapping_template.md"
        self.data_mapping_template_ccs_path = template_dir / "data_mapping_template_ccs.md"
        self.planning_template_path = template_dir / "planning_template.md"
        self.execution_template_path = template_dir / "execution_template.md"

        for template_path in [
            self.data_mapping_template_path,
            self.planning_template_path,
            self.execution_template_path,
        ]:
            if not template_path.exists():
                raise FileNotFoundError(f"템플릿을 찾을 수 없습니다: {template_path}")

        # CCS 템플릿은 선택적 (없어도 에러 아님)
        if not self.data_mapping_template_ccs_path.exists():
            logger.warning(
                f"CCS 템플릿이 없습니다: {self.data_mapping_template_ccs_path}. "
                "CCS 프로젝트에서는 일반 템플릿을 사용합니다."
            )

        # BaseContextGenerator.create_batches()에서 토큰 계산을 위해 사용하는 속성
        self.template_path = self.planning_template_path

        # 출력 디렉토리 초기화 (부모 클래스 메서드 사용)
        self._init_output_directory()

    # ========== 추상 메서드 구현 ==========

    def _get_output_subdir_name(self) -> str:
        """출력 디렉토리 하위 폴더명 반환"""
        return "three_step_results"

    def _get_step_config(self) -> ThreeStepConfig:
        """ThreeStepConfig 반환"""
        return self.three_step_config

    def _get_execution_provider(self) -> LLMProvider:
        """Execution LLM provider 반환"""
        return self.execution_provider

    def _get_execution_template_path(self) -> Path:
        """Execution 템플릿 경로 반환"""
        return self.execution_template_path

    def _get_step_name(self) -> str:
        """Step 이름 반환 (로깅용)"""
        return "3-Step"

    def _get_execution_step_number(self) -> int:
        """Execution phase의 단계 번호 반환"""
        return 3

    def _get_last_planning_step_number(self) -> int:
        """마지막 Planning 단계의 번호 반환 (ThreeStep은 Step 2)"""
        return 2

    def _get_last_planning_phase_name(self) -> str:
        """마지막 Planning 단계의 이름 반환"""
        return "planning"

    def _get_planning_reasons(self, planning_result: Dict[str, Any]) -> Dict[str, str]:
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
        Planning phases를 실행합니다.

        ThreeStep은 Phase 1 (Data Mapping) + Phase 2 (Planning)를 실행합니다.

        Args:
            session_dir: 세션 디렉토리 경로
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보

        Returns:
            Tuple[Dict[str, Any], int]: (planning_result, total_tokens_used)
        """
        total_tokens = 0

        # ===== Phase 1: Data Mapping Extraction =====
        mapping_info, phase1_tokens = self._execute_data_mapping_phase(
            session_dir, modification_context, table_access_info
        )
        total_tokens += phase1_tokens

        # ===== Phase 2: Planning =====
        planning_result, phase2_tokens = self._execute_planning_phase(
            session_dir, modification_context, table_access_info, mapping_info
        )
        total_tokens += phase2_tokens

        return planning_result, total_tokens

    # ========== ThreeStep 고유 메서드: Phase 1 (Data Mapping Extraction) ==========

    def _create_data_mapping_prompt(
        self,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> str:
        """Phase 1 (Data Mapping Extraction) 프롬프트를 생성합니다."""
        # 테이블/칼럼 정보 (★ 타겟 테이블 명시)
        table_info = {
            "table_name": modification_context.table_name,
            "columns": modification_context.columns,
        }
        table_info_str = json.dumps(table_info, indent=2, ensure_ascii=False)

        # VO 파일 내용 (context_files)
        vo_files_str = self._read_file_contents(
            modification_context.context_files or []
        )

        # SQL 쿼리 정보
        sql_queries_str = self._get_sql_queries_for_prompt(
            table_access_info, modification_context.file_paths
        )

        variables = {
            "table_info": table_info_str,
            "vo_files": vo_files_str,
            "sql_queries": sql_queries_str,
        }

        template_str = self._load_template(self.data_mapping_template_path)
        return self._render_template(template_str, variables)

    # ========== CCS 전용 메서드: Phase 1 (resultMap 기반 필드 매핑) ==========

    def _is_ccs_project(self) -> bool:
        """
        CCS 프로젝트 여부를 판단합니다.

        Returns:
            bool: CCS 프로젝트이면 True
        """
        return self.config.sql_wrapping_type in ("mybatis_ccs", "mybatis_ccs_batch")

    def _extract_field_mappings_from_sql_queries(
        self,
        table_access_info: TableAccessInfo,
        file_paths: list,
    ) -> Dict[str, Any]:
        """
        SQL 쿼리에서 필드 매핑 정보를 추출합니다.

        DQM.xml의 resultMap에서 추출한 컬럼↔필드 매핑과
        SQL 내 #{fieldName} 패턴을 사용합니다.

        Args:
            table_access_info: 테이블 접근 정보
            file_paths: 수정 대상 파일 경로 목록

        Returns:
            Dict with structure:
            {
                "select_mappings": [
                    {
                        "query_id": "selectEmp",
                        "result_type": "com.example.EmpDVO",
                        "result_map": "selectEmp-result",
                        "mappings": [
                            {"java_field": "empNm", "db_column": "EMP_NM"},
                            ...
                        ]
                    }
                ],
                "write_mappings": [
                    {
                        "query_id": "insertEmp",
                        "query_type": "INSERT",
                        "parameter_type": "com.example.EmpDVO",
                        "fields": ["empNm", "birthDt", ...]
                    }
                ]
            }
        """
        select_mappings = []
        write_mappings = []

        for sql_query in table_access_info.sql_queries:
            strategy_specific = sql_query.get("strategy_specific", {})
            query_type = sql_query.get("query_type", "").upper()
            query_id = sql_query.get("id", "")

            if query_type == "SELECT":
                # resultMap에서 추출한 필드 매핑
                result_field_mappings = strategy_specific.get(
                    "result_field_mappings", []
                )
                if result_field_mappings:
                    select_mappings.append(
                        {
                            "query_id": query_id,
                            "result_type": strategy_specific.get("result_type", ""),
                            "result_map": strategy_specific.get("result_map", ""),
                            "mappings": [
                                {"java_field": fm[0], "db_column": fm[1]}
                                for fm in result_field_mappings
                            ],
                        }
                    )

            elif query_type in ("INSERT", "UPDATE"):
                # SQL 내 #{fieldName} 패턴에서 추출한 파라미터 필드
                param_fields = strategy_specific.get("parameter_field_mappings", [])
                if param_fields:
                    write_mappings.append(
                        {
                            "query_id": query_id,
                            "query_type": query_type,
                            "parameter_type": strategy_specific.get(
                                "parameter_type", ""
                            ),
                            "fields": param_fields,
                        }
                    )

        logger.info(
            f"CCS 필드 매핑 추출 완료: SELECT {len(select_mappings)}개, "
            f"INSERT/UPDATE {len(write_mappings)}개"
        )

        return {
            "select_mappings": select_mappings,
            "write_mappings": write_mappings,
        }

    def _format_ccs_sql_with_relevant_mappings(
        self,
        table_access_info: TableAccessInfo,
        file_paths: list,
        target_columns: list,
    ) -> str:
        """
        CCS용 SQL 쿼리와 관련 필드 매핑을 함께 포맷팅합니다.

        각 쿼리 밑에 target_columns에 해당하는 필드 매핑만 자연어로 설명합니다.

        Args:
            table_access_info: 테이블 접근 정보
            file_paths: 파일 경로 리스트 (필터링용)
            target_columns: 관심 대상 컬럼명 리스트

        Returns:
            str: 포맷팅된 SQL 쿼리 + 매핑 정보 문자열
        """
        from pathlib import Path

        # target_columns를 대문자로 정규화 (비교용)
        target_cols_upper = {col.upper() for col in target_columns}

        # 파일 경로에서 클래스명 추출 (필터링용)
        file_class_names = set()
        if file_paths:
            for file_path in file_paths:
                class_name = Path(file_path).stem
                file_class_names.add(class_name)

        output_parts = []
        query_num = 0

        for sql_query in table_access_info.sql_queries:
            # 파일 경로가 지정된 경우 관련 SQL만 필터링
            if file_paths and file_class_names:
                call_stacks = sql_query.get("call_stacks", [])
                is_relevant = False
                for call_stack in call_stacks:
                    if not isinstance(call_stack, list):
                        continue
                    for method_sig in call_stack:
                        if not isinstance(method_sig, str):
                            continue
                        method_class_name = (
                            method_sig.split(".")[0] if "." in method_sig else method_sig
                        )
                        if method_class_name in file_class_names:
                            is_relevant = True
                            break
                    if is_relevant:
                        break
                if not is_relevant:
                    continue

            query_num += 1
            query_id = sql_query.get("id", "unknown")
            query_type = sql_query.get("query_type", "SELECT")
            sql_text = sql_query.get("sql", "")
            strategy_specific = sql_query.get("strategy_specific", {})

            # 쿼리 헤더
            output_parts.append(f"### Query {query_num}: {query_id} ({query_type})")
            output_parts.append("")

            # SQL 텍스트 (strategy_specific 제외한 간결한 형태)
            output_parts.append("**SQL:**")
            output_parts.append("```sql")
            output_parts.append(sql_text.strip())
            output_parts.append("```")
            output_parts.append("")

            # 메타 정보
            param_type = strategy_specific.get("parameter_type", "")
            result_type = strategy_specific.get("result_type", "")
            if param_type:
                output_parts.append(f"- **Parameter Type:** `{param_type}`")
            if result_type:
                output_parts.append(f"- **Result Type:** `{result_type}`")
            output_parts.append("")

            # 관련 필드 매핑 (target_columns만 필터링)
            relevant_mappings = []

            if query_type == "SELECT":
                # resultMap에서 추출한 필드 매핑
                result_field_mappings = strategy_specific.get("result_field_mappings", [])
                for java_field, db_column in result_field_mappings:
                    if db_column.upper() in target_cols_upper:
                        relevant_mappings.append(
                            f"- Column `{db_column}` → Java field `{java_field}`"
                        )

            elif query_type in ("INSERT", "UPDATE"):
                # SQL 내 #{fieldName} 패턴
                param_fields = strategy_specific.get("parameter_field_mappings", [])
                # INSERT/UPDATE는 SQL에서 컬럼명을 직접 추출해야 함
                # 여기서는 parameter_fields만 제공하고, 컬럼 매칭은 LLM에게 위임
                if param_fields:
                    # 간단히 모든 파라미터 필드 나열 (LLM이 컬럼과 매칭)
                    for field in param_fields:
                        relevant_mappings.append(
                            f"- Parameter `#{{{field}}}` → Java field `{field}`"
                        )

            if relevant_mappings:
                output_parts.append(
                    f"**Relevant Field Mappings for Target Columns ({', '.join(target_columns)}):**"
                )
                output_parts.extend(relevant_mappings)
            else:
                output_parts.append(
                    "**Field Mappings:** No direct mapping found for target columns. "
                    "Infer from SQL parameter names or use camelCase conversion."
                )

            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        if query_num == 0:
            return "No relevant SQL queries found for this context."

        logger.info(f"CCS SQL 쿼리 포맷팅 완료: {query_num}개 쿼리")
        return "\n".join(output_parts)

    def _create_ccs_data_mapping_prompt(
        self,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> str:
        """
        CCS 전용 Phase 1 프롬프트를 생성합니다.

        VO 파일 전체 대신 DQM.xml에서 추출한 필드 매핑 정보를
        각 SQL 쿼리와 함께 컴팩트하게 제공합니다.

        Args:
            modification_context: 수정 컨텍스트
            table_access_info: 테이블 접근 정보

        Returns:
            str: 렌더링된 프롬프트
        """
        # 테이블/칼럼 정보
        table_info = {
            "table_name": modification_context.table_name,
            "columns": modification_context.columns,
        }
        table_info_str = json.dumps(table_info, indent=2, ensure_ascii=False)

        # target columns 추출 (관심 대상 컬럼명 리스트)
        target_columns = [col.get("name", "") for col in modification_context.columns]

        # SQL 쿼리 + 관련 필드 매핑을 함께 포맷팅 (중복 제거, 컴팩트화)
        sql_queries_with_mappings = self._format_ccs_sql_with_relevant_mappings(
            table_access_info, modification_context.file_paths, target_columns
        )

        variables = {
            "table_info": table_info_str,
            "sql_queries_with_mappings": sql_queries_with_mappings,
        }

        # CCS 템플릿 사용 (없으면 일반 템플릿 사용)
        if self.data_mapping_template_ccs_path.exists():
            template_str = self._load_template(self.data_mapping_template_ccs_path)
        else:
            logger.warning("CCS 템플릿이 없어 일반 템플릿을 사용합니다.")
            template_str = self._load_template(self.data_mapping_template_path)

        return self._render_template(template_str, variables)

    def _execute_data_mapping_phase(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
    ) -> Tuple[Dict[str, Any], int]:
        """Step 1 (Query Analysis): VO 파일과 SQL 쿼리에서 데이터 매핑 정보를 추출합니다."""
        logger.info("-" * 40)
        logger.info("[Step 1] Query Analysis 시작...")

        # CCS 프로젝트 여부에 따라 프롬프트 생성 방식 분기
        if self._is_ccs_project():
            logger.info(
                "CCS 프로젝트 감지: resultMap 기반 필드 매핑을 사용합니다."
            )
            prompt = self._create_ccs_data_mapping_prompt(
                modification_context, table_access_info
            )
        else:
            logger.info(f"VO 파일 수: {len(modification_context.context_files or [])}")
            prompt = self._create_data_mapping_prompt(
                modification_context, table_access_info
            )

        logger.debug(f"Query Analysis 프롬프트 길이: {len(prompt)} chars")

        # 프롬프트 저장 (LLM 호출 직전)
        self._save_prompt_to_file(prompt, modification_context, "data_mapping")

        # LLM 호출
        response = self.analysis_provider.call(prompt)
        tokens_used = response.get("tokens_used", 0)
        logger.info(f"Query Analysis 응답 완료 (토큰: {tokens_used})")

        # 응답 파싱 (부모 클래스의 범용 메서드 사용)
        mapping_info = self._parse_json_response(response, "Query Analysis")

        # 결과 저장
        self._save_phase_result(
            session_dir=session_dir,
            modification_context=modification_context,
            step_number=1,
            phase_name="query_analysis",
            result=mapping_info,
            tokens_used=tokens_used,
        )

        return mapping_info, tokens_used

    # ========== ThreeStep 고유 메서드: Phase 2 (Planning) ==========

    def _create_planning_prompt(
        self,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
        mapping_info: Dict[str, Any],
    ) -> str:
        """Phase 2 (Planning) 프롬프트를 생성합니다.

        Note: SQL 쿼리와 데이터 매핑 정보는 Phase 1의 mapping_info에 포함되어 있습니다.
        """
        # 테이블/칼럼 정보
        table_info = {
            "table_name": modification_context.table_name,
            "columns": modification_context.columns,
        }
        table_info_str = json.dumps(table_info, indent=2, ensure_ascii=False)

        # 소스 파일 내용
        source_files_str = self._read_file_contents(modification_context.file_paths)

        # mapping_info (Phase 1 결과)를 JSON 문자열로 변환
        mapping_info_str = json.dumps(mapping_info, indent=2, ensure_ascii=False)

        # Call Stacks 정보 (각 call chain 별로 data flow 분석 수행)
        call_stacks_str = self._get_callstacks_from_table_access_info(
            modification_context.file_paths, table_access_info
        )

        variables = {
            "table_info": table_info_str,
            "source_files": source_files_str,
            "mapping_info": mapping_info_str,
            "call_stacks": call_stacks_str,
        }

        template_str = self._load_template(self.planning_template_path)
        return self._render_template(template_str, variables)

    def _execute_planning_phase(
        self,
        session_dir: Path,
        modification_context: ModificationContext,
        table_access_info: TableAccessInfo,
        mapping_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        """Step 2 (Planning): mapping_info를 기반으로 Data Flow를 분석하고 수정 지침을 생성합니다."""
        logger.info("-" * 40)
        logger.info("[Step 2] Planning 시작...")
        logger.info(f"소스 파일 수: {len(modification_context.file_paths)}")

        # 프롬프트 생성
        prompt = self._create_planning_prompt(
            modification_context, table_access_info, mapping_info
        )
        logger.debug(f"Planning 프롬프트 길이: {len(prompt)} chars")

        # 프롬프트 저장 (LLM 호출 직전)
        self._save_prompt_to_file(prompt, modification_context, "planning")

        # LLM 호출
        response = self.analysis_provider.call(prompt)
        tokens_used = response.get("tokens_used", 0)
        logger.info(f"Planning 응답 완료 (토큰: {tokens_used})")

        # 응답 파싱 (부모 클래스의 범용 메서드 사용)
        modification_instructions = self._parse_json_response(response, "Planning")

        # 결과 저장
        self._save_phase_result(
            session_dir=session_dir,
            modification_context=modification_context,
            step_number=2,
            phase_name="planning",
            result=modification_instructions,
            tokens_used=tokens_used,
        )

        # 요약 로깅
        instruction_count = len(
            modification_instructions.get("modification_instructions", [])
        )
        logger.info(f"생성된 수정 지침 수: {instruction_count}")

        return modification_instructions, tokens_used
