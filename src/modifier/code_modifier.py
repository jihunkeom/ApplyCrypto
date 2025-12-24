"""
Code Modifier 메인 모듈

LLM을 활용하여 소스 코드에 암호화/복호화 코드를 자동으로 적용하는 메인 클래스입니다.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config_manager import Configuration
from models.diff_generator import DiffGeneratorInput
from models.modification_context import ModificationContext
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo

from .batch_processor import BatchProcessor
from .code_patcher import CodePatcher
from .diff_generator import DiffGeneratorFactory
from .error_handler import ErrorHandler
from .llm.llm_factory import create_llm_provider
from .llm.llm_provider import LLMProvider
from .modification_context_generator import ModificationContextGeneratorFactory
from .result_tracker import ResultTracker

logger = logging.getLogger("applycrypto.code_patcher")


class CodeModifier:
    """
    Code Modifier 메인 클래스

    설정 파일의 테이블/칼럼 정보를 기반으로 식별된 DB 접근 소스 파일들에 대해
    k-sign.CryptoService의 encrypt/decrypt를 사용하여 암호화 코드를 자동으로 적용합니다.
    """

    def __init__(
        self,
        config: Configuration,
        llm_provider: Optional[LLMProvider] = None,
        project_root: Optional[Path] = None,
    ):
        """
        CodeModifier 초기화

        Args:
            config: 설정 객체
            llm_provider: LLM 프로바이더 (선택적, 설정에서 자동 생성)
            project_root: 프로젝트 루트 디렉토리 (선택적)
        """
        self.config = config
        self.project_root = (
            Path(project_root) if project_root else Path(config.target_project)
        )

        # LLM 프로바이더 초기화
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            llm_provider_name = config.llm_provider
            self.llm_provider = create_llm_provider(provider_name=llm_provider_name)

        self.diff_generator = DiffGeneratorFactory.create(
            config=self.config, llm_provider=self.llm_provider
        )
        self.modification_context_generator = (
            ModificationContextGeneratorFactory.create(
                config=self.config,
                diff_generator=self.diff_generator,
                generator_type="parallel",
            )
        )

        self.batch_processor = BatchProcessor(
            max_workers=config.max_workers,
        )
        self.code_patcher = CodePatcher(
            project_root=self.project_root, config=self.config
        )
        self.error_handler = ErrorHandler(max_retries=config.max_retries)
        self.result_tracker = ResultTracker()

        logger.info(
            f"CodeModifier 초기화 완료: {self.llm_provider.get_provider_name()}"
        )

    def _get_api_key_from_env(self, provider_name: str) -> Optional[str]:
        """
        환경변수에서 API 키를 가져옵니다.

        Args:
            provider_name: 프로바이더 이름

        Returns:
            Optional[str]: API 키
        """
        import os

        if provider_name.lower() in ["watsonx_ai", "watsonx"]:
            return os.getenv("WATSONX_API_KEY") or os.getenv("IBM_API_KEY")
        elif provider_name.lower() == "openai":
            return os.getenv("OPENAI_API_KEY")

        return None

    def modify_sources(
        self, table_access_info: TableAccessInfo, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        소스 파일들을 수정합니다.

        Args:
            table_access_info: 테이블 접근 정보
            dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)

        Returns:
            Dict[str, Any]: 수정 결과
        """
        logger.info(f"소스 파일 수정 시작: {table_access_info.table_name}")
        self.result_tracker.start_tracking()

        try:
            # 1. 수정 계획 생성
            plans: List[ModificationPlan] = self.generate_modification_plans(
                table_access_info
            )

            all_modifications = []

            # 2. 계획 적용
            for plan in plans:
                result = self.apply_modification_plan(plan, dry_run=dry_run)
                all_modifications.append(result)

            # 결과 추적
            self.result_tracker.end_tracking()
            self.result_tracker.update_table_access_info(
                table_access_info, all_modifications
            )

            # 수정 이력 저장
            self.result_tracker.save_modification_history(
                table_access_info.table_name, all_modifications
            )

            # 통계 저장
            self.result_tracker.save_statistics()

            logger.info(
                f"소스 파일 수정 완료: {table_access_info.table_name} "
                f"({len(all_modifications)}개 파일 수정)"
            )

            return {
                "success": True,
                "modifications": all_modifications,
                "statistics": self.result_tracker.get_statistics(),
            }

        except Exception as e:
            logger.error(f"소스 파일 수정 실패: {e}")
            self.result_tracker.end_tracking()
            return {
                "success": False,
                "error": str(e),
                "statistics": self.result_tracker.get_statistics(),
            }

    def generate_modification_plans(
        self, table_access_info: TableAccessInfo
    ) -> List[ModificationPlan]:
        """
        수정 계획을 생성합니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            List[Dict[str, Any]]: 수정 계획 리스트
        """
        plans = []

        try:
            # ModificationContextGenerator를 사용하여 배치 생성
            all_batches = self.modification_context_generator.generate(
                table_access_info
            )

            # 병렬 배치 처리 함수 정의
            def process_context_wrapper(
                context: ModificationContext,
            ) -> List[ModificationPlan]:
                if not context.code_snippets:
                    return []
                return self._generate_batch_plans(
                    modification_context=context,
                    template_type="default",
                    modification_type="encryption",
                )

            # 병렬 처리 실행
            plan_batches = self.batch_processor.process_items_parallel(
                items=all_batches,
                process_func=process_context_wrapper,
                desc="파일 수정 계획 생성 중",
            )

            # 결과 통합
            for plan_batch in plan_batches:
                if plan_batch:
                    plans.extend(plan_batch)

            return plans

        except Exception as e:
            logger.error(f"계획 생성 실패: {e}")
            raise

    def apply_modification_plan(
        self, plan: ModificationPlan, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        수정 계획을 적용합니다.

        Args:
            plan: 수정 계획
            dry_run: 시뮬레이션 모드

        Returns:
            Dict[str, Any]: 적용 결과
        """
        file_path_str = plan.file_path
        unified_diff = plan.unified_diff
        status = plan.status
        error_msg = plan.error
        layer_name = plan.layer_name
        modification_type = plan.modification_type
        tokens_used = plan.tokens_used
        reason = plan.reason or ""

        # 이미 실패했거나 스킵된 계획인 경우 그대로 반환 기록
        if status in ["failed", "skipped"]:
            return self.result_tracker.record_modification(
                file_path=file_path_str,
                layer=layer_name,
                modification_type=modification_type,
                status=status,
                diff=unified_diff,
                error=error_msg,
                tokens_used=tokens_used,
                reason=reason,
            )

        if not unified_diff:
            # Diff가 없으면 스킵
            return self.result_tracker.record_modification(
                file_path=file_path_str,
                layer=layer_name,
                modification_type=modification_type,
                status="skipped",
                diff=None,
                error="No diff generated",
                tokens_used=tokens_used,
                reason=reason,
            )

        try:
            file_path = Path(file_path_str)

            # 파일 백업
            if not dry_run:
                self.error_handler.backup_file(file_path)

            # 패치 적용
            success, error = self.code_patcher.apply_patch(
                file_path=file_path, unified_diff=unified_diff, dry_run=dry_run
            )

            if success:
                status = "success"
                error_msg = None
            else:
                status = "failed"
                error_msg = error
                # 롤백은 apply_patch 내부나 error_handler에서 처리하지 않으므로 여기서 복구
                if not dry_run:
                    self.error_handler.restore_file(file_path)

            # 수정 정보 기록
            return self.result_tracker.record_modification(
                file_path=str(file_path),
                layer=layer_name,
                modification_type=modification_type,
                status=status,
                diff=unified_diff if status == "success" else None,
                error=error_msg,
                tokens_used=tokens_used,
                reason=reason,
            )

        except Exception as e:
            logger.error(f"패치 적용 실패: {file_path_str} - {e}")
            return self.result_tracker.record_modification(
                file_path=file_path_str,
                layer=layer_name,
                modification_type=modification_type,
                status="failed",
                error=str(e),
                tokens_used=tokens_used,
                reason=reason,
            )

    def _generate_batch_plans(
        self,
        modification_context: ModificationContext,
        template_type: str,
        modification_type: str,
    ) -> List[ModificationPlan]:
        """
        배치 처리를 통해 수정 계획을 생성합니다.

        Args:
            modification_context: 수정 컨텍스트 (배치 정보 포함)
            template_type: 템플릿 타입
            modification_type: 수정 타입

        Returns:
            List[Dict[str, Any]]: 수정 계획 리스트
        """
        plans = []
        batch = modification_context.code_snippets
        layer_name = modification_context.layer
        table_access_info = modification_context.table_access_info

        # 배치 처리
        try:
            # variables에서 table_info와 layer_name 추출
            table_info_str = self._format_table_info(table_access_info)

            # extra_variables 준비
            extra_vars = {"file_count": len(batch)}

            input_data = DiffGeneratorInput(
                code_snippets=batch,
                table_info=table_info_str,
                layer_name=layer_name,
                extra_variables=extra_vars,
            )

            # Diff 생성
            diff_out = self.diff_generator.generate(input_data)

            tokens_used = diff_out.tokens_used

            # LLM 응답 파싱
            parsed_modifications = diff_out.parsed_out
            if parsed_modifications is None:
                raise ValueError(
                    "LLM Response parsing failed (parsed_out is None). Check logs for details."
                )

            # 각 수정 사항에 대해 계획 생성
            for mod in parsed_modifications:
                file_path_str = mod.get("file_path", "")
                reason = mod.get("reason", "")
                unified_diff = mod.get("unified_diff", "")

                # LLM 응답에서 받은 절대 경로를 그대로 사용
                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    logger.warning(
                        f"LLM 응답에 상대 경로가 포함되었습니다: {file_path_str}. 절대 경로로 변환합니다."
                    )
                    file_path = self.project_root / file_path

                # 절대 경로로 정규화
                file_path = file_path.resolve()

                # 계획 객체 생성
                plan = ModificationPlan(
                    file_path=str(file_path),
                    layer_name=layer_name,
                    modification_type=modification_type,
                    unified_diff=unified_diff,
                    reason=reason,
                    tokens_used=tokens_used,
                    status="pending",
                )

                # unified_diff가 빈 문자열인 경우 스킵 상태로 설정
                if not unified_diff or unified_diff.strip() == "":
                    logger.info(
                        f"파일 수정 건너뜀 (계획): {file_path} (이유: {reason})"
                    )
                    plan.status = "skipped"

                plans.append(plan)

        except Exception as e:
            logger.error(f"배치 계획 생성 실패: {e}")
            # 배치 내 모든 파일에 대해 실패 계획 기록
            for file_info in batch:
                plans.append(
                    ModificationPlan(
                        file_path=file_info.path,
                        layer_name=layer_name,
                        modification_type=modification_type,
                        status="failed",
                        error=str(e),
                        tokens_used=0,
                    )
                )

        return plans

    def _format_table_info(self, table_access_info: TableAccessInfo) -> str:
        """
        테이블 정보를 JSON 형식으로 포맷팅합니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            str: JSON 형식의 테이블 정보
        """
        import json

        table_info = {
            "table_name": table_access_info.table_name,
            "columns": table_access_info.columns,
        }

        return json.dumps(table_info, indent=2, ensure_ascii=False)
