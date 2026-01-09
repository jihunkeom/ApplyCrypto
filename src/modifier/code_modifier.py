"""
Code Modifier 메인 모듈

LLM을 활용하여 소스 코드에 암호화/복호화 코드를 자동으로 적용하는 메인 클래스입니다.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config.config_manager import Configuration
from models.code_generator import CodeGeneratorInput
from models.modification_context import ModificationContext
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo

from .code_generator.code_generator_factory import CodeGeneratorFactory
from .code_generator.base_code_generator import BaseCodeGenerator
from .context_generator.context_generator_factory import ContextGeneratorFactory
from .code_patcher import CodePatcher
from .error_handler import ErrorHandler
from .llm.llm_factory import create_llm_provider
from .llm.llm_provider import LLMProvider
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
    ):
        """
        CodeModifier 초기화

        Args:
            config: 설정 객체
            llm_provider: LLM 프로바이더 (선택적, 설정에서 자동 생성)
        """
        self.config = config
        self.target_project = Path(config.target_project)

        # LLM 프로바이더 초기화
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            llm_provider_name = config.llm_provider
            self.llm_provider = create_llm_provider(provider_name=llm_provider_name)

        # CodeGenerator 생성
        self.code_generator = CodeGeneratorFactory.create(
            config=self.config,
            llm_provider=self.llm_provider,
        )

        # ContextGenerator 생성
        self.context_generator = ContextGeneratorFactory.create(
            config=self.config,
            code_generator=self.code_generator,
        )

        self.code_patcher = CodePatcher(
            project_root=self.target_project, config=self.config
        )
        self.error_handler = ErrorHandler(max_retries=config.max_retries)
        self.result_tracker = ResultTracker(self.target_project)

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

    # def modify_sources(
    #     self, table_access_info: TableAccessInfo, dry_run: bool = False
    # ) -> Dict[str, Any]:
    #     """
    #     소스 파일들을 수정합니다.

    #     Args:
    #         table_access_info: 테이블 접근 정보
    #         dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)

    #     Returns:
    #         Dict[str, Any]: 수정 결과
    #     """
    #     logger.info(f"소스 파일 수정 시작: {table_access_info.table_name}")
    #     self.result_tracker.start_tracking()

    #     try:
    #         # 1. 수정 계획 생성
    #         plans: List[ModificationPlan] = self.generate_modification_plans(
    #             table_access_info
    #         )

    #         all_modifications = []

    #         # 2. 계획 적용
    #         for plan in plans:
    #             result = self.apply_modification_plan(plan, dry_run=dry_run)
    #             all_modifications.append(result)

    #         # 결과 추적
    #         self.result_tracker.end_tracking()
    #         self.result_tracker.update_table_access_info(
    #             table_access_info, all_modifications
    #         )

    #         # 수정 이력 저장
    #         self.result_tracker.save_modification_history(
    #             table_access_info.table_name, all_modifications
    #         )

    #         # 통계 저장
    #         self.result_tracker.save_statistics()

    #         logger.info(
    #             f"소스 파일 수정 완료: {table_access_info.table_name} "
    #             f"({len(all_modifications)}개 파일 수정)"
    #         )

    #         return {
    #             "success": True,
    #             "modifications": all_modifications,
    #             "statistics": self.result_tracker.get_statistics(),
    #         }

    #     except Exception as e:
    #         logger.error(f"소스 파일 수정 실패: {e}")
    #         self.result_tracker.end_tracking()
    #         return {
    #             "success": False,
    #             "error": str(e),
    #             "statistics": self.result_tracker.get_statistics(),
    #         }

    

    def generate_contexts(
        self, table_access_info: TableAccessInfo
    ) -> List[ModificationContext]:
        """
        수정 컨텍스트를 생성합니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            List[ModificationContext]: 수정 컨텍스트 리스트
        """
        return self.context_generator.generate(
            layer_files=table_access_info.layer_files,
            table_name=table_access_info.table_name,
            columns=table_access_info.columns,
        )

    def generate_plan(
        self, modification_context: ModificationContext
    ) -> List[ModificationPlan]:
        """
        수정 계획을 생성합니다.

        Args:
            modification_context: 수정 컨텍스트

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        return self.code_generator.generate_modification_plan(modification_context)

    def apply_plan(
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
        modified_code = plan.modified_code
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
                diff=modified_code,
                error=error_msg,
                tokens_used=tokens_used,
                reason=reason,
            )

        if not modified_code:
            # Code가 없으면 스킵
            return self.result_tracker.record_modification(
                file_path=file_path_str,
                layer=layer_name,
                modification_type=modification_type,
                status="skipped",
                diff=None,
                error="No modified code generated",
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
                file_path=file_path, modified_code=modified_code, dry_run=dry_run
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
                diff=modified_code if status == "success" else None,
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

