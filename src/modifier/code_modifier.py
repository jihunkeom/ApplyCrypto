"""
Code Modifier 메인 모듈

LLM을 활용하여 소스 코드에 암호화/복호화 코드를 자동으로 적용하는 메인 클래스입니다.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config_manager import Configuration
from models.diff_generator import DiffGeneratorInput
from models.modification_context import CodeSnippet, ModificationContext
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo

from .batch_processor import BatchProcessor
from .code_patcher import CodePatcher
from .diff_generator import BaseDiffGenerator
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

        self.diff_generator = self._get_diff_generator()

        self.batch_processor = BatchProcessor(
            max_workers=config.max_workers,
        )
        self.code_patcher = CodePatcher(project_root=self.project_root)
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

    def _get_diff_generator(self) -> BaseDiffGenerator:
        """
        DiffGenerator 인스턴스를 가져옵니다.

        Returns:
            BaseDiffGenerator: DiffGenerator 인스턴스
        """
        diff_type = self.config.diff_gen_type

        try:
            # 동적 임포트 경로 생성
            # 예: .diff_generator.mybatis_service.mybatis_service_diff_generator
            module_path = f".diff_generator.{diff_type}.{diff_type}_diff_generator"

            # 현재 패키지를 기준으로 모듈 임포트
            module = importlib.import_module(module_path, package=__package__)

            # 모듈 내에서 BaseDiffGenerator를 상속받는 클래스 탐색
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseDiffGenerator)
                    and obj is not BaseDiffGenerator
                    and obj.__module__ == module.__name__
                ):
                    return obj(llm_provider=self.llm_provider)

            raise ValueError(f"No suitable DiffGenerator class found in {module_path}")

        except ImportError as e:
            logger.error(f"Failed to import generator module for {diff_type}: {e}")
            raise ValueError(f"Invalid diff_type or module not found: {diff_type}")
        except Exception as e:
            logger.error(f"Error loading generator for {diff_type}: {e}")
            raise

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
            # 레이어별로 파일 그룹화
            layer_files = table_access_info.layer_files

            # 모든 레이어의 배치를 수집
            all_batches: List[ModificationContext] = []

            # 각 레이어별로 처리
            for layer_name, file_paths in layer_files.items():
                if not file_paths:
                    continue

                logger.info(
                    f"레이어 '{layer_name}' 계획 생성 준비: {len(file_paths)}개 파일"
                )

                # 파일 내용 읽기 (항상 절대 경로 사용)
                code_snippets: List[CodeSnippet] = []
                for file_path in file_paths:
                    try:
                        # 파일 경로를 절대 경로로 변환
                        file_path_obj = Path(file_path)
                        if not file_path_obj.is_absolute():
                            # 상대 경로인 경우 project_root와 결합
                            full_path = self.project_root / file_path_obj
                        else:
                            # 절대 경로인 경우 그대로 사용
                            full_path = file_path_obj

                        # 절대 경로로 정규화
                        full_path = full_path.resolve()

                        if full_path.exists():
                            with open(full_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            code_snippets.append(
                                CodeSnippet(
                                    path=str(full_path),  # 절대 경로 저장
                                    content=content,
                                )
                            )
                        else:
                            logger.warning(
                                f"파일이 존재하지 않습니다: {full_path} (원본 경로: {file_path})"
                            )
                            # 파일이 없으면 'failed' 상태의 더미 계획 추가
                            plans.append(
                                ModificationPlan(
                                    file_path=str(full_path),
                                    layer_name=layer_name,
                                    modification_type="encryption",
                                    status="failed",
                                    error="File not found",
                                )
                            )

                    except Exception as e:
                        logger.error(f"파일 읽기 실패: {file_path} - {e}")
                        plans.append(
                            ModificationPlan(
                                file_path=str(file_path),
                                layer_name=layer_name,
                                modification_type="encryption",
                                status="failed",
                                error=str(e),
                            )
                        )

                # 배치를 생성하여 토큰 제한 고려
                modification_context_batches: List[ModificationContext] = (
                    self.create_batches(
                        code_snippets=code_snippets,
                        table_access_info=table_access_info,
                        layer=layer_name,
                    )
                )

                all_batches.extend(modification_context_batches)

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
            )

        if not unified_diff:
            # Diff가 없으면 스킵
            return self.result_tracker.record_modification(
                file_path=file_path_str,
                layer=layer_name,
                modification_type=modification_type,
                status="skipped",
                diff=None,
                error=reason if reason else "No diff generated",
                tokens_used=tokens_used,
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
            parsed_modifications = self.code_patcher.parse_llm_response(diff_out)

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

    def create_batches(
        self,
        code_snippets: List[CodeSnippet],
        table_access_info: TableAccessInfo,
        layer: str,
    ) -> List[ModificationContext]:
        """
        파일 목록을 토큰 크기 기반으로 배치로 분할합니다.

        Args:
            code_snippets: 코드 스니펫 리스트
            table_access_info: 테이블 접근 정보
            layer: 레이어 이름

        Returns:
            List[ModificationContext]: 분할된 수정 컨텍스트 리스트
        """
        if not code_snippets:
            return []

        batches: List[ModificationContext] = []
        current_snippets: List[CodeSnippet] = []

        # 기본 정보 준비 (반복문 밖에서 한 번만 처리)
        formatted_table_info = self._format_table_info(table_access_info)
        max_tokens = self.config.max_tokens_per_batch

        input_empty_data = DiffGeneratorInput(
            code_snippets=[], table_info=formatted_table_info, layer_name=layer
        )

        # 빈 프롬프트 생성 및 토큰 계산
        empty_prompt = self.diff_generator.create_prompt(input_empty_data)
        empty_num_tokens = self.diff_generator.calculate_token_size(empty_prompt)

        # 스니펫 간 구분자 토큰 계산 ("\n\n")
        separator_tokens = self.diff_generator.calculate_token_size("\n\n")

        current_batch_tokens = empty_num_tokens

        for snippet in code_snippets:
            # 스니펫 포맷팅 및 토큰 계산
            snippet_formatted = (
                f"=== File Path (Absolute): {snippet.path} ===\n{snippet.content}"
            )
            snippet_tokens = self.diff_generator.calculate_token_size(snippet_formatted)

            # 첫 번째 스니펫이 아니면 구분자 추가
            tokens_to_add = snippet_tokens
            if current_snippets:
                tokens_to_add += separator_tokens

            if current_snippets and (current_batch_tokens + tokens_to_add) > max_tokens:
                # 현재 배치를 저장하고 새로운 배치 시작
                batches.append(
                    ModificationContext(
                        code_snippets=current_snippets,
                        table_access_info=table_access_info,
                        file_count=len(current_snippets),
                        layer=layer,
                    )
                )
                current_snippets = [snippet]
                current_batch_tokens = empty_num_tokens + snippet_tokens
            else:
                # 현재 배치에 추가
                current_snippets.append(snippet)
                current_batch_tokens += tokens_to_add

        # 마지막 배치 추가
        if current_snippets:
            batches.append(
                ModificationContext(
                    code_snippets=current_snippets,
                    table_access_info=table_access_info,
                    file_count=len(current_snippets),
                    layer=layer,
                )
            )

        logger.info(
            f"총 {len(code_snippets)}개 파일을 {len(batches)}개 배치(ModificationContext)로 분할했습니다."
        )
        return batches
