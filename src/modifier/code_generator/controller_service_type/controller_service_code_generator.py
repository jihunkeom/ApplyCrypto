"""
Controller/Service Code Generator

Controller/Service 레이어를 대상으로 코드 수정을 수행하는 CodeGenerator입니다.
기존 ControllerOrServiceStrategy의 로직을 이관했습니다.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from config.config_manager import Configuration
from models.code_generator import CodeGeneratorInput, CodeGeneratorOutput
from models.modification_context import ModificationContext
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo
from modifier.batch_processor import BatchProcessor
from modifier.context_generator import ContextGeneratorFactory
from modifier.llm.llm_provider import LLMProvider

from ..base_code_generator import BaseCodeGenerator

logger = logging.getLogger("applycrypto")


class ControllerOrServiceCodeGenerator(BaseCodeGenerator):
    """Controller/Service Code 생성기"""

    def create_file_mapping(self, input_data: CodeGeneratorInput) -> Dict[str, str]:
        """
        파일명 -> 절대 경로 매핑을 생성합니다.

        Args:
            input_data: Diff 생성 입력

        Returns:
            Dict[str, str]: 파일명 -> 절대 경로 매핑
        """
        return {
            Path(snippet.path).name: snippet.path
            for snippet in input_data.code_snippets
        }

    def generate(self, input_data: CodeGeneratorInput) -> CodeGeneratorOutput:
        """
        입력 데이터를 바탕으로 Code를 생성합니다.
        BaseCodeGenerator의 기존 구현을 이관했습니다.

        Args:
            input_data: Code 생성 입력

        Returns:
            CodeGeneratorOutput: LLM 응답 (Code 포함)
        """
        prompt = self.create_prompt(input_data)
        file_mapping = self.create_file_mapping(input_data)

        # 캐시 확인
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._prompt_cache:
            logger.debug(f"캐시에서 응답을 가져왔습니다: {cache_key[:50]}...")
            cached_output = self._prompt_cache[cache_key]
            # 캐시된 응답에도 file_mapping 추가
            cached_output.file_mapping = file_mapping
            return cached_output

        # LLM 호출
        try:
            response = self.llm_provider.call(prompt)

            # CodeGeneratorOutput 객체 생성
            output = CodeGeneratorOutput(
                content=response.get("content", ""),
                tokens_used=response.get("tokens_used", 0),
                file_mapping=file_mapping,
            )

            # 파싱 시도 (실패 시 무시 - CallChain 등 다른 포맷일 수 있음)
            try:
                output.parsed_out = self.parse_llm_response(output)
            except Exception as e:
                logger.debug(f"자동 Code 파싱 실패 (포맷 불일치 가능성): {e}")
                output.parsed_out = None

            # 캐시에 저장
            self._prompt_cache[cache_key] = output

            return output
        except Exception as e:
            logger.error(f"Code 생성 중 오류 발생: {e}")
            raise

    def generate_modification_plans(
        self, table_access_info: TableAccessInfo
    ) -> List[ModificationPlan]:
        """
        수정 계획을 생성합니다.
        ControllerOrServiceStrategy의 로직을 이관했습니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        plans = []

        try:
            # ContextGenerator를 사용하여 배치 생성
            context_generator = ContextGeneratorFactory.create(
                config=self.config,
                code_generator=self,
            )
            all_batches = context_generator.generate(
                table_access_info=table_access_info,
            )

            # BatchProcessor 초기화
            # batch_processor = BatchProcessor(max_workers=self.config.max_workers)

            # 병렬 처리 실행
            # plan_batches = batch_processor.process_items_parallel(
            #     items=all_batches,
            #     process_func=self._generate_batch_plans,
            #     desc="파일 수정 계획 생성 중",
            # )

            # 결과 통합
            # for plan_batch in plan_batches:
            #     if plan_batch:
            #         plans.extend(plan_batch)

            # 순차 처리: all_batches의 각 item을 순차적으로 처리
            for batch in tqdm(all_batches, desc="파일 수정 계획 생성 중", unit="batch"):
                plan_batch = self._generate_batch_plans(batch)
                if plan_batch:
                    plans.extend(plan_batch)

            return plans

        except Exception as e:
            logger.error(f"계획 생성 실패: {e}")
            raise

    def _generate_batch_plans(
        self, modification_context: ModificationContext
    ) -> List[ModificationPlan]:
        """
        배치 처리를 통해 수정 계획을 생성합니다.
        ControllerOrServiceStrategy의 로직을 이관했습니다.

        Args:
            modification_context: 수정 컨텍스트 (배치 정보 포함)

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        # code_snippets가 비어있으면 빈 리스트 반환
        if not modification_context.code_snippets:
            return []

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

            input_data = CodeGeneratorInput(
                code_snippets=batch,
                table_info=table_info_str,
                layer_name=layer_name,
                extra_variables=extra_vars,
            )

            logger.info("--------------------------------")
            logger.info("_generate_batch_plans():")
            for code_snippet in batch:
                logger.info(f"code_snippet.path: {code_snippet.path}")
            logger.info("--------------------------------")
            # Code 생성
            code_out = self.generate(input_data)

            tokens_used = code_out.tokens_used

            # LLM 응답 파싱
            parsed_modifications = code_out.parsed_out
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
                    file_path = Path(self.config.target_project) / file_path

                # 절대 경로로 정규화
                file_path = file_path.resolve()

                # 계획 객체 생성
                plan = ModificationPlan(
                    file_path=str(file_path),
                    layer_name=layer_name,
                    modification_type="encryption",
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
                        modification_type="encryption",
                        status="failed",
                        error=str(e),
                        tokens_used=0,
                    )
                )

        return plans

    def _format_table_info(self, table_access_info: TableAccessInfo) -> str:
        """
        테이블 정보를 JSON 형식으로 포맷팅합니다.
        ControllerOrServiceStrategy의 로직을 이관했습니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            str: JSON 형식의 테이블 정보
        """
        table_info = {
            "table_name": table_access_info.table_name,
            "columns": table_access_info.columns,
        }

        return json.dumps(table_info, indent=2, ensure_ascii=False)

    def _get_cache_key(self, prompt: str) -> str:
        """프롬프트의 캐시 키를 생성합니다."""
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()
