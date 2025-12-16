"""
Batch Processor 모듈

토큰 크기를 기반으로 동적 배치 분할 및 병렬 처리를 수행하는 모듈입니다.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .prompt_template_manager import PromptTemplateManager

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    배치 처리 클래스

    레이어별/파일 개수별 배치 처리를 수행하고, 토큰 크기를 기반으로 동적 배치 분할을 수행합니다.
    """

    def __init__(
        self,
        template_manager: PromptTemplateManager,
        max_tokens_per_batch: int = 3000,
        max_workers: int = 4,
    ):
        """
        BatchProcessor 초기화

        Args:
            template_manager: 프롬프트 템플릿 매니저
            max_tokens_per_batch: 배치당 최대 토큰 수 (기본값: 3000)
            max_workers: 병렬 처리 최대 워커 수 (기본값: 4)
        """
        self.template_manager = template_manager
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_workers = max_workers

        # 캐시: 프롬프트 -> LLM 응답
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}

    def create_batches(
        self, files: List[Dict[str, Any]], template_type: str, variables: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """
        파일 목록을 토큰 크기 기반으로 배치로 분할합니다.

        Args:
            files: 파일 정보 리스트 [{"path": "...", "content": "..."}, ...]
            template_type: 템플릿 타입
            variables: 템플릿 변수

        Returns:
            List[List[Dict[str, Any]]]: 배치 리스트 (각 배치는 파일 리스트)
        """
        if not files:
            return []

        # 템플릿 로드
        template = self.template_manager.load_template(template_type)

        batches = []
        current_batch = []
        current_tokens = 0

        base_prompt = self.template_manager.render_template(template, {})
        base_tokens = self.template_manager.calculate_token_size(base_prompt)

        for file_info in files:
            # 파일별 프롬프트 생성
            file_variables = {**variables, "source_files": file_info.get("content", "")}
            file_prompt = self.template_manager.render_template(
                template, file_variables
            )
            file_tokens = self.template_manager.calculate_token_size(file_prompt)

            # 배치에 추가 가능한지 확인
            if (
                current_batch
                and (current_tokens + file_tokens) > self.max_tokens_per_batch
            ):
                # 현재 배치 저장 및 새 배치 시작
                batches.append(current_batch)
                current_batch = [file_info]
                current_tokens = base_tokens + file_tokens
            else:
                # 현재 배치에 추가
                current_batch.append(file_info)
                if current_batch:
                    # 배치 전체 프롬프트 크기 재계산
                    batch_variables = {
                        **variables,
                        "source_files": "\n\n".join(
                            [f.get("content", "") for f in current_batch]
                        ),
                    }
                    batch_prompt = self.template_manager.render_template(
                        template, batch_variables
                    )
                    current_tokens = self.template_manager.calculate_token_size(
                        batch_prompt
                    )
                else:
                    current_tokens = base_tokens + file_tokens

        # 마지막 배치 추가
        if current_batch:
            batches.append(current_batch)

        logger.info(f"총 {len(files)}개 파일을 {len(batches)}개 배치로 분할했습니다.")
        return batches

    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        template_type: str,
        variables: Dict[str, Any],
        llm_call_func: Callable[[str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        단일 배치를 처리합니다.

        Args:
            batch: 배치 파일 리스트
            template_type: 템플릿 타입
            variables: 템플릿 변수
            llm_call_func: LLM 호출 함수 (prompt -> response)

        Returns:
            List[Dict[str, Any]]: 처리 결과 리스트
        """
        if not batch:
            return []

        # 배치 프롬프트 생성 (절대 경로 포함)
        batch_variables = {
            **variables,
            "source_files": "\n\n".join(
                [
                    f"=== File Path (Absolute): {f.get('path', '')} ===\n{f.get('content', '')}"
                    for f in batch
                ]
            ),
            "file_count": len(batch),
        }

        template = self.template_manager.load_template(template_type)
        prompt = self.template_manager.render_template(template, batch_variables)

        # 캐시 확인
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._prompt_cache:
            logger.debug(f"캐시에서 응답을 가져왔습니다: {cache_key[:50]}...")
            return self._prompt_cache[cache_key]

        # LLM 호출
        try:
            response = llm_call_func(prompt)

            # 캐시에 저장
            self._prompt_cache[cache_key] = response

            return response
        except Exception as e:
            logger.error(f"배치 처리 중 오류 발생: {e}")
            raise

    def process_batches_parallel(
        self,
        batches: List[List[Dict[str, Any]]],
        template_type: str,
        variables: Dict[str, Any],
        llm_call_func: Callable[[str], Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        여러 배치를 병렬로 처리합니다.

        Args:
            batches: 배치 리스트
            template_type: 템플릿 타입
            variables: 템플릿 변수
            llm_call_func: LLM 호출 함수
            show_progress: 진행 상황 표시 여부

        Returns:
            List[Dict[str, Any]]: 모든 배치의 처리 결과 리스트
        """
        results = []

        # 진행 상황 표시
        if show_progress and TQDM_AVAILABLE:
            batch_iterator = tqdm(batches, desc="배치 처리 중")
        else:
            batch_iterator = batches

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_batch, batch, template_type, variables, llm_call_func
                ): i
                for i, batch in enumerate(batch_iterator)
            }

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"배치 {batch_idx + 1}/{len(batches)} 처리 완료")
                except Exception as e:
                    logger.error(f"배치 {batch_idx + 1} 처리 실패: {e}")
                    # 실패한 배치도 결과에 포함 (에러 정보 포함)
                    results.append({"error": str(e), "batch_index": batch_idx})

        return results

    def _get_cache_key(self, prompt: str) -> str:
        """
        프롬프트의 캐시 키를 생성합니다.

        Args:
            prompt: 프롬프트 문자열

        Returns:
            str: 캐시 키 (해시)
        """
        import hashlib

        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def clear_cache(self):
        """캐시를 비웁니다."""
        self._prompt_cache.clear()
        logger.debug("배치 처리 캐시가 비워졌습니다.")
