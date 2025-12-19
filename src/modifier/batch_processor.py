"""
Batch Processor 모듈

병렬로 작업을 처리하는 범용 배치 처리 모듈입니다.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, TypeVar

from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """
    범용 배치 처리 클래스

    주어진 아이템 리스트를 병렬로 처리합니다.
    """

    def __init__(self, max_workers: int = 4):
        """
        BatchProcessor 초기화

        Args:
            max_workers: 병렬 처리 최대 워커 수 (기본값: 4)
        """
        self.max_workers = max_workers

    def process_items_parallel(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        show_progress: bool = True,
        desc: str = "배치 처리 중",
    ) -> List[R]:
        """
        아이템 리스트를 병렬로 처리합니다.
        결과는 입력 아이템의 순서와 동일하게 반환됩니다.

        Args:
            items: 처리할 아이템 리스트
            process_func: 각 아이템을 처리할 함수
            show_progress: 진행 상황 표시 여부
            desc: 진행 상황 설명

        Returns:
            List[R]: 처리 결과 리스트 (입력 순서 보장)
        """
        if not items:
            return []

        results = [None] * len(items)

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Future 객체와 아이템 인덱스 매핑
            future_to_index = {
                executor.submit(process_func, item): i for i, item in enumerate(items)
            }

            # 완료된 작업 처리 (진행 상황 표시용)
            completed_iter = as_completed(future_to_index)
            if show_progress:
                completed_iter = tqdm(completed_iter, total=len(items), desc=desc)

            for future in completed_iter:
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    logger.debug(f"아이템 {index + 1}/{len(items)} 처리 완료")
                except Exception as e:
                    logger.error(f"아이템 {index + 1} 처리 실패: {e}")
                    # 에러 발생 시에도 None이 아닌 의미있는 값이나 예외를 처리해야 할 수 있음
                    # 현재 구조에서는 None으로 남겨둠 (호출 측에서 처리 필요할 수 있음)
                    results[index] = None

        return results
