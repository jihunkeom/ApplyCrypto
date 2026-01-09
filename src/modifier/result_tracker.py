"""
Result Tracker 모듈

수정 결과를 추적하고 통계를 계산하는 모듈입니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.table_access_info import TableAccessInfo

logger = logging.getLogger(__name__)


class ResultTracker:
    """
    결과 추적 클래스

    수정 이력을 저장하고 통계를 계산합니다.
    """

    def __init__(self, target_project: Path):
        """
        ResultTracker 초기화

        Args:
            target_project: 대상 프로젝트 루트 경로
        """
        self.output_dir = target_project / ".applycrypto" / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 통계 정보
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "processing_time": 0.0,
            "start_time": None,
            "end_time": None,
        }

    def start_tracking(self):
        """추적 시작"""
        self.stats["start_time"] = datetime.now().isoformat()
        logger.info("결과 추적 시작")

    def end_tracking(self):
        """추적 종료"""
        self.stats["end_time"] = datetime.now().isoformat()

        if self.stats["start_time"]:
            start = datetime.fromisoformat(self.stats["start_time"])
            end = datetime.fromisoformat(self.stats["end_time"])
            self.stats["processing_time"] = (end - start).total_seconds()

        logger.info(f"결과 추적 종료: {self.stats['processing_time']:.2f}초")

    def record_modification(
        self,
        file_path: str,
        layer: str,
        modification_type: str,
        status: str,
        diff: Optional[str] = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        파일 수정 정보를 기록합니다.

        Args:
            file_path: 수정된 파일 경로
            layer: 파일 레이어
            modification_type: 수정 타입 ("plaintext" 또는 "partial_encryption")
            status: 수정 상태 ("success" 또는 "failed")
            diff: 수정 전후 차이 (Unified Format, 선택적)
            error: 에러 메시지 (선택적)
            tokens_used: 사용된 토큰 수
            reason: 수정/스킵 이유 (선택적, error와 별개로 기록)

        Returns:
            Dict[str, Any]: 기록된 수정 정보
        """
        modification_info = {
            "file_path": file_path,
            "layer": layer,
            "modification_type": modification_type,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": tokens_used,
        }

        if diff:
            modification_info["diff"] = diff

        if error:
            modification_info["error"] = error

        if reason:
            modification_info["reason"] = reason

        # 통계 업데이트
        self.stats["total_files"] += 1
        if status == "success":
            self.stats["successful_files"] += 1
        else:
            self.stats["failed_files"] += 1

        self.stats["total_tokens"] += tokens_used

        logger.debug(f"수정 정보 기록: {file_path} ({status})")
        return modification_info

    def update_table_access_info(
        self, table_access_info: TableAccessInfo, modifications: List[Dict[str, Any]]
    ):
        """
        TableAccessInfo에 수정 정보를 추가합니다.

        Args:
            table_access_info: 업데이트할 TableAccessInfo 객체
            modifications: 수정 정보 리스트
        """
        table_access_info.modified_files = modifications
        logger.info(
            f"TableAccessInfo 업데이트: {table_access_info.table_name} "
            f"({len(modifications)}개 파일 수정)"
        )

    def save_modification_history(
        self, table_name: str, modifications: List[Dict[str, Any]]
    ) -> Path:
        """
        수정 이력을 JSON 파일로 저장합니다.

        Args:
            table_name: 테이블명
            modifications: 수정 정보 리스트

        Returns:
            Path: 저장된 파일 경로
        """
        history_file = self.output_dir / f"modification_history_{table_name}.json"

        history_data = {
            "table_name": table_name,
            "modifications": modifications,
            "statistics": {
                "total_files": len(modifications),
                "successful_files": sum(
                    1 for m in modifications if m.get("status") == "success"
                ),
                "failed_files": sum(
                    1 for m in modifications if m.get("status") == "failed"
                ),
                "total_tokens": sum(m.get("tokens_used", 0) for m in modifications),
            },
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            logger.info(f"수정 이력 저장 완료: {history_file}")
            return history_file

        except Exception as e:
            logger.error(f"수정 이력 저장 실패: {e}")
            raise

    def save_statistics(self) -> Path:
        """
        전체 통계를 JSON 파일로 저장합니다.

        Returns:
            Path: 저장된 파일 경로
        """
        stats_file = self.output_dir / "modification_statistics.json"

        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)

            logger.info(f"통계 저장 완료: {stats_file}")
            return stats_file

        except Exception as e:
            logger.error(f"통계 저장 실패: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        현재 통계를 반환합니다.

        Returns:
            Dict[str, Any]: 통계 딕셔너리
        """
        return self.stats.copy()
