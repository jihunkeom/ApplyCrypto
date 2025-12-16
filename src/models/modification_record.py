"""
ModificationRecord 데이터 모델

코드 수정 내역을 저장하는 데이터 모델입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class ModificationRecord:
    """
    코드 수정 내역을 저장하는 데이터 모델

    Attributes:
        file_path: 수정된 파일 경로
        table_name: 관련 테이블명
        column_name: 관련 칼럼명
        modified_methods: 수정된 메서드 목록
        added_imports: 추가된 import 문 목록
        timestamp: 수정 일시
        status: 수정 상태 (success, failed, skipped)
        error_message: 에러 메시지 (실패 시)
        diff: 수정 전후 코드 diff (선택적)
    """

    file_path: str
    table_name: str
    column_name: str
    modified_methods: List[str]
    added_imports: List[str]
    timestamp: datetime
    status: str = "success"
    error_message: Optional[str] = None
    diff: Optional[str] = None

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            "file_path": self.file_path,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "modified_methods": self.modified_methods,
            "added_imports": self.added_imports,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "error_message": self.error_message,
            "diff": self.diff,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModificationRecord":
        """딕셔너리로부터 ModificationRecord 객체 생성"""
        # datetime 객체가 이미 변환되었는지 확인
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            file_path=data["file_path"],
            table_name=data["table_name"],
            column_name=data["column_name"],
            modified_methods=data.get("modified_methods", []),
            added_imports=data.get("added_imports", []),
            timestamp=timestamp,
            status=data.get("status", "success"),
            error_message=data.get("error_message"),
            diff=data.get("diff"),
        )
