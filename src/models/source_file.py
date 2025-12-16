"""
SourceFile 데이터 모델

소스 파일의 메타데이터를 저장하는 데이터 모델입니다.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class SourceFile:
    """
    소스 파일 메타데이터를 저장하는 데이터 모델

    Attributes:
        path: 파일의 절대 경로
        relative_path: 프로젝트 루트 기준 상대 경로
        filename: 파일명 (확장자 포함)
        extension: 파일 확장자 (예: .java, .xml)
        size: 파일 크기 (바이트)
        modified_time: 파일 수정 시간
        tags: 파일에 부여된 태그 목록 (예: 테이블명)
    """

    path: Path
    relative_path: Path
    filename: str
    extension: str
    size: int
    modified_time: datetime
    tags: List[str]

    def __post_init__(self):
        """데이터 검증 및 타입 변환"""
        # Path 객체로 변환
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if isinstance(self.relative_path, str):
            self.relative_path = Path(self.relative_path)

        # tags가 None이면 빈 리스트로 초기화
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict:
        """
        딕셔너리 형태로 변환

        Returns:
            dict: SourceFile의 딕셔너리 표현
        """
        return {
            "path": str(self.path),
            "relative_path": str(self.relative_path),
            "filename": self.filename,
            "extension": self.extension,
            "size": self.size,
            "modified_time": self.modified_time.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceFile":
        """
        딕셔너리로부터 SourceFile 객체 생성

        Args:
            data: SourceFile 데이터 딕셔너리

        Returns:
            SourceFile: 생성된 SourceFile 객체
        """
        # Path 객체가 이미 변환되었는지 확인
        path = data["path"]
        if not isinstance(path, Path):
            path = Path(path)

        relative_path = data["relative_path"]
        if not isinstance(relative_path, Path):
            relative_path = Path(relative_path)

        # datetime 객체가 이미 변환되었는지 확인
        modified_time = data["modified_time"]
        if isinstance(modified_time, str):
            modified_time = datetime.fromisoformat(modified_time)

        return cls(
            path=path,
            relative_path=relative_path,
            filename=data["filename"],
            extension=data["extension"],
            size=data["size"],
            modified_time=modified_time,
            tags=data.get("tags", []),
        )
