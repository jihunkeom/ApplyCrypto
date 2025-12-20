from dataclasses import dataclass
from typing import Any, Dict, List

from models.modification_context import CodeSnippet


@dataclass
class DiffGeneratorOutput:
    content: str
    tokens_used: int = 0


@dataclass
class DiffGeneratorInput:
    """Diff 생성기 입력 데이터"""

    code_snippets: List["CodeSnippet"]
    """
    소스 파일 리스트
    [CodeSnippet(path="/abs/path/to/file.py", content="..."), ...]
    """

    table_info: str
    """테이블 스키마 정보 (JSON string or formatted string)"""

    layer_name: str
    """현재 처리 중인 레이어 이름 (Service, Controller 등)"""

    extra_variables: Dict[str, Any] = None
    """기타 템플릿 변수"""

    def __post_init__(self):
        if self.extra_variables is None:
            self.extra_variables = {}
