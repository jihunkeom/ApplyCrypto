from dataclasses import dataclass
from typing import List, Literal

from .table_access_info import TableAccessInfo


@dataclass
class CodeSnippet:
    path: str
    content: str


@dataclass
class ModificationContext:
    code_snippets: List[CodeSnippet]
    table_access_info: TableAccessInfo
    file_count: int
    layer: Literal["service", "mapper"]
