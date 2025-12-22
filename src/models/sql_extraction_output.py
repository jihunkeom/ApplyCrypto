from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.source_file import SourceFile


@dataclass
class ExtractedSQLQuery:
    """
    Extracted SQL query information data model

    Attributes:
        id: Query identifier (MyBatis: query id, JDBC/JPA: method name, etc.)
        query_type: Query type (SELECT, INSERT, UPDATE, DELETE)
        sql: SQL query string
        strategy_specific: Strategy specific information (dict)
    """

    id: str
    query_type: str
    sql: str
    strategy_specific: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query_type": self.query_type,
            "sql": self.sql,
            "strategy_specific": self.strategy_specific,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedSQLQuery":
        return cls(
            id=data.get("id", ""),
            query_type=data.get("query_type", "SELECT"),
            sql=data.get("sql", ""),
            strategy_specific=data.get("strategy_specific", {}),
        )


@dataclass
class SQLExtractionOutput:
    """
    SQL extraction output data model
    Contains the source file and the list of extracted SQL queries from that file.

    Attributes:
        file: The source file object
        sql_queries: List of extracted SQL queries
    """

    file: SourceFile
    sql_queries: List[ExtractedSQLQuery]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file.to_dict(),
            "sql_queries": [q.to_dict() for q in self.sql_queries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SQLExtractionOutput":
        return cls(
            file=SourceFile.from_dict(data.get("file", {})),
            sql_queries=[
                ExtractedSQLQuery.from_dict(q) for q in data.get("sql_queries", [])
            ],
        )
