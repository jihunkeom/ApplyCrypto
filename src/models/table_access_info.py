"""
TableAccessInfo 데이터 모델

데이터베이스 테이블 접근 정보를 저장하는 데이터 모델입니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TableAccessInfo:
    """
    데이터베이스 테이블 접근 정보를 저장하는 데이터 모델
    
    Attributes:
        table_name: 테이블명
        columns: 접근하는 칼럼 정보 목록 (칼럼명과 new_column 정보 포함)
        access_files: 접근하는 파일 경로 목록
        query_type: 쿼리 타입 (SELECT, INSERT, UPDATE, DELETE)
        sql_query: SQL 쿼리 (선택적)
        layer: 레이어 정보 (Mapper, DAO, Service, Controller)
        sql_queries: SQL 쿼리 목록 (상세 정보 포함)
        layer_files: 레이어별 파일 경로 목록
    """
    table_name: str
    columns: List[Dict[str, Any]]  # [{"name": "column_name", "new_column": bool}, ...]
    access_files: List[str]
    query_type: str
    sql_query: Optional[str] = None
    layer: str = ""
    sql_queries: List[Dict[str, Any]] = field(default_factory=list)
    layer_files: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            "table_name": self.table_name,
            "columns": self.columns,
            "access_files": self.access_files,
            "query_type": self.query_type,
            "sql_query": self.sql_query,
            "layer": self.layer,
            "sql_queries": self.sql_queries,
            "layer_files": self.layer_files
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TableAccessInfo":
        """딕셔너리로부터 TableAccessInfo 객체 생성"""
        # columns가 문자열 배열인 경우 객체 배열로 변환 (하위 호환성)
        columns = data.get("columns", [])
        if columns and isinstance(columns[0], str):
            # 기존 형식 (문자열 배열)을 객체 배열로 변환
            columns = [{"name": col, "new_column": False} for col in columns]
        
        return cls(
            table_name=data["table_name"],
            columns=columns,
            access_files=data.get("access_files", []),
            query_type=data["query_type"],
            sql_query=data.get("sql_query"),
            layer=data.get("layer", ""),
            sql_queries=data.get("sql_queries", []),
            layer_files=data.get("layer_files", {})
        )

