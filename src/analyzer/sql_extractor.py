"""
SQL Extractor 모듈

SQLParsingStrategy에 따라 서로 다른 방식으로 SQL을 추출하고
sql_extraction_results.json에 저장하는 모듈입니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from models.source_file import SourceFile
from parser.xml_mapper_parser import XMLMapperParser
from parser.java_ast_parser import JavaASTParser
from .sql_parsing_strategy import SQLParsingStrategy


@dataclass
class ExtractedSQLQuery:
    """
    추출된 SQL 쿼리 정보를 저장하는 데이터 모델
    
    Attributes:
        id: 쿼리 식별자 (MyBatis: query id, JDBC/JPA: method name 등)
        query_type: 쿼리 타입 (SELECT, INSERT, UPDATE, DELETE)
        sql: SQL 쿼리 문자열
        strategy_specific: 전략별 특정 정보 (dict)
    """
    id: str
    query_type: str
    sql: str
    strategy_specific: Dict[str, Any] = field(default_factory=dict)


class SQLExtractor:
    """
    SQL Extractor 클래스
    
    SQLParsingStrategy에 따라 서로 다른 방식으로 SQL을 추출합니다.
    """
    
    def __init__(
        self,
        strategy: SQLParsingStrategy,
        xml_parser: Optional[XMLMapperParser] = None,
        java_parser: Optional[JavaASTParser] = None
    ):
        """
        SQLExtractor 초기화
        
        Args:
            strategy: SQL 파싱 전략 (MyBatisStrategy, JPAStrategy, JDBCStrategy)
            xml_parser: XML Mapper 파서 (MyBatis 전략일 때 사용)
            java_parser: Java AST 파서 (JDBC/JPA 전략일 때 사용)
        """
        self.strategy = strategy
        self.xml_parser = xml_parser or XMLMapperParser()
        self.java_parser = java_parser or JavaASTParser()
        self.logger = logging.getLogger(__name__)
    
    def extract_from_files(
        self,
        source_files: List[SourceFile]
    ) -> List[Dict[str, Any]]:
        """
        소스 파일들에서 SQL 쿼리 추출
        
        Args:
            source_files: 분석할 소스 파일 목록
            
        Returns:
            List[Dict[str, Any]]: 추출된 SQL 쿼리 정보 목록
                각 항목은 {"file": {...}, "sql_queries": [...]} 형태
        """
        results = []
        
        # 전략 타입에 따라 다른 방식으로 추출
        strategy_type = type(self.strategy).__name__
        
        if strategy_type == "MyBatisStrategy":
            results = self._extract_mybatis(source_files)
        elif strategy_type == "JDBCStrategy":
            results = self._extract_jdbc(source_files)
        elif strategy_type == "JPAStrategy":
            results = self._extract_jpa(source_files)
        else:
            self.logger.warning(f"알 수 없는 전략 타입: {strategy_type}")
        
        return results
    
    def _extract_mybatis(
        self,
        source_files: List[SourceFile]
    ) -> List[Dict[str, Any]]:
        """
        MyBatis 전략: XML Mapper 파일에서 SQL 추출
        
        Args:
            source_files: 소스 파일 목록
            
        Returns:
            List[Dict[str, Any]]: 추출 결과
        """
        results = []
        xml_files = [f for f in source_files if f.extension == ".xml"]
        
        for xml_file in xml_files:
            try:
                parse_result = self.xml_parser.parse_mapper_file(xml_file.path)
                if parse_result.get("error"):
                    continue
                
                # SQL 쿼리들을 strategy 중립적인 형식으로 변환
                sql_queries = []
                for query in parse_result.get("sql_queries", []):
                    # strategy_specific에 MyBatis 특정 정보 저장
                    strategy_specific = {
                        "namespace": query.get("namespace", ""),
                        "parameter_type": query.get("parameter_type"),
                        "result_type": query.get("result_type"),
                        "result_map": query.get("result_map")
                    }
                    
                    sql_queries.append({
                        "id": query.get("id", ""),
                        "query_type": query.get("query_type", "SELECT"),
                        "sql": query.get("sql", ""),
                        "strategy_specific": strategy_specific
                    })
                
                if sql_queries:
                    results.append({
                        "file": xml_file.to_dict(),
                        "sql_queries": sql_queries
                    })
                    
            except Exception as e:
                self.logger.warning(f"XML 파일 추출 실패: {xml_file.path} - {e}")
        
        return results
    
    def _extract_jdbc(
        self,
        source_files: List[SourceFile]
    ) -> List[Dict[str, Any]]:
        """
        JDBC 전략: Java 파일에서 JDBC SQL 추출
        
        Args:
            source_files: 소스 파일 목록
            
        Returns:
            List[Dict[str, Any]]: 추출 결과
        """
        results = []
        java_files = [f for f in source_files if f.extension == ".java"]
        
        for java_file in java_files:
            try:
                # JavaASTParser의 JDBC SQL 추출 기능 사용
                sql_queries = self.java_parser.extract_jdbc_sql(java_file.path)
                
                if sql_queries:
                    # strategy_specific에 JDBC 특정 정보 저장
                    for query in sql_queries:
                        query["strategy_specific"] = query.get("strategy_specific", {})
                    
                    results.append({
                        "file": java_file.to_dict(),
                        "sql_queries": sql_queries
                    })
                    
            except Exception as e:
                self.logger.warning(f"JDBC SQL 추출 실패: {java_file.path} - {e}")
        
        return results
    
    def _extract_jpa(
        self,
        source_files: List[SourceFile]
    ) -> List[Dict[str, Any]]:
        """
        JPA 전략: Java 파일에서 JPA/JPQL 쿼리 추출
        
        Args:
            source_files: 소스 파일 목록
            
        Returns:
            List[Dict[str, Any]]: 추출 결과
        """
        results = []
        java_files = [f for f in source_files if f.extension == ".java"]
        
        for java_file in java_files:
            try:
                # JavaASTParser의 JPA SQL 추출 기능 사용
                sql_queries = self.java_parser.extract_jpa_sql(java_file.path)
                
                if sql_queries:
                    # strategy_specific에 JPA 특정 정보 저장
                    for query in sql_queries:
                        query["strategy_specific"] = query.get("strategy_specific", {})
                    
                    results.append({
                        "file": java_file.to_dict(),
                        "sql_queries": sql_queries
                    })
                    
            except Exception as e:
                self.logger.warning(f"JPA SQL 추출 실패: {java_file.path} - {e}")
        
        return results

