"""
SQL Parsing Strategy 모듈

Strategy Pattern을 사용하여 sql_wrapping_type별 SQL 쿼리 분석 전략을 제공합니다.
"""

import re
from abc import ABC, abstractmethod
from typing import Set


class SQLParsingStrategy(ABC):
    """
    SQL 파싱 전략 인터페이스

    각 SQL Wrapping 타입(MyBatis, JPA, JDBC)에 맞는 SQL 쿼리 분석 전략을 정의합니다.
    """

    @abstractmethod
    def extract_table_names(self, sql_query: str) -> Set[str]:
        """
        SQL 쿼리에서 테이블명 추출

        Args:
            sql_query: SQL 쿼리 문자열

        Returns:
            Set[str]: 추출된 테이블명 집합
        """
        pass

    @abstractmethod
    def extract_column_names(self, sql_query: str, table_name: str) -> Set[str]:
        """
        SQL 쿼리에서 특정 테이블의 칼럼명 추출

        Args:
            sql_query: SQL 쿼리 문자열
            table_name: 테이블명

        Returns:
            Set[str]: 추출된 칼럼명 집합
        """
        pass


class MyBatisStrategy(SQLParsingStrategy):
    """
    MyBatis SQL 파싱 전략

    MyBatis XML Mapper 파일의 SQL 쿼리를 분석합니다.
    """

    def extract_table_names(self, sql_query: str) -> Set[str]:
        """
        MyBatis SQL 쿼리에서 테이블명 추출

        Args:
            sql_query: SQL 쿼리 문자열

        Returns:
            Set[str]: 추출된 테이블명 집합
        """
        tables = set()

        # FROM 절에서 테이블명 추출
        from_pattern = (
            r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        )
        from_matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
        for match in from_matches:
            # 스키마명 제거 (예: schema.table -> table)
            table = match.split(".")[-1]
            tables.add(table.upper())

        # JOIN 절에서 테이블명 추출
        join_pattern = r"\b(?:INNER|LEFT|RIGHT|FULL|OUTER)?\s+JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        join_matches = re.findall(join_pattern, sql_query, re.IGNORECASE)
        for match in join_matches:
            table = match.split(".")[-1]
            tables.add(table.upper())

        # INSERT INTO 절에서 테이블명 추출
        insert_pattern = (
            r"\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        )
        insert_matches = re.findall(insert_pattern, sql_query, re.IGNORECASE)
        for match in insert_matches:
            table = match.split(".")[-1]
            tables.add(table.upper())

        # UPDATE 절에서 테이블명 추출
        update_pattern = (
            r"\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        )
        update_matches = re.findall(update_pattern, sql_query, re.IGNORECASE)
        for match in update_matches:
            table = match.split(".")[-1]
            tables.add(table.upper())

        # DELETE FROM 절에서 테이블명 추출
        delete_pattern = (
            r"\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b"
        )
        delete_matches = re.findall(delete_pattern, sql_query, re.IGNORECASE)
        for match in delete_matches:
            table = match.split(".")[-1]
            tables.add(table.upper())

        return tables

    def extract_column_names(self, sql_query: str, table_name: str) -> Set[str]:
        """
        MyBatis SQL 쿼리에서 특정 테이블의 칼럼명 추출

        Args:
            sql_query: SQL 쿼리 문자열
            table_name: 테이블명

        Returns:
            Set[str]: 추출된 칼럼명 집합
        """
        columns = set()
        table_alias = None

        # 테이블 별칭 찾기 (예: FROM users u -> u)
        alias_pattern = (
            rf"\bFROM\s+{re.escape(table_name)}\s+([a-zA-Z_][a-zA-Z0-9_]*)\b"
        )
        alias_match = re.search(alias_pattern, sql_query, re.IGNORECASE)
        if alias_match:
            table_alias = alias_match.group(1)

        # SELECT 절에서 칼럼명 추출
        select_pattern = r"\bSELECT\s+(.*?)\s+FROM\b"
        select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 각 칼럼 추출
            column_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)"
            column_matches = re.findall(column_pattern, select_clause)
            for match in column_matches:
                if "." in match:
                    # table.column 또는 alias.column 형식
                    parts = match.split(".")
                    if len(parts) == 2:
                        if (
                            parts[0].upper() == table_name.upper()
                            or parts[0] == table_alias
                        ):
                            columns.add(parts[1].upper())
                else:
                    # 단순 칼럼명
                    columns.add(match.upper())

        # INSERT INTO 절에서 칼럼명 추출
        insert_pattern = rf"\bINSERT\s+INTO\s+{re.escape(table_name)}\s*\(([^)]+)\)"
        insert_match = re.search(insert_pattern, sql_query, re.IGNORECASE)
        if insert_match:
            column_list = insert_match.group(1)
            for col in column_list.split(","):
                col = col.strip()
                if col:
                    columns.add(col.upper())

        # UPDATE SET 절에서 칼럼명 추출
        update_pattern = (
            rf"\bUPDATE\s+{re.escape(table_name)}\s+SET\s+(.*?)(?:\s+WHERE|\s*$)"
        )
        update_match = re.search(update_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if update_match:
            set_clause = update_match.group(1)
            # column = value 형식 추출
            set_column_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*="
            set_matches = re.findall(set_column_pattern, set_clause)
            for match in set_matches:
                columns.add(match.upper())

        return columns


class JPAStrategy(SQLParsingStrategy):
    """
    JPA SQL 파싱 전략

    JPA Entity와 JPQL 쿼리를 분석합니다.
    """

    def extract_table_names(self, sql_query: str) -> Set[str]:
        """
        JPA/JPQL 쿼리에서 테이블명 추출

        Args:
            sql_query: JPQL 쿼리 문자열

        Returns:
            Set[str]: 추출된 테이블명 집합 (Entity 이름)
        """
        tables = set()

        # JPQL FROM 절에서 Entity 이름 추출
        from_pattern = r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s+(?:[a-zA-Z_][a-zA-Z0-9_]*)?\b"
        from_matches = re.findall(from_pattern, sql_query, re.IGNORECASE)
        for match in from_matches:
            # 패키지명 제거 (예: com.example.User -> User)
            entity = match.split(".")[-1]
            tables.add(entity.upper())

        return tables

    def extract_column_names(self, sql_query: str, table_name: str) -> Set[str]:
        """
        JPA/JPQL 쿼리에서 특정 Entity의 속성명 추출

        Args:
            sql_query: JPQL 쿼리 문자열
            table_name: Entity 이름

        Returns:
            Set[str]: 추출된 속성명 집합
        """
        columns = set()

        # SELECT 절에서 속성명 추출
        select_pattern = r"\bSELECT\s+(.*?)\s+FROM\b"
        select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 각 속성 추출
            property_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)"
            property_matches = re.findall(property_pattern, select_clause)
            for match in property_matches:
                if "." in match:
                    # entity.property 형식
                    parts = match.split(".")
                    if len(parts) == 2 and parts[0].upper() == table_name.upper():
                        columns.add(parts[1].upper())
                else:
                    columns.add(match.upper())

        return columns


class JDBCStrategy(SQLParsingStrategy):
    """
    JDBC SQL 파싱 전략

    순수 JDBC SQL 쿼리를 분석합니다.
    """

    def extract_table_names(self, sql_query: str) -> Set[str]:
        """
        JDBC SQL 쿼리에서 테이블명 추출

        Args:
            sql_query: SQL 쿼리 문자열

        Returns:
            Set[str]: 추출된 테이블명 집합
        """
        # MyBatis 전략과 동일한 로직 사용
        mybatis_strategy = MyBatisStrategy()
        return mybatis_strategy.extract_table_names(sql_query)

    def extract_column_names(self, sql_query: str, table_name: str) -> Set[str]:
        """
        JDBC SQL 쿼리에서 특정 테이블의 칼럼명 추출

        Args:
            sql_query: SQL 쿼리 문자열
            table_name: 테이블명

        Returns:
            Set[str]: 추출된 칼럼명 집합
        """
        # MyBatis 전략과 동일한 로직 사용
        mybatis_strategy = MyBatisStrategy()
        return mybatis_strategy.extract_column_names(sql_query, table_name)


def create_strategy(sql_wrapping_type: str) -> SQLParsingStrategy:
    """
    SQL Wrapping 타입에 맞는 전략 인스턴스 생성

    Args:
        sql_wrapping_type: SQL Wrapping 타입 ("mybatis", "jpa", "jdbc")

    Returns:
        SQLParsingStrategy: 생성된 전략 인스턴스

    Raises:
        ValueError: 지원하지 않는 sql_wrapping_type인 경우
    """
    strategy_map = {
        "mybatis": MyBatisStrategy,
        "jpa": JPAStrategy,
        "jdbc": JDBCStrategy,
    }

    strategy_class = strategy_map.get(sql_wrapping_type.lower())
    if strategy_class is None:
        raise ValueError(f"지원하지 않는 sql_wrapping_type: {sql_wrapping_type}")

    return strategy_class()
