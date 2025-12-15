"""
XML Mapper Parser

lxml을 사용하여 MyBatis Mapper XML 파일을 파싱하고,
SQL 쿼리에서 테이블명과 칼럼명을 추출하는 모듈입니다.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from lxml import etree

from models.table_access_info import TableAccessInfo


@dataclass
class SQLQuery:
    """
    SQL 쿼리 정보를 저장하는 데이터 모델
    
    Attributes:
        id: SQL 태그의 id 속성 (메서드명)
        query_type: 쿼리 타입 (SELECT, INSERT, UPDATE, DELETE)
        sql: SQL 쿼리 문자열
        parameter_type: parameterType 속성
        result_type: resultType 속성 또는 resultMap의 type 속성
        result_map: resultMap 속성 (resultMap 태그의 id)
        namespace: Mapper 네임스페이스
    """
    id: str
    query_type: str
    sql: str
    parameter_type: Optional[str] = None
    result_type: Optional[str] = None
    result_map: Optional[str] = None
    namespace: str = ""


@dataclass
class MapperMethodMapping:
    """
    Mapper 메서드와 SQL 쿼리 매핑 정보
    
    Attributes:
        method_signature: 메서드 시그니처 (namespace.methodName)
        sql_query: SQL 쿼리 정보
        parameters: 파라미터 목록
    """
    method_signature: str
    sql_query: SQLQuery
    parameters: List[str] = field(default_factory=list)


class XMLMapperParser:
    """
    XML Mapper 파서 클래스
    
    lxml을 사용하여 MyBatis Mapper XML 파일을 파싱하고,
    SQL 쿼리에서 테이블명과 칼럼명을 추출합니다.
    """
    
    def __init__(self):
        """XMLMapperParser 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # SQL 키워드 패턴 (대소문자 무시)
        self.sql_keywords = {
            'select', 'insert', 'update', 'delete',
            'from', 'into', 'join', 'inner', 'left', 'right', 'full', 'outer',
            'where', 'group', 'order', 'having', 'union', 'union all'
        }
    
    def parse_file(self, file_path: Path) -> Tuple[Optional[etree.ElementTree], Optional[str]]:
        """
        XML 파일을 파싱
        
        Args:
            file_path: XML 파일 경로
            
        Returns:
            Tuple[Optional[etree.ElementTree], Optional[str]]: (파싱된 XML 트리, 에러 메시지)
        """
        try:
            # XML 파일 파싱
            tree = etree.parse(str(file_path))
            return tree, None
            
        except etree.XMLSyntaxError as e:
            error_msg = f"XML 구문 오류: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
        except FileNotFoundError as e:
            error_msg = f"파일을 찾을 수 없습니다: {file_path}"
            self.logger.error(error_msg)
            return None, error_msg
        except OSError as e:
            if "No such file" in str(e) or "파일" in str(e):
                error_msg = f"파일을 찾을 수 없습니다: {file_path}"
            else:
                error_msg = f"파일 읽기 오류: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"XML 파싱 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
    
    def extract_sql_tags(self, tree: etree.ElementTree) -> List[SQLQuery]:
        """
        SQL 태그에서 쿼리 추출
        
        Args:
            tree: 파싱된 XML 트리
            
        Returns:
            List[SQLQuery]: SQL 쿼리 목록
        """
        sql_queries = []
        root = tree.getroot()
        
        # 네임스페이스 추출
        namespace = root.get('namespace', '')
        
        # resultMap 태그에서 type 정보 추출 (id -> type 매핑)
        result_map_types = self._extract_result_map_types(root)
        
        # SQL 태그 찾기
        sql_tags = ['select', 'insert', 'update', 'delete']
        
        for tag_name in sql_tags:
            # XPath를 사용하여 태그 찾기 (네임스페이스 고려)
            xpath = f".//{tag_name}"
            elements = root.xpath(xpath)
            
            for element in elements:
                sql_query = self._extract_sql_from_element(
                    element, 
                    tag_name.upper(), 
                    namespace,
                    result_map_types
                )
                if sql_query:
                    sql_queries.append(sql_query)
        
        return sql_queries
    
    def _extract_result_map_types(self, root: etree.Element) -> Dict[str, str]:
        """
        resultMap 태그에서 id와 type 매핑 추출
        
        Args:
            root: XML 루트 요소
            
        Returns:
            Dict[str, str]: resultMap id -> type 매핑
        """
        result_map_types = {}
        
        # resultMap 태그 찾기
        result_maps = root.xpath(".//resultMap")
        
        for result_map in result_maps:
            result_map_id = result_map.get('id')
            result_map_type = result_map.get('type')
            
            if result_map_id and result_map_type:
                result_map_types[result_map_id] = result_map_type
                self.logger.debug(f"resultMap 발견: id={result_map_id}, type={result_map_type}")
        
        return result_map_types
    
    def _extract_sql_from_element(
        self, 
        element: etree.Element, 
        query_type: str,
        namespace: str,
        result_map_types: Dict[str, str] = None
    ) -> Optional[SQLQuery]:
        """
        XML 요소에서 SQL 쿼리 추출
        
        Args:
            element: XML 요소
            query_type: 쿼리 타입
            namespace: 네임스페이스
            result_map_types: resultMap id -> type 매핑
            
        Returns:
            Optional[SQLQuery]: SQL 쿼리 정보
        """
        if result_map_types is None:
            result_map_types = {}
        
        # id 속성 추출
        query_id = element.get('id', '')
        if not query_id:
            self.logger.warning("SQL 태그에 id 속성이 없습니다.")
            return None
        
        # parameterType, resultType 추출
        parameter_type = element.get('parameterType')
        result_type = element.get('resultType')
        result_map = element.get('resultMap')
        
        # resultMap이 지정된 경우, resultMap 태그에서 type 추출
        if result_map and not result_type:
            # resultMap id로 type 찾기
            if result_map in result_map_types:
                result_type = result_map_types[result_map]
                self.logger.debug(f"resultMap '{result_map}'에서 type 추출: {result_type}")
            else:
                self.logger.warning(f"resultMap '{result_map}'를 찾을 수 없습니다.")
        
        # SQL 쿼리 텍스트 추출
        sql_text = self._extract_text_content(element)
        
        if not sql_text:
            self.logger.warning(f"SQL 태그 '{query_id}'에 쿼리가 없습니다.")
            return None
        
        return SQLQuery(
            id=query_id,
            query_type=query_type,
            sql=sql_text,
            parameter_type=parameter_type,
            result_type=result_type,
            result_map=result_map,
            namespace=namespace
        )
    
    def _extract_text_content(self, element: etree.Element) -> str:
        """
        XML 요소의 텍스트 콘텐츠 추출 (CDATA 포함)
        
        Args:
            element: XML 요소
            
        Returns:
            str: 추출된 텍스트
        """
        # lxml은 CDATA를 자동으로 처리하므로 직접 텍스트 추출
        text_parts = []
        
        # 직접 텍스트 노드
        if element.text:
            text_parts.append(element.text.strip())
        
        # 자식 요소의 텍스트 (동적 SQL 요소는 일단 무시)
        for child in element:
            if child.tag in ['if', 'choose', 'when', 'otherwise', 'foreach', 'where', 'set', 'trim']:
                # 동적 SQL 요소는 일단 텍스트만 추출
                if child.text:
                    text_parts.append(child.text.strip())
                if child.tail:
                    text_parts.append(child.tail.strip())
            else:
                # 일반 요소는 재귀적으로 처리
                child_text = self._extract_text_content(child)
                if child_text:
                    text_parts.append(child_text)
                if child.tail:
                    text_parts.append(child.tail.strip())
        
        return ' '.join(text_parts)
    
    def remove_sql_comments(self, sql: str) -> str:
        """
        SQL 주석 제거
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            str: 주석이 제거된 SQL
        """
        # 여러 줄 주석 제거 (/* ... */)
        # 문자열 리터럴 내 주석은 보존
        sql_lines = []
        in_string = False
        string_char = None
        
        i = 0
        while i < len(sql):
            char = sql[i]
            
            # 문자열 리터럴 감지
            if char in ("'", '"') and (i == 0 or sql[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            # 여러 줄 주석 처리
            if not in_string and i < len(sql) - 1 and sql[i:i+2] == '/*':
                # 주석 끝 찾기
                end = sql.find('*/', i + 2)
                if end != -1:
                    i = end + 2
                    continue
            
            # 한 줄 주석 처리
            if not in_string and i < len(sql) - 1 and sql[i:i+2] == '--':
                # 줄 끝까지 건너뛰기
                end = sql.find('\n', i + 2)
                if end != -1:
                    i = end + 1
                    continue
                else:
                    break
            
            sql_lines.append(char)
            i += 1
        
        result = ''.join(sql_lines)
        
        # 공백 정규화
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        return result
    
    def extract_table_names(self, sql: str) -> List[str]:
        """
        SQL 쿼리에서 테이블명 추출
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            List[str]: 테이블명 목록
        """
        tables = []
        sql_upper = sql.upper()
        
        # FROM 절에서 테이블명 추출
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        from_matches = re.findall(from_pattern, sql_upper, re.IGNORECASE)
        tables.extend([t.lower() for t in from_matches])
        
        # JOIN 절에서 테이블명 추출
        join_pattern = r'\b(?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        join_matches = re.findall(join_pattern, sql_upper, re.IGNORECASE)
        tables.extend([t.lower() for t in join_matches])
        
        # INSERT INTO 절에서 테이블명 추출
        insert_pattern = r'\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        insert_matches = re.findall(insert_pattern, sql_upper, re.IGNORECASE)
        tables.extend([t.lower() for t in insert_matches])
        
        # UPDATE 절에서 테이블명 추출
        update_pattern = r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        update_matches = re.findall(update_pattern, sql_upper, re.IGNORECASE)
        tables.extend([t.lower() for t in update_matches])
        
        # DELETE FROM 절에서 테이블명 추출
        delete_pattern = r'\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
        delete_matches = re.findall(delete_pattern, sql_upper, re.IGNORECASE)
        tables.extend([t.lower() for t in delete_matches])
        
        # 중복 제거 및 정렬
        tables = list(set(tables))
        tables.sort()
        
        return tables
    
    def extract_column_names(self, sql: str) -> List[str]:
        """
        SQL 쿼리에서 칼럼명 추출
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            List[str]: 칼럼명 목록
        """
        columns = []
        
        # SELECT 절에서 칼럼명 추출
        select_pattern = r'\bSELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 쉼표로 분리
            column_parts = re.split(r',', select_clause)
            for part in column_parts:
                part = part.strip()
                # AS 별칭 제거
                if ' AS ' in part.upper():
                    part = part.split(' AS ', 1)[0].strip()
                elif ' ' in part and not part.startswith('('):
                    # 함수 호출이 아닌 경우만
                    if not re.match(r'^\w+\s*\(', part):
                        part = part.split()[0]
                
                # 테이블명.칼럼명 형식 처리
                if '.' in part:
                    part = part.split('.')[-1]
                
                # 기본 칼럼명만 추출 (함수, 산술 연산 제외)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                    columns.append(part.lower())
        
        # INSERT INTO 절에서 칼럼명 추출
        insert_pattern = r'\bINSERT\s+INTO\s+\w+\s*\((.*?)\)'
        insert_match = re.search(insert_pattern, sql, re.IGNORECASE | re.DOTALL)
        if insert_match:
            column_list = insert_match.group(1)
            column_parts = re.split(r',', column_list)
            for part in column_parts:
                part = part.strip()
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                    columns.append(part.lower())
        
        # UPDATE 절에서 칼럼명 추출
        update_pattern = r'\bSET\s+(.*?)(?:\s+WHERE|\s*$)'
        update_match = re.search(update_pattern, sql, re.IGNORECASE | re.DOTALL)
        if update_match:
            set_clause = update_match.group(1)
            # SET column = value 형식
            assignments = re.split(r',', set_clause)
            for assignment in assignments:
                assignment = assignment.strip()
                if '=' in assignment:
                    column = assignment.split('=')[0].strip()
                    # 테이블명.칼럼명 형식 처리
                    if '.' in column:
                        column = column.split('.')[-1]
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                        columns.append(column.lower())
        
        # 중복 제거 및 정렬
        columns = list(set(columns))
        columns.sort()
        
        return columns
    
    def extract_mybatis_parameters(self, sql: str) -> List[str]:
        """
        MyBatis 파라미터 표기법 추출
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            List[str]: 파라미터 이름 목록
        """
        parameters = []
        
        # #{paramName} 패턴 추출
        hash_pattern = r'#\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        hash_matches = re.findall(hash_pattern, sql)
        parameters.extend(hash_matches)
        
        # ${paramName} 패턴 추출
        dollar_pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        dollar_matches = re.findall(dollar_pattern, sql)
        parameters.extend(dollar_matches)
        
        # 중복 제거
        parameters = list(set(parameters))
        
        return parameters
    
    def create_method_mapping(
        self, 
        sql_query: SQLQuery
    ) -> MapperMethodMapping:
        """
        Mapper 메서드와 SQL 쿼리 매핑 생성
        
        Args:
            sql_query: SQL 쿼리 정보
            
        Returns:
            MapperMethodMapping: 매핑 정보
        """
        # 메서드 시그니처 생성
        if sql_query.namespace:
            method_signature = f"{sql_query.namespace}.{sql_query.id}"
        else:
            method_signature = sql_query.id
        
        # 파라미터 추출
        parameters = self.extract_mybatis_parameters(sql_query.sql)
        
        return MapperMethodMapping(
            method_signature=method_signature,
            sql_query=sql_query,
            parameters=parameters
        )
    
    def extract_table_access_info(
        self, 
        file_path: Path
    ) -> List[TableAccessInfo]:
        """
        XML 파일에서 테이블 접근 정보 추출
        
        Args:
            file_path: XML 파일 경로
            
        Returns:
            List[TableAccessInfo]: 테이블 접근 정보 목록
        """
        table_access_list = []
        
        # XML 파싱
        tree, error = self.parse_file(file_path)
        if error:
            self.logger.error(f"XML 파싱 실패: {error}")
            return table_access_list
        
        # SQL 태그 추출
        sql_queries = self.extract_sql_tags(tree)
        
        # 각 SQL 쿼리에서 테이블 접근 정보 추출
        for sql_query in sql_queries:
            # 주석 제거
            clean_sql = self.remove_sql_comments(sql_query.sql)
            
            # 테이블명 추출
            tables = self.extract_table_names(clean_sql)
            
            # 칼럼명 추출
            columns = self.extract_column_names(clean_sql)
            
            # 각 테이블에 대해 TableAccessInfo 생성
            for table in tables:
                table_access = TableAccessInfo(
                    table_name=table,
                    columns=columns,
                    access_files=[str(file_path)],
                    query_type=sql_query.query_type,
                    sql_query=clean_sql,
                    layer="Mapper"
                )
                table_access_list.append(table_access)
        
        return table_access_list
    
    def parse_mapper_file(
        self, 
        file_path: Path
    ) -> Dict[str, Any]:
        """
        Mapper XML 파일을 완전히 파싱하여 모든 정보 반환
        
        Args:
            file_path: XML 파일 경로
            
        Returns:
            Dict[str, Any]: 파싱 결과
        """
        result = {
            "file_path": str(file_path),
            "sql_queries": [],
            "method_mappings": [],
            "table_access_info": [],
            "error": None
        }
        
        # XML 파싱
        tree, error = self.parse_file(file_path)
        if error:
            result["error"] = error
            return result
        
        # SQL 태그 추출
        sql_queries = self.extract_sql_tags(tree)
        result["sql_queries"] = [
            {
                "id": q.id,
                "query_type": q.query_type,
                "sql": q.sql,
                "parameter_type": q.parameter_type,
                "result_type": q.result_type,
                "result_map": q.result_map,
                "namespace": q.namespace
            }
            for q in sql_queries
        ]
        
        # 메서드 매핑 생성
        method_mappings = []
        for sql_query in sql_queries:
            mapping = self.create_method_mapping(sql_query)
            method_mappings.append({
                "method_signature": mapping.method_signature,
                "parameters": mapping.parameters,
                "query_type": sql_query.query_type
            })
        result["method_mappings"] = method_mappings
        
        # 테이블 접근 정보 추출
        table_access_info = self.extract_table_access_info(file_path)
        result["table_access_info"] = [
            {
                "table_name": t.table_name,
                "columns": t.columns,
                "query_type": t.query_type,
                "layer": t.layer
            }
            for t in table_access_info
        ]
        
        return result

