"""
DB Access Analyzer 모듈

config.json에 설정된 DB 테이블과 칼럼에 접근하는 소스 파일 목록을 작성합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict

from models.table_access_info import TableAccessInfo
from models.source_file import SourceFile
from parser.xml_mapper_parser import XMLMapperParser
from parser.java_ast_parser import JavaASTParser
from parser.call_graph_builder import CallGraphBuilder
from config.config_manager import ConfigurationManager
from .sql_parsing_strategy import SQLParsingStrategy


class DBAccessAnalyzer:
    """
    DB Access Analyzer 클래스
    
    config.json에 설정된 DB 테이블과 칼럼에 접근하는 소스 파일을 식별합니다.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        sql_strategy: SQLParsingStrategy,
        xml_parser: Optional[XMLMapperParser] = None,
        java_parser: Optional[JavaASTParser] = None,
        call_graph_builder: Optional[CallGraphBuilder] = None
    ):
        """
        DBAccessAnalyzer 초기화
        
        Args:
            config_manager: 설정 매니저
            sql_strategy: SQL 파싱 전략 (필수)
            xml_parser: XML Mapper 파서 (선택적, 하위 호환성을 위해 유지)
            java_parser: Java AST 파서 (선택적)
            call_graph_builder: Call Graph Builder (선택적)
        """
        self.config_manager = config_manager
        self.sql_strategy = sql_strategy
        self.xml_parser = xml_parser  # 하위 호환성을 위해 유지하지만 사용하지 않음
        self.java_parser = java_parser or JavaASTParser()
        self.call_graph_builder = call_graph_builder
        self.logger = logging.getLogger(__name__)
        
        # 설정에서 테이블 정보 가져오기
        self.access_tables = config_manager.get("access_tables", [])
        if not self.access_tables:
            self.access_tables = config_manager.access_tables
        
        # 테이블별 칼럼 매핑 생성 (대소문자 무시)
        # 칼럼 정보: {column_name: {"new_column": bool}}
        self.table_column_map: Dict[str, Set[str]] = {}
        self.table_column_info: Dict[str, Dict[str, Dict[str, Any]]] = {}  # table_name -> {column_name: {"new_column": bool}}
        
        for table_info in self.access_tables:
            table_name = table_info.get("table_name", "").lower()
            columns_set = set()
            column_info_dict = {}
            
            for col in table_info.get("columns", []):
                if isinstance(col, str):
                    # 문자열 형식: "column_name"
                    col_name = col.lower()
                    columns_set.add(col_name)
                    column_info_dict[col_name] = {"new_column": False}
                elif isinstance(col, dict):
                    # 객체 형식: {"name": "column_name", "new_column": true}
                    col_name = col.get("name", "").lower()
                    if col_name:
                        columns_set.add(col_name)
                        column_info_dict[col_name] = {
                            "new_column": col.get("new_column", False)
                        }
            
            self.table_column_map[table_name] = columns_set
            self.table_column_info[table_name] = column_info_dict
    
    def analyze(
        self,
        source_files: List[SourceFile]
    ) -> List[TableAccessInfo]:
        """
        소스 파일들을 분석하여 테이블 접근 정보 추출
        
        Args:
            source_files: 분석할 소스 파일 목록
            
        Returns:
            List[TableAccessInfo]: 테이블 접근 정보 목록
        """
        # sql_extraction_results.json에서 SQL 쿼리 정보 로드 (한 번만)
        from ..persistence.data_persistence_manager import DataPersistenceManager
        
        persistence_manager = DataPersistenceManager(self.config_manager.target_project)
        sql_extraction_results = persistence_manager.load_from_file("sql_extraction_results.json")
        
        if not sql_extraction_results:
            self.logger.warning("sql_extraction_results.json을 찾을 수 없습니다. 먼저 analyze 명령어를 실행하세요.")
            return []
        
        # ClassInfo 목록 수집 (CallGraphBuilder에서 재사용)
        class_info_map = self._collect_class_info_map(source_files)
        
        # config.json에 설정된 각 DB 테이블에 대해 분석
        table_access_info_list = []
        
        for table_config in self.access_tables:
            table_name = table_config.get("table_name", "").lower()
            
            if not table_name:
                continue
            
            # 칼럼 목록 추출 (문자열 또는 객체 형식 모두 지원)
            columns = set()
            for col in table_config.get("columns", []):
                if isinstance(col, str):
                    columns.add(col.lower())
                elif isinstance(col, dict):
                    col_name = col.get("name", "")
                    if col_name:
                        columns.add(col_name.lower())
            
            # 규칙 1: 테이블만 있고 칼럼 정보가 없는 테이블은 무시
            if not columns:
                self.logger.debug(f"테이블 '{table_name}'은 칼럼 정보가 없어 무시됩니다.")
                continue
            
            # 테이블별 접근 정보 수집
            table_info = self._analyze_table_access(
                table_name,
                columns,
                source_files,
                class_info_map,
                sql_extraction_results
            )
            
            if table_info:
                table_access_info_list.append(table_info)
        
        return table_access_info_list
    
    def _collect_class_info_map(
        self,
        source_files: List[SourceFile]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        ClassInfo 목록을 수집하여 클래스명 -> 파일 경로 매핑 생성
        
        Args:
            source_files: 소스 파일 목록
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 클래스명 -> ClassInfo 리스트 매핑
        """
        class_info_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # CallGraphBuilder에서 이미 파싱된 클래스 정보 재사용
        if self.call_graph_builder and self.call_graph_builder.file_to_classes_map:
            for file_path_str, classes in self.call_graph_builder.file_to_classes_map.items():
                for cls in classes:
                    # 클래스명 (단순 이름)
                    class_info_map[cls.name].append({
                        "class_name": cls.name,
                        "package": cls.package,
                        "full_class_name": f"{cls.package}.{cls.name}" if cls.package else cls.name,
                        "file_path": file_path_str
                    })
                    
                    # 패키지 포함 전체 클래스명
                    if cls.package:
                        full_class_name = f"{cls.package}.{cls.name}"
                        class_info_map[full_class_name].append({
                            "class_name": cls.name,
                            "package": cls.package,
                            "full_class_name": full_class_name,
                            "file_path": file_path_str
                        })
        else:
            # CallGraphBuilder가 없으면 직접 파싱
            java_files = [f for f in source_files if f.extension == ".java"]
            for java_file in java_files:
                try:
                    tree, error = self.java_parser.parse_file(java_file.path)
                    if error:
                        continue
                    
                    classes = self.java_parser.extract_class_info(tree, java_file.path)
                    for cls in classes:
                        class_info_map[cls.name].append({
                            "class_name": cls.name,
                            "package": cls.package,
                            "full_class_name": f"{cls.package}.{cls.name}" if cls.package else cls.name,
                            "file_path": str(java_file.path)
                        })
                        
                        if cls.package:
                            full_class_name = f"{cls.package}.{cls.name}"
                            class_info_map[full_class_name].append({
                                "class_name": cls.name,
                                "package": cls.package,
                                "full_class_name": full_class_name,
                                "file_path": str(java_file.path)
                            })
                except Exception as e:
                    self.logger.debug(f"Java 파일 파싱 중 오류 (무시): {java_file.path} - {e}")
        
        return dict(class_info_map)
    
    def _analyze_table_access(
        self,
        table_name: str,
        columns: Set[str],
        source_files: List[SourceFile],
        class_info_map: Dict[str, List[Dict[str, Any]]],
        sql_extraction_results: List[Dict[str, Any]]
    ) -> Optional[TableAccessInfo]:
        """
        특정 테이블에 대한 접근 정보 분석
        
        Args:
            table_name: 테이블명
            columns: 칼럼 목록
            source_files: 소스 파일 목록
            class_info_map: 클래스 정보 매핑
            sql_extraction_results: SQL 추출 결과 목록
            
        Returns:
            Optional[TableAccessInfo]: 테이블 접근 정보
        """
        
        # 수집된 정보
        sql_queries: List[Dict[str, Any]] = []
        interface_files: Set[str] = set()
        dao_files: Set[str] = set()
        layer_files: Dict[str, Set[str]] = defaultdict(set)  # layer -> file_paths
        all_added_files: Set[str] = set()  # 모든 레이어에 추가된 파일 추적 (중복 방지용)
        
        # 각 파일의 SQL 쿼리에 대해 분석
        for file_result in sql_extraction_results:
            file_info = file_result.get("file", {})
            file_path = file_info.get("path", "")
            
            # SQL 쿼리 중 설정된 테이블을 사용하는 쿼리 수집
            # 규칙에 따라 쿼리 필터링:
            # - new_column=false인 칼럼 중 하나 이상 사용되는 쿼리 포함
            # - new_column=true인 칼럼이 있으면 테이블만 사용되면 포함 (칼럼 사용 여부 무관)
            false_columns, true_columns = self._get_column_groups(table_name, columns)
            matching_queries = self._find_matching_sql_queries(
                file_result.get("sql_queries", []),
                table_name,
                false_columns,
                true_columns
            )
            
            if not matching_queries:
                continue
            
            # 각 SQL 쿼리에 대해 처리
            for sql_query_info in matching_queries:
                sql_queries.append(sql_query_info)
                
                # strategy_specific에서 정보 추출 (전략별로 다름)
                strategy_specific = sql_query_info.get("strategy_specific", {})
                strategy_type = type(self.sql_strategy).__name__
                
                if strategy_type == "MyBatisStrategy":
                    # MyBatis: namespace와 result_type 사용
                    namespace = strategy_specific.get("namespace", "")
                    if namespace:
                        interface_file = self._find_class_file(namespace, class_info_map)
                        if interface_file and interface_file not in all_added_files:
                            interface_files.add(interface_file)
                            layer_files["Repository"].add(interface_file)
                            all_added_files.add(interface_file)
                    
                    result_type = strategy_specific.get("result_type")
                    if result_type:
                        dao_file = self._find_class_file(result_type, class_info_map)
                        if dao_file and dao_file not in all_added_files:
                            dao_files.add(dao_file)
                            layer_files["Repository"].add(dao_file)
                            all_added_files.add(dao_file)
                    
                    # namespace의 class_name + sql query의 id로 method string 조합
                    method_string = self._build_method_string(namespace, sql_query_info.get("id", ""))
                elif strategy_type in ["JDBCStrategy", "JPAStrategy"]:
                    # JDBC/JPA: method_name 사용
                    method_name = strategy_specific.get("method_name", "")
                    file_path_str = strategy_specific.get("file_path", file_path)
                    
                    # 파일 경로에서 클래스 찾기
                    method_string = None
                    if file_path_str:
                        # 파일 경로로 클래스 찾기
                        for class_name, class_infos in class_info_map.items():
                            for class_info in class_infos:
                                if class_info["file_path"] == file_path_str:
                                    method_string = f"{class_info['class_name']}.{method_name}"
                                    break
                            if method_string:
                                break
                    
                    if not method_string:
                        method_string = method_name
                else:
                    method_string = None
                
                if method_string and self.call_graph_builder and self.call_graph_builder.call_graph:
                    # Call Graph에서 역방향으로 탐색하여 상위 layer 파일 찾기
                    upper_layer_files = self._find_upper_layer_files(method_string)
                    for layer, file_path in upper_layer_files:
                        if layer and file_path and file_path not in all_added_files:
                            layer_files[layer].add(file_path)
                            all_added_files.add(file_path)
        
        # TableAccessInfo 생성
        if not sql_queries:
            return None
        
        # 모든 레이어 파일 경로 수집
        all_access_files = set(interface_files)
        all_access_files.update(dao_files)
        for files in layer_files.values():
            all_access_files.update(files)
        
        # 레이어 결정 (가장 많은 파일이 있는 레이어)
        main_layer = self._determine_main_layer(layer_files)
        
        # 칼럼 목록 추출 (SQL 쿼리에서 사용된 칼럼)
        used_columns = self._extract_used_columns(sql_queries, table_name, columns)
        
        # 칼럼 정보 생성 (config.json에 설정된 모든 칼럼 포함)
        # SQL 쿼리에서 사용된 칼럼은 실제로 사용된 것으로 표시
        columns_list = []
        table_lower = table_name.lower()
        column_info_dict = self.table_column_info.get(table_lower, {})
        
        # config.json에 설정된 모든 칼럼을 포함
        for col_name, col_info in sorted(column_info_dict.items()):
            columns_list.append({
                "name": col_name,
                "new_column": col_info.get("new_column", False)
            })
        
        # layer_files를 딕셔너리로 변환 (Set -> List)
        layer_files_dict = {
            layer: sorted(list(files))
            for layer, files in layer_files.items()
        }
        
        table_access_info = TableAccessInfo(
            table_name=table_name,
            columns=columns_list,  # 칼럼 정보 목록 (칼럼명과 new_column 정보 포함)
            access_files=sorted(list(all_access_files)),
            query_type=sql_queries[0].get("query_type", "SELECT") if sql_queries else "SELECT",
            sql_query=sql_queries[0].get("sql", "") if sql_queries else None,
            layer=main_layer,
            sql_queries=sql_queries,  # SQL 쿼리 목록 저장
            layer_files=layer_files_dict  # 레이어별 파일 경로 목록 저장
        )
        
        return table_access_info
    
    def _get_column_groups(
        self,
        table_name: str,
        columns: Set[str]
    ) -> tuple[Set[str], Set[str]]:
        """
        칼럼을 new_column 값에 따라 그룹화
        
        Args:
            table_name: 테이블명
            columns: 전체 칼럼 목록
            
        Returns:
            tuple[Set[str], Set[str]]: (new_column=false인 칼럼 목록, new_column=true인 칼럼 목록)
        """
        table_lower = table_name.lower()
        column_info_dict = self.table_column_info.get(table_lower, {})
        
        false_columns = set()  # new_column=false인 칼럼
        true_columns = set()   # new_column=true인 칼럼
        
        for col in columns:
            col_lower = col.lower()
            col_info = column_info_dict.get(col_lower, {"new_column": False})
            if col_info.get("new_column", False):
                true_columns.add(col_lower)
            else:
                false_columns.add(col_lower)
        
        return false_columns, true_columns
    
    def _find_matching_sql_queries(
        self,
        sql_queries: List[Dict[str, Any]],
        table_name: str,
        false_columns: Set[str],
        true_columns: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        SQL 쿼리 중 설정된 테이블을 사용하는 쿼리 찾기
        
        Args:
            sql_queries: SQL 쿼리 목록
            table_name: 테이블명
            false_columns: new_column=false인 칼럼 목록
            true_columns: new_column=true인 칼럼 목록
            
        Returns:
            List[Dict[str, Any]]: 매칭되는 SQL 쿼리 목록
            
        규칙:
            - 테이블명이 일치해야 함
            - new_column=false인 칼럼이 있으면: 그 중 하나 이상 사용되는 쿼리만 포함
            - new_column=true인 칼럼이 있으면: 테이블만 사용되면 포함 (칼럼 사용 여부 무관)
        """
        matching_queries = []
        
        for sql_query_info in sql_queries:
            sql = sql_query_info.get("sql", "")
            if not sql:
                continue
            
            # 테이블명 확인
            tables = self.sql_strategy.extract_table_names(sql)
            if table_name.lower() not in {t.lower() for t in tables}:
                continue
            
            # SQL에서 칼럼 추출
            sql_columns = self.sql_strategy.extract_column_names(sql, table_name)
            sql_columns_lower = {c.lower() for c in sql_columns}
            
            # 규칙 2: new_column=false인 칼럼이 있으면 그 중 하나 이상 사용되는 쿼리만 포함
            if false_columns:
                if not false_columns.intersection(sql_columns_lower):
                    # new_column=false인 칼럼이 하나도 사용되지 않으면 제외
                    continue
            
            # 규칙 3: new_column=true인 칼럼이 있으면 테이블만 사용되면 포함 (칼럼 사용 여부 무관)
            # 테이블명이 일치하면 이미 포함됨 (위에서 확인 완료)
            
            matching_queries.append(sql_query_info)
        
        return matching_queries
    
    def _find_class_file(
        self,
        full_class_name: str,
        class_info_map: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[str]:
        """
        클래스명으로 파일 경로 찾기
        
        Args:
            full_class_name: 전체 클래스명 (패키지 포함)
            class_info_map: 클래스 정보 매핑
            
        Returns:
            Optional[str]: 파일 경로
        """
        # 전체 클래스명으로 찾기
        if full_class_name in class_info_map:
            class_infos = class_info_map[full_class_name]
            if class_infos:
                return class_infos[0]["file_path"]
        
        # 단순 클래스명으로 찾기
        simple_class_name = full_class_name.split('.')[-1]
        if simple_class_name in class_info_map:
            class_infos = class_info_map[simple_class_name]
            # 패키지명이 일치하는 것 우선
            for class_info in class_infos:
                if class_info["full_class_name"] == full_class_name:
                    return class_info["file_path"]
            # 없으면 첫 번째 것
            if class_infos:
                return class_infos[0]["file_path"]
        
        return None
    
    def _build_method_string(
        self,
        namespace: str,
        query_id: str
    ) -> Optional[str]:
        """
        namespace의 class_name + sql query의 id로 method string 조합
        
        Args:
            namespace: Mapper namespace (예: "com.example.UserMapper")
            query_id: SQL query의 id (예: "getUserById")
            
        Returns:
            Optional[str]: method string (예: "UserMapper.getUserById")
        """
        if not namespace or not query_id:
            return None
        
        # namespace에서 마지막 클래스명 추출
        class_name = namespace.split('.')[-1]
        
        return f"{class_name}.{query_id}"
    
    def _find_upper_layer_files(
        self,
        method_string: str
    ) -> List[tuple[str, str]]:
        """
        Call Graph에서 method string과 일치하는 부분을 찾아 root까지 상위 layer로 거슬러 올라가면서
        layer 이름과 file_path가 모두 존재하는 경우를 수집
        
        Args:
            method_string: 메서드 시그니처 (예: "UserMapper.getUserById")
            
        Returns:
            List[tuple[str, str]]: (layer, file_path) 튜플 리스트
        """
        result = []
        
        if not self.call_graph_builder or not self.call_graph_builder.call_graph:
            return result
        
        call_graph = self.call_graph_builder.call_graph
        method_metadata = self.call_graph_builder.method_metadata
        
        # method_string이 call graph에 있는지 확인
        if method_string not in call_graph:
            # 정확히 일치하지 않으면 부분 매칭 시도
            for node in call_graph.nodes():
                if method_string in node or node.endswith(f".{method_string.split('.')[-1]}"):
                    method_string = node
                    break
            else:
                return result
        
        # 역방향으로 탐색 (이 메서드를 호출하는 상위 메서드들)
        visited = set()
        
        def traverse_up(node: str, depth: int = 0, max_depth: int = 20):
            """역방향으로 탐색하여 root까지 올라가기"""
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            # 현재 노드의 메타데이터에서 layer와 file_path 추출
            metadata = method_metadata.get(node, {})
            layer = self.call_graph_builder._get_layer(node)
            file_path = metadata.get("file_path", "")
            
            # layer와 file_path가 모두 있으면 결과에 추가
            if layer and file_path and layer != "Unknown":
                result.append((layer.lower(), file_path))
            
            # 이 노드를 호출하는 상위 노드들 찾기 (predecessors)
            if call_graph.has_node(node):
                predecessors = list(call_graph.predecessors(node))
                for predecessor in predecessors:
                    traverse_up(predecessor, depth + 1, max_depth)
        
        # 시작 노드부터 역방향 탐색
        traverse_up(method_string)
        
        return result
    
    def _determine_main_layer(
        self,
        layer_files: Dict[str, Set[str]]
    ) -> str:
        """
        주요 레이어 결정 (가장 많은 파일이 있는 레이어)
        
        Args:
            layer_files: 레이어별 파일 집합
            
        Returns:
            str: 주요 레이어명
        """
        if not layer_files:
            return "Unknown"
        
        # 파일 수가 가장 많은 레이어
        max_layer = max(layer_files.items(), key=lambda x: len(x[1]))
        return max_layer[0].capitalize() if max_layer[0] else "Unknown"
    
    def _extract_used_columns(
        self,
        sql_queries: List[Dict[str, Any]],
        table_name: str,
        config_columns: Set[str]
    ) -> Set[str]:
        """
        SQL 쿼리에서 사용된 칼럼 추출 (설정된 칼럼 중에서만)
        
        Args:
            sql_queries: SQL 쿼리 목록
            table_name: 테이블명
            config_columns: 설정된 칼럼 목록
            
        Returns:
            Set[str]: 사용된 칼럼 목록
        """
        used_columns = set()
        
        for sql_query_info in sql_queries:
            sql = sql_query_info.get("sql", "")
            if not sql:
                continue
            
            # SQL에서 칼럼 추출
            sql_columns = self.sql_strategy.extract_column_names(sql, table_name)
            sql_columns_lower = {c.lower() for c in sql_columns}
            
            # 설정된 칼럼과 교집합
            used_columns.update(config_columns.intersection(sql_columns_lower))
        
        return used_columns
