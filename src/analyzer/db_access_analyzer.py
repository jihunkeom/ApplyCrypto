"""
DB Access Analyzer 모듈

config.json에 설정된 DB 테이블과 칼럼에 접근하는 소스 파일 목록을 작성합니다.
"""

import logging
import re
from collections import defaultdict, deque
from parser.call_graph_builder import CallGraphBuilder
from parser.java_ast_parser import JavaASTParser
from parser.xml_mapper_parser import XMLMapperParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config.config_manager import Configuration
from models.source_file import SourceFile
from models.table_access_info import TableAccessInfo
from persistence.data_persistence_manager import DataPersistenceManager

from .sql_extractor import SQLExtractor


class DBAccessAnalyzer:
    """
    DB Access Analyzer 클래스

    config.json에 설정된 DB 테이블과 칼럼에 접근하는 소스 파일을 식별합니다.
    """

    def __init__(
        self,
        config: Configuration,
        sql_extractor: SQLExtractor,
        xml_parser: Optional[XMLMapperParser] = None,
        java_parser: Optional[JavaASTParser] = None,
        call_graph_builder: Optional[CallGraphBuilder] = None,
    ):
        """
        DBAccessAnalyzer 초기화

        Args:
            config: 설정 객체
            sql_extractor: SQL 추출기 (필수)
            xml_parser: XML Mapper 파서 (선택적, 하위 호환성을 위해 유지)
            java_parser: Java AST 파서 (선택적)
            call_graph_builder: Call Graph Builder (선택적)
        """
        self.config = config
        self.sql_extractor = sql_extractor
        self.xml_parser = xml_parser  # 하위 호환성을 위해 유지하지만 사용하지 않음
        self.java_parser = java_parser or JavaASTParser()
        self.call_graph_builder = call_graph_builder
        self.class_info_map = self.call_graph_builder.get_class_info_map()
        self.logger = logging.getLogger(__name__)

        # 설정에서 테이블 정보 가져오기
        self.access_tables = config.access_tables

        # 테이블별 칼럼 매핑 생성 (대소문자 무시)
        # 칼럼 정보: {column_name: {"new_column": bool}}
        self.table_column_map: Dict[str, Set[str]] = {}
        self.table_column_info: Dict[
            str, Dict[str, Dict[str, Any]]
        ] = {}  # table_name -> {column_name: {"new_column": bool}}

        # SQL 추출 결과 (analyze 메서드에서 로드)
        self.sql_extraction_results: List[Dict[str, Any]] = []

        for table_info in self.access_tables:
            table_name = table_info.table_name.lower()
            columns_set = set()
            column_info_dict = {}

            for col in table_info.columns:
                if isinstance(col, str):
                    # 문자열 형식: "column_name"
                    col_name = col.lower()
                    columns_set.add(col_name)
                    column_info_dict[col_name] = {"new_column": False}
                else:
                    # 객체 형식: ColumnDetail(name="column_name", new_column=True, column_type="name", encryption_code="P017")
                    col_name = col.name.lower()
                    if col_name:
                        columns_set.add(col_name)
                        col_detail = {"new_column": col.new_column}
                        # column_type과 encryption_code가 있으면 포함
                        if col.column_type:
                            col_detail["column_type"] = col.column_type
                        if col.encryption_code:
                            col_detail["encryption_code"] = col.encryption_code
                        column_info_dict[col_name] = col_detail

            self.table_column_map[table_name] = columns_set
            self.table_column_info[table_name] = column_info_dict

    def analyze(self, source_files: List[SourceFile]) -> List[TableAccessInfo]:
        """
        소스 파일들을 분석하여 테이블 접근 정보 추출

        Args:
            source_files: 분석할 소스 파일 목록

        Returns:
            List[TableAccessInfo]: 테이블 접근 정보 목록
        """

        persistence_manager = DataPersistenceManager(Path(self.config.target_project))
        self.sql_extraction_results = persistence_manager.load_from_file(
            "sql_extraction_results.json"
        )

        if not self.sql_extraction_results:
            self.logger.warning(
                "sql_extraction_results.json을 찾을 수 없습니다. 먼저 analyze 명령어를 실행하세요."
            )
            return []

        # config.json에 설정된 각 DB 테이블에 대해 분석
        table_access_info_list = []

        for table_name in self.table_column_map.keys():
            # self.table_column_map에서 칼럼 목록 가져오기
            columns = self.table_column_map.get(table_name, set())

            # 규칙 1: 테이블만 있고 칼럼 정보가 없는 테이블은 무시
            if not columns:
                self.logger.debug(
                    f"테이블 '{table_name}'은 칼럼 정보가 없어 무시됩니다."
                )
                continue

            # 테이블별 접근 정보 수집
            table_info = self._analyze_table_access(table_name, columns, source_files)

            if table_info:
                table_access_info_list.append(table_info)

        return table_access_info_list

    def _analyze_table_access(
        self, table_name: str, columns: Set[str], source_files: List[SourceFile]
    ) -> Optional[TableAccessInfo]:
        """
        특정 테이블에 대한 접근 정보 분석

        Args:
            table_name: 테이블명
            columns: 칼럼 목록
            source_files: 소스 파일 목록
            sql_extraction_results: SQL 추출 결과 목록 (선택적, 없으면 self.sql_extraction_results 사용)

        Returns:
            Optional[TableAccessInfo]: 테이블 접근 정보
        """

        # 수집된 정보
        sql_queries: List[Dict[str, Any]] = []
        interface_files: Set[str] = set()
        dao_files: Set[str] = set()
        layer_files: Dict[str, Set[str]] = defaultdict(set)  # layer -> file_paths
        all_added_files: Set[str] = (
            set()
        )  # 모든 레이어에 추가된 파일 추적 (중복 방지용)

        # 각 파일의 SQL 쿼리에 대해 분석
        for file_result in self.sql_extraction_results:
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
                true_columns,
            )

            if not matching_queries:
                continue

            # 각 SQL 쿼리에 대해 처리
            for sql_query_info in matching_queries:
                # enriched_query 생성 - source_file_path 추가
                enriched_query = dict(sql_query_info)
                enriched_query["source_file_path"] = file_path
                sql_queries.append(enriched_query)

                # sql query에서 사용되는 클래스 파일 경로 정보 추출 (sql_wrapping_type에 따라 다름 )
                method_string, query_layer_files, query_added_files = (
                    self.sql_extractor.get_class_files_from_sql_query(sql_query_info)
                )

                if query_layer_files:
                    for layer, files in query_layer_files.items():
                        layer_files[layer].update(files)

                if query_added_files:
                    all_added_files.update(query_added_files)

                if (
                    method_string
                    and self.call_graph_builder
                    and self.call_graph_builder.call_graph
                ):
                    # Call Graph에서 역방향으로 탐색하여 상위 layer 파일 찾기
                    upper_layer_files, reduced_call_stacks = (
                        self._find_upper_layer_files(method_string)
                    )
                    for layer, layer_file_path in upper_layer_files:
                        if (
                            layer
                            and layer_file_path
                            and layer_file_path not in all_added_files
                        ):
                            layer_files[layer].add(layer_file_path)
                            all_added_files.add(layer_file_path)

                    # call_stacks 생성 (endpoint에서 target까지의 경로)
                    call_stacks = self._expand_call_stacks(
                        reduced_call_stacks, method_string
                    )
                    enriched_query["call_stacks"] = call_stacks

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
        # 칼럼 목록 추출 (SQL 쿼리에서 사용된 칼럼) - 현재는 사용되지 않음 (디버깅용)
        # self._extract_used_columns(sql_queries, table_name, columns)

        # 칼럼 정보 생성 (config.json에 설정된 모든 칼럼 포함)
        # SQL 쿼리에서 사용된 칼럼은 실제로 사용된 것으로 표시
        columns_list = []
        table_lower = table_name.lower()
        column_info_dict = self.table_column_info.get(table_lower, {})

        # config.json에 설정된 모든 칼럼을 포함 (column_type, encryption_code 포함)
        for col_name, col_info in sorted(column_info_dict.items()):
            col_entry = {
                "name": col_name,
                "new_column": col_info.get("new_column", False),
            }
            # column_type과 encryption_code가 있으면 포함
            if "column_type" in col_info:
                col_entry["column_type"] = col_info["column_type"]
            if "encryption_code" in col_info:
                col_entry["encryption_code"] = col_info["encryption_code"]
            columns_list.append(col_entry)

        # layer_files를 딕셔너리로 변환 (Set -> List)
        layer_files_dict = {
            layer: sorted(list(files)) for layer, files in layer_files.items()
        }

        table_access_info = TableAccessInfo(
            table_name=table_name,
            columns=columns_list,  # 칼럼 정보 목록 (칼럼명과 new_column 정보 포함)
            access_files=sorted(list(all_access_files)),
            query_type=sql_queries[0].get("query_type", "SELECT")
            if sql_queries
            else "SELECT",
            sql_query=sql_queries[0].get("sql", "") if sql_queries else None,
            layer=main_layer,
            sql_queries=sql_queries,  # SQL 쿼리 목록 저장
            layer_files=layer_files_dict,  # 레이어별 파일 경로 목록 저장
        )

        return table_access_info

    def _get_column_groups(
        self, table_name: str, columns: Set[str]
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
        true_columns = set()  # new_column=true인 칼럼

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
        true_columns: Set[str],
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

        # TODO: 김한섭 부장님 수정한 코드 확인할 것.

        for sql_query_info in sql_queries:
            sql = sql_query_info.get("sql", "")
            if not sql:
                continue

            # 테이블명 확인
            tables = self.sql_extractor.extract_table_names(sql)
            if table_name.lower() not in {t.lower() for t in tables}:
                continue

            # SQL에서 칼럼 추출
            sql_columns = self.sql_extractor.extract_column_names(sql, table_name)

            # sql_columns가 비어있고 SELECT * FROM 패턴인 경우 false_columns로 설정
            if not sql_columns:
                # SELECT ~ FROM 패턴 확인 (대소문자 무시, 여러 줄 지원)
                select_from_pattern = r"SELECT\s+(.*?)\s+FROM\s+"
                select_match = re.search(
                    select_from_pattern, sql, re.IGNORECASE | re.DOTALL
                )
                if select_match:
                    select_clause = select_match.group(1).strip()
                    # SELECT와 FROM 사이에 *가 포함되어 있는지 확인
                    if "*" in select_clause:
                        # FROM 다음 부분 확인 (서브쿼리가 아닌지)
                        from_after = sql[select_match.end() :].strip()
                        # FROM 다음이 (로 시작하지 않으면
                        if not from_after.startswith("("):
                            sql_columns = false_columns.copy()

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

    def _build_method_string(self, namespace: str, query_id: str) -> Optional[str]:
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
        class_name = namespace.split(".")[-1]

        return f"{class_name}.{query_id}"

    def _find_upper_layer_files(
        self, method_string: str
    ) -> tuple[List[tuple[str, str]], List[str]]:
        """
        Call Graph에서 method string과 일치하는 부분을 찾아 root까지 상위 layer로 거슬러 올라가면서
        layer 이름과 file_path가 모두 존재하는 경우를 수집하고, call_stacks도 수집

        Args:
            method_string: 메서드 시그니처 (예: "UserMapper.getUserById")

        Returns:
            tuple[List[tuple[str, str]], List[str]]: ((layer, file_path) 튜플 리스트, reduced_call_stacks)
        """
        result = []
        reduced_call_stacks: List[str] = []

        if not self.call_graph_builder or not self.call_graph_builder.call_graph:
            return result, reduced_call_stacks

        call_graph = self.call_graph_builder.call_graph
        method_metadata = self.call_graph_builder.method_metadata

        # method_string이 call graph에 있는지 확인
        if method_string not in call_graph:
            # 정확히 일치하지 않으면 부분 매칭 시도
            for node in call_graph.nodes():
                if method_string in node or node.endswith(
                    f".{method_string.split('.')[-1]}"
                ):
                    method_string = node
                    break
            else:
                return result, reduced_call_stacks

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

            # layer와 file_path가 모두 있고 Unknown이 아니면 결과에 추가
            if layer and file_path and layer != "Unknown":
                result.append((layer.lower(), file_path))
                # call_stacks에도 추가 (시작점인 method_string은 항상 포함)
                if node not in reduced_call_stacks:
                    reduced_call_stacks.append(node)

            # 이 노드를 호출하는 상위 노드들 찾기 (predecessors)
            if call_graph.has_node(node):
                predecessors = list(call_graph.predecessors(node))
                for predecessor in predecessors:
                    traverse_up(predecessor, depth + 1, max_depth)

        # 시작 노드(method_string)는 항상 call_stacks에 포함
        reduced_call_stacks.append(method_string)

        # 시작 노드부터 역방향 탐색
        traverse_up(method_string)

        # reduced_call_stacks 역순 정렬 (시작점이 마지막에 오도록)
        reduced_call_stacks.reverse()

        return result, reduced_call_stacks

    def _expand_call_stacks(
        self, reduced_call_stacks: List[str], target_method: str
    ) -> List[List[str]]:
        """
        reduced_call_stacks에서 endpoint를 찾아서 그 endpoint부터 target_method까지의
        경로를 생성하여 call_stacks 리스트를 만듭니다.

        Args:
            reduced_call_stacks: 역방향 탐색으로 수집된 method signature 리스트
            target_method: 시작점인 target method signature

        Returns:
            List[List[str]]: 각 endpoint에서 target까지의 경로 리스트
        """
        if not self.call_graph_builder or not self.call_graph_builder.call_graph:
            return []

        call_graph = self.call_graph_builder.call_graph
        endpoint_method_signatures = (
            self.call_graph_builder.get_endpoint_method_signatures()
        )

        all_call_stacks: List[List[str]] = []

        # reduced_call_stacks에서 endpoint인 것들 찾기
        for method_sig in reduced_call_stacks:
            if method_sig not in endpoint_method_signatures:
                continue

            # BFS로 endpoint에서 target_method까지의 경로 찾기
            queue = deque([(method_sig, [method_sig])])
            visited = {method_sig}

            while queue:
                current_node, path = queue.popleft()

                # target_method에 도달하면 경로 저장
                if current_node == target_method:
                    # Unknown layer가 아닌 노드만 포함
                    filtered_path = []
                    for node in path:
                        layer = self.call_graph_builder._get_layer(node)
                        if layer != "Unknown":
                            filtered_path.append(node)
                    if filtered_path:
                        all_call_stacks.append(filtered_path)
                    continue

                # 다음 노드들 탐색 (successors)
                if call_graph.has_node(current_node):
                    successors = list(call_graph.successors(current_node))
                    for successor in successors:
                        if successor not in visited:
                            visited.add(successor)
                            queue.append((successor, path + [successor]))

        return all_call_stacks

    def _determine_main_layer(self, layer_files: Dict[str, Set[str]]) -> str:
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
        config_columns: Set[str],
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
            sql_columns = self.sql_extractor.extract_column_names(sql, table_name)
            sql_columns_lower = {c.lower() for c in sql_columns}

            # 설정된 칼럼과 교집합
            used_columns.update(config_columns.intersection(sql_columns_lower))

        return used_columns
