"""
Anyframe JDBC Batch SQL Extractor

Anyframe 프레임워크를 사용하는 JDBC Java 파일에서 SQL을 추출하는 구현 클래스입니다.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, override
from collections import defaultdict
from config.config_manager import Configuration
from models.source_file import SourceFile
from models.sql_extraction_output import SQLExtractionOutput
from parser.xml_mapper_parser import XMLMapperParser

from ..llm_sql_extractor.llm_sql_extractor import LLMSQLExtractor
from ..sql_extractor import SQLExtractor


class AnyframeJDBCBatSQLExtractor(SQLExtractor):
    """
    Anyframe JDBC SQL Extractor 구현 클래스

    Anyframe 프레임워크를 사용하는 JDBC Java 파일에서 SQL을 추출합니다.
    config.use_llm_parser가 True인 경우 LLMSQLExtractor를 사용합니다.
    """

    def __init__(
        self,
        config: Configuration,
        xml_parser: XMLMapperParser = None,
        java_parse_results: List[dict] = None,
        call_graph_builder=None,
    ):
        """
        AnyframeJDBCSQLExtractor 초기화

        Args:
            config: 설정 객체
            xml_parser: XML Mapper 파서 (사용하지 않지만 호환성을 위해 유지)
            java_parse_results: Java 파싱 결과 리스트
            call_graph_builder: CallGraphBuilder 인스턴스 (선택적)
        """
        super().__init__(
            config=config,
            xml_parser=xml_parser,
            java_parse_results=java_parse_results,
            call_graph_builder=call_graph_builder,
        )
        self.logger = logging.getLogger(__name__)

        # class_info_map 가져오기
        if self.call_graph_builder:
            self.class_info_map = self.call_graph_builder.get_class_info_map()
        else:
            self.class_info_map = {}
        self.source_files_cache: List[SourceFile] = []

    @override
    def extract_from_files(
        self, source_files: List[SourceFile]
    ) -> List[SQLExtractionOutput]:
        """
        소스 파일들에서 SQL 쿼리 추출

        Args:
            source_files: 분석할 소스 파일 목록

        Returns:
            List[SQLExtractionOutput]: 추출된 SQL 쿼리 정보 목록
        """
        self.source_files_cache = source_files

        # 먼저 파일 필터링 수행
        filtered_files = self.filter_sql_files(source_files)

        if self.config.use_llm_parser:
            # LLM 기반 추출 사용
            if filtered_files:
                llm_extractor = LLMSQLExtractor(
                    llm_provider_name=self.config.llm_provider
                )
                return llm_extractor.extract_from_files(filtered_files)
            return []
        else:
            # 기존 방식 사용 (이미 필터링된 파일 목록 사용)
            return self.extract_sqls(filtered_files)

    @override
    def extract_sqls(self, source_files: List[SourceFile]) -> List[SQLExtractionOutput]:
        """
        Anyframe JDBC 전략: Java 파일에서 Anyframe JDBC SQL 추출

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SQLExtractionOutput]: 추출 결과
        """
        from models.sql_extraction_output import ExtractedSQLQuery

        results = []

        for sql_xml_file in source_files:
            try:
                # *_SQL.xml 파일에서 SQL 추출
                sql_queries_data = self._extract_sql_from_batch_xml(sql_xml_file.path)

                if sql_queries_data:
                    # strategy_specific에 파일 특정 정보 저장
                    sql_queries = []
                    for query in sql_queries_data:
                        sql_queries.append(
                            ExtractedSQLQuery(
                                id=query.get("id", ""),
                                query_type=query.get("query_type", "SELECT"),
                                sql=query.get("sql", ""),
                                strategy_specific=query.get("strategy_specific", {}),
                            )
                        )

                    results.append(
                        SQLExtractionOutput(file=sql_xml_file, sql_queries=sql_queries)
                    )

            except Exception as e:
                self.logger.warning(
                    f"배치 SQL XML 파일 추출 실패: {sql_xml_file.path} - {e}"
                )

        return results

    def _extract_sql_from_batch_xml(self, file_path: Path) -> List[dict]:
        """
        배치 *_SQL.xml 파일에서 sql 쿼리 추출

        Args:
            file_path *_SQL.xml 파일 경로

        Returns:
            List[dict]: 추출된 SQL 쿼리 목록
                각 항복은 {"id": str, "query_type": str, "sql": str, "strategy_specific": dict} 형태
        """
        sql_queries = []

        try:
            # XML 파일 파싱
            tree, error = self.xml_parser.parse_file(file_path)
            if error:
                self.logger.warning(f"XML 파싱 실패: {file_path} - {error}")
                return sql_queries

            root = tree.getroot()

            for element in root.xpath(".//*[@id]"):
                query_id = element.get("id", "")
                if not query_id:
                    continue

                sql_text = "".join(element.itertext()).strip()
                if not sql_text:
                    continue

                clean_sql = self.xml_parser.remove_sql_comments(sql_text)
                if not clean_sql:
                    continue

                query_type = self._detect_query_type(clean_sql) or "SELECT"

                sql_queries.append(
                    {
                        "id": query_id,
                        "query_type": query_type,
                        "sql": clean_sql,
                        "strategy_specific": {"sql_xml_file": str(file_path)},
                    }
                )
        except Exception as e:
            self.logger.warning(f"배치 SQL XML 파일 추출 중 오류: {file_path} - {e}")
        return sql_queries

    def _extract_anyframe_jdbc_sql_from_file(self, file_path: Path) -> List[dict]:
        """
        Anyframe JDBC를 사용하는 Java 파일에서 SQL 쿼리 추출
        java_parse_results에서 클래스 정보를 가져와서 사용

        Args:
            file_path: Java 파일 경로

        Returns:
            List[dict]: 추출된 SQL 쿼리 목록
                각 항목은 {"id": str, "query_type": str, "sql": str, "strategy_specific": dict} 형태
        """
        sql_queries = []

        # java_parse_results에서 해당 파일의 클래스 정보 찾기
        file_path_str = str(file_path)
        classes_info = None

        for parse_result in self.java_parse_results or []:
            result_file = parse_result.get("file", {})
            result_file_path = result_file.get("path", "")

            # 경로 비교 (정규화)
            if Path(result_file_path).resolve() == Path(file_path_str).resolve():
                classes_info = parse_result.get("classes", [])
                break

        if not classes_info:
            self.logger.warning(
                f"java_parse_results에서 클래스 정보를 찾을 수 없습니다: {file_path}"
            )
            return sql_queries

        # 파일 읽기 (메서드 body 추출을 위해 필요)
        source_code = None
        encodings = ["utf-8", "euc-kr", "cp949", "latin-1", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    source_code = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue

        if not source_code:
            return sql_queries

        # 각 클래스의 메서드 처리
        for class_info in classes_info:
            class_name = class_info.get("name", "")
            methods = class_info.get("methods", [])

            for method_info in methods:
                method_name = method_info.get("name", "")
                if not method_name:
                    continue

                # 메서드 body 추출 (정규식으로 메서드 시그니처부터 찾기)
                # 메서드 시그니처 패턴: 접근제어자 반환타입 메서드명(파라미터)
                # 반환 타입은 제네릭 타입(List<T>, Map<K,V> 등)을 포함할 수 있음
                # 예: public List<AdBkgImgDVO> selBkgImgList(...)
                # 제네릭 타입을 포함한 반환 타입 매칭
                # 패턴: (접근제어자)? (반환타입) 메서드명(
                # 반환타입: \w+(?:<[^>]+>)? 또는 void 등
                method_signature_pattern = rf"(?:public|private|protected)?\s+(?:\w+(?:<[^>]+>)?|void)\s+{re.escape(method_name)}\s*\("
                method_match = re.search(method_signature_pattern, source_code)

                # 위 패턴이 실패하면 더 유연한 패턴 시도 (제네릭이 아닌 경우)
                if not method_match:
                    method_signature_pattern = rf"(?:public|private|protected)?\s+\w+\s+{re.escape(method_name)}\s*\("
                    method_match = re.search(method_signature_pattern, source_code)

                if not method_match:
                    continue

                method_start = method_match.start()

                # 메서드 끝 찾기 (다음 메서드 또는 클래스 끝)
                # 중괄호 매칭을 사용하여 메서드 body 추출
                brace_count = 0
                method_end = method_start
                in_method = False

                for i in range(method_start, len(source_code)):
                    char = source_code[i]
                    if char == "{":
                        brace_count += 1
                        in_method = True
                    elif char == "}":
                        brace_count -= 1
                        if in_method and brace_count == 0:
                            method_end = i + 1
                            break

                if method_end <= method_start:
                    continue

                method_body = source_code[method_start:method_end]

                # method body 내에서 대소문자 구분 없이 SQL을 포함하는 StringBuilder 변수 찾기
                # StringBuilder 변수 선언 패턴 (대소문자 구분 없음)
                # 변수명에 "sql"이 포함된 경우만 찾기
                stringbuilder_pattern = r"(?i)StringBuilder\s+(\w*[Ss][Qq][Ll]\w*)\s*=\s*new\s+StringBuilder\s*\(\)"
                sb_matches = re.finditer(stringbuilder_pattern, method_body)

                for sb_match in sb_matches:
                    var_name = sb_match.group(1)

                    # 해당 변수의 append 호출들 찾기 (대소문자 구분 없음)
                    # <var_name>.append("...") 또는 <var_name>.append('...')
                    append_pattern = rf'(?i){re.escape(var_name)}\s*\.\s*append\s*\(\s*["\']([^"\']*)["\']'
                    append_matches = re.finditer(append_pattern, method_body, re.DOTALL)

                    # SQL 조각 수집
                    sql_parts = []
                    for append_match in append_matches:
                        sql_part = append_match.group(1)
                        # 이스케이프된 문자 처리 (\n, \t 등)
                        sql_part = (
                            sql_part.replace("\\n", "\n")
                            .replace("\\t", "\t")
                            .replace("\\r", "\r")
                        )
                        sql_parts.append(sql_part)

                    if sql_parts:
                        # StringBuilder의 toString() 호출 위치 찾기 (대소문자 구분 없음)
                        # <var_name>.toString()
                        tostring_pattern = (
                            rf"(?i){re.escape(var_name)}\s*\.\s*toString\s*\("
                        )
                        tostring_match = re.search(tostring_pattern, method_body)

                        if tostring_match:
                            # toString() 호출 이후의 코드에서 메서드 호출 확인
                            # queryForObject, queryForList, update 등의 메서드 호출 찾기
                            after_tostring = method_body[tostring_match.end() :]

                            # SQL 조합
                            combined_sql = "".join(sql_parts).strip()

                            if combined_sql:
                                # SQL 내용으로 타입 결정
                                query_type = self._detect_query_type(combined_sql)
                                if not query_type:
                                    query_type = "SELECT"  # 기본값

                                # parameter_type 찾기: method_info.parameters 중에서 type을 key로 해서 self.class_info_map에서 full_class_name 찾기
                                parameter_type = None
                                parameters = method_info.get("parameters", [])
                                for param in parameters:
                                    param_type = param.get("type", "")
                                    if param_type:
                                        # 제네릭 타입 처리 (예: List<AdBkgImgDVO> -> AdBkgImgDVO)
                                        param_type = self._extract_generic_inner_type(
                                            param_type
                                        )

                                        # self.class_info_map에서 찾기
                                        if (
                                            param_type
                                            and param_type in self.class_info_map
                                        ):
                                            class_infos = self.class_info_map[
                                                param_type
                                            ]
                                            if class_infos:
                                                parameter_type = class_infos[0].get(
                                                    "full_class_name"
                                                )
                                                break

                                # result_type: method_info.return_type 값을 가져와서 (제네릭 타입인 경우 내부 type을 꺼내야 함) self.class_info_map에서 full_class_name 찾기
                                result_type = None
                                return_type = method_info.get("return_type", "")
                                if return_type:
                                    # 제네릭 타입 처리 (예: List<AdBkgImgDVO> -> AdBkgImgDVO)
                                    return_type = self._extract_generic_inner_type(
                                        return_type
                                    )

                                    # self.class_info_map에서 찾기
                                    if (
                                        return_type
                                        and return_type in self.class_info_map
                                    ):
                                        class_infos = self.class_info_map[return_type]
                                        if class_infos:
                                            result_type = class_infos[0].get(
                                                "full_class_name"
                                            )

                                sql_queries.append(
                                    {
                                        "id": method_name,
                                        "query_type": query_type,
                                        "sql": combined_sql,
                                        "strategy_specific": {
                                            "class_name": class_name,
                                            "parameter_type": parameter_type,
                                            "result_type": result_type,
                                        },
                                    }
                                )

        return sql_queries

    @override
    def filter_sql_files(self, source_files: List[SourceFile]) -> List[SourceFile]:
        """
        Anyframe JDBC 관련 파일 필터링 (DEM.java 또는 DQM.java로 끝나는 Java 파일)

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SourceFile]: 필터링된 파일 목록
        """
        filtered = []
        for f in source_files:
            if f.extension == ".xml":
                try:
                    file_name = f.path.name.upper()
                    # *_SQL.xml 파일인지 확인
                    if file_name.endswith("_SQL.XML"):
                        filtered.append(f)
                except Exception as e:
                    self.logger.warning(f"Failed to check file name for {f.path}: {e}")
        return filtered

    @override
    def get_class_files_from_sql_query(
        self, sql_query: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Set[str]], Set[str]]:
        """
        SQL 쿼리에서 관련 Java 배치 파일 목록 추출
        3가지 방법으로 Java 배치 파일을 찾습니다:
        1. Java 파일에서 SetQueryPath를 통해 직접 사용하는 경우
        2. *_CFG.xml 파일에서 query-file 속성으로 선언된 경우
        3. _SQL을 제거한 샅은 파일명의 Java에서 사용하는 경우

        Args:
            sql_query: SQL 쿼리 정보 딕셔너리

        Returns:
            Tuple[Optional[str], Dict[str, Set[str]], Set[str]]: (method_string, layer_files, all_files) 튜플
                - method_string: 메서드 시그니처 문자열 (예: "ClassName.methodName")
                - layer_files: 레이어별 파일 경로 집합을 담은 딕셔너리
                - all_files: 모든 관련 파일 경로 집합
        """
        layer_files: Dict[str, Set[str]] = defaultdict(set)
        all_files: Set[str] = set()

        strategy_specific = sql_query.get("strategy_specific", {})

        sql_xml_file = strategy_specific.get("sql_xml_file", "")

        if not sql_xml_file:
            return None, layer_files, all_files

        sql_xml_path = Path(sql_xml_file)
        sql_xml_name = sql_xml_path.name

        # SQL XML 파일명에서 _SQL.xml 제거하여 기본 이름 추출
        base_name = sql_xml_name.replace("_SQL.xml", "").replace("_SQL.XML", "")

        # 방법 1: Java 파일에서 setQueryPath를 통해 직접 사용하는 경우
        java_files_method1 = self._find_java_files_by_setquerypath(
            sql_xml_name, sql_xml_path
        )

        # 방법 2: *_CFG.xml 파일에 query-file 속성으로 선언된 경우
        java_files_method2 = self._find_java_files_by_cfg_xml(
            sql_xml_name, sql_xml_path
        )

        # 방법 3: _SQL을 제거한 같은 파일명의 Java에서 사용하는 경우
        java_files_method3 = self._find_java_files_by_name_convention(
            base_name, sql_xml_path
        )

        # 모든 방법에서 찾은 Java 파일 수집
        all_java_files = set[Any]()
        all_java_files.update(java_files_method1)
        all_java_files.update(java_files_method2)
        all_java_files.update(java_files_method3)

        for java_file in all_java_files:
            layer_files["Batch"].add(java_file)
            all_files.add(java_file)

        # method_string 생성: class_name.method_name
        method_string = None

        if all_java_files:
            first_java_file = list[Any](all_java_files)[0]
            class_name = self._extract_class_name_from_java_file(first_java_file)
            query_id = sql_query.get("id", "")
        if class_name and query_id:
            method_string = f"{class_name}.{query_id}"

        return method_string, layer_files, all_files

    def _find_java_files_by_setquerypath(
        self, sql_xml_name: str, sql_xml_path: Path
    ) -> Set[str]:
        """
        방법 1: Java 파일에서 setQueryPath를 통해 직접 사용하는 경우 찾기

        Args:
            sql_xml_name: SQL XML 파일명
            sql_xml_path: SQL XML 파일 경로

        Returns:
            Set[str]: 찾은 Java 파일 경로 집합
        """
        java_files = set[Any]()

        # SQL XML 파일명에서 경로 부분 추출
        # 또는 전체 파일명만 사용
        sql_xml_name_only = sql_xml_path.name

        # 모든 Java 파일 검색
        for source_file in self.source_files_cache:
            if source_file.extension != ".java":
                continue

            try:
                # 파일 읽기
                encodings = ["utf-8", "euc-kr", "cp949", "latin-1", "iso-8859-1"]
                source_code = None

                for encoding in encodings:
                    try:
                        with open(source_file.path, "r", encoding=encoding) as f:
                            source_code = f.read()
                        break
                    except (UnicodeDecodeError, Exception):
                        continue

                    if not source_code:
                        continue

                    # setQueryPath 패턴 찾기
                    pattern = (
                        rf"setQueryPath\s*\(\s*[^)]*{re.escape(sql_xml_name_only)}"
                    )

                    if re.search(pattern, source_code, re.IGNORECASE):
                        java_files.add(str(source_file.path))
                        self.logger.debug(
                            f"방법 1로 찾음: {source_file.path} -> {sql_xml_name}"
                        )
            except Exception as e:
                self.logger.warning(f"Java 파일 검색 중 오류: {source_file.path} - {e}")

        return java_files

    def _find_java_files_by_cfg_xml(
        self, sql_xml_name: str, sql_xml_path: Path
    ) -> Set[str]:
        """
        방법 2: *_CFG_XML 파일에 query-file 속성으로 선언된 경우 찾기

        Args:
            sql_xml_name: SQL XML 파일명
            sql_xml_path: SQL XML 파일 경로

        Returns:
            Set[str]: 찾은 Java 파일 경로 집합
        """
        java_files = set[Any]()

        # 모든 소스 파일에서 *_CFG.xml 파일 찾기
        cfg_files = []

        for source_file in self.source_files_cache:
            if source_file.extension == ".xml":
                file_name = source_file.path.name.upper()
                if file_name.endswith("_CFG.XML"):
                    cfg_files.append(source_file.path)

        # SQL XML 파일의 디렉토리에서도 찾기
        sql_xml_dir = sql_xml_path.parent
        for cfg_file in sql_xml_dir.glob("*_CFG.xml"):
            if cfg_file not in cfg_files:
                cfg_files.append(cfg_file)

        # 각 CFG XML 파일 검색
        for cfg_file in cfg_files:
            try:
                # XML 파일 파싱
                tree, error = self.xml_parser.parse_file(cfg_file)
                if error:
                    continue

                root = tree.getroot()

                # query-file 속성을 가진 요소 찾기
                elements_with_query_file = root.xpath(".//*[@query-file]")

                for element in elements_with_query_file:
                    query_file = element.get("query-file", "")
                    if not query_file:
                        continue

                    # query-file 값이 SQL XML 파일명과 일치하는지 확인
                    query_file_name = (
                        Path(query_file).name
                        if "/" in query_file or "\\" in query_file
                        else query_file
                    )
                    if (
                        sql_xml_name == query_file_name
                        or sql_xml_name.upper() == query_file_name.upper()
                        or sql_xml_name in query_file
                        or sql_xml_name.upper() in query_file.upper()
                    ):
                        class_name = element.get("class", "")
                        if class_name:
                            java_file = self._find_java_file_by_class_name(class_name)
                            if java_file:
                                java_files.add(java_file)
                                self.logger.debug(
                                    f"방법 2로 찾음: {cfg_file} -> {java_file}"
                                )
            except Exception as e:
                self.logger.warning(f"CFG XML 파일 파싱 중 오류: {cfg_file} - {e}")

        return java_files

    def _find_java_files_by_name_convention(
        self, base_name: str, sql_xml_path: Path
    ) -> Set[str]:
        """
        방법 3: _SQL을 제거한 같은 파일명의 Java에서 사용하는 경우 찾기

        Args:
            base_name: SQL XML 파일명에서 _SQL.xml을 제거한 기본 이름
            sql_xml_path: SQL XML 파일 경로

        Returns:
            Set[str]: 찾은 Java 파일 경로 집합
        """
        java_files = set[Any]()

        # 같은 디렉토리에서 {base_name}.java 파일 찾기
        sql_xml_dir = sql_xml_path.parent

        java_file_path = sql_xml_dir / f"{base_name}.java"
        if java_file_path.exists():
            java_files.add(str(java_file_path))
            self.logger.debug(f"방법 3으로 찾음: {java_file_path}")

        java_file_path_upper = sql_xml_dir / f"{base_name.upper()}.java"
        if (
            java_file_path_upper.exists()
            and str(java_file_path_upper) not in java_files
        ):
            java_files.add(str(java_file_path_upper))
            self.logger.debug(f"방법 3으로 찾음: {java_file_path_upper}")

        java_file_path_lower = sql_xml_dir / f"{base_name.lower()}.java"
        if (
            java_file_path_lower.exists()
            and str(java_file_path_lower) not in java_files
        ):
            java_files.add(str(java_file_path_lower))
            self.logger.debug(f"방법 3으로 찾음: {java_file_path_lower}")

        return java_files

    def _find_java_file_by_class_name(self, class_name: str) -> Optional[str]:
        """
        클래스명으로 Java 파일 찾기

        Args:
            class_name: 클래스명 (전체 패키지명 포함 또는 단순 클래스명)

        Returns:
            Optional[str]: Java 파일 경로
        """
        # 전체 클래스명으로 찾기
        if class_name in self.class_info_map:
            class_infos = self.class_info_map[class_name]
            if class_infos:
                return class_infos[0].get("file_path")

        # 단순 클래스명으로 찾기
        simple_class_name = class_name.split(".")[-1]

        if simple_class_name in self.class_info_map:
            class_infos = self.class_info_map[simple_class_name]

            for class_info in class_infos:
                if class_info.get("full_class_name") == class_name:
                    return class_info.get("file_path")
            if class_infos:
                return class_infos[0].get("file_path")

        simple_class_name_java = f"{simple_class_name}.java"

        for source_file in self.source_files_cache:
            if (
                source_file.path.name == simple_class_name_java
                or source_file.path.name == simple_class_name_java.upper()
            ):
                return str(source_file.path)
        return None

    def _extract_class_name_from_java_file(self, java_file_path: str) -> Optional[str]:
        """
        Java 파일에서 클래스명 추출

        Args:
            java_file_path: Java 파일 경로

        Returns:
            Optional[str]: 클래스명
        """
        # class_info_map에서 찾기
        for class_name, class_infos in self.class_info_map.items():
            for class_info in class_infos:
                if class_info.get("file_path") == java_file_path:
                    return class_info.get("class_name")

        java_path = Path(java_file_path)
        class_name = java_path.stem
        return class_name
