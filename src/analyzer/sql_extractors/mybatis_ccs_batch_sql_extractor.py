"""
MyBatis CCS Batch SQL Extractor

CCS 배치 프로그램의 *BAT_SQL.xml 파일에서 SQL을 추출하는 구현 클래스입니다.

특징:
    - XML 파일 패턴: *BAT_SQL.xml
    - XML 구조: <sql>/<query id="...">SQL</query></sql>
    - Java 매핑: xxxBAT_SQL.xml → xxxBAT.java (파일명 컨벤션)
    - BATVO 수집: batvo/ 디렉토리에서 관련 VO 파일 수집
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, override

from config.config_manager import Configuration
from models.source_file import SourceFile
from models.sql_extraction_output import ExtractedSQLQuery, SQLExtractionOutput
from parser.xml_mapper_parser import XMLMapperParser

from ..sql_extractor import SQLExtractor


class MyBatisCCSBatchSQLExtractor(SQLExtractor):
    """
    CCS 배치용 SQL Extractor 구현 클래스

    CCS 배치 프로그램의 *BAT_SQL.xml 파일에서 SQL을 추출합니다.
    XML 구조는 <sql>/<query id="...">SQL</query></sql> 형태입니다.
    """

    def __init__(
        self,
        config: Configuration,
        xml_parser: XMLMapperParser = None,
        java_parse_results: List[dict] = None,
        call_graph_builder=None,
    ):
        """
        MyBatisCCSBatchSQLExtractor 초기화

        Args:
            config: 설정 객체
            xml_parser: XML Mapper 파서
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

        # 소스 파일 캐시 (Java 파일 매핑용)
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
        # 소스 파일 캐시 저장 (Java 파일 매핑에 사용)
        self.source_files_cache = source_files

        # 파일 필터링 수행
        filtered_files = self.filter_sql_files(source_files)

        self.logger.info(
            f"MyBatisCCSBatch SQL 파일 필터링 완료: {len(filtered_files)}개 파일"
        )

        if filtered_files:
            return self.extract_sqls(filtered_files)
        return []

    @override
    def filter_sql_files(self, source_files: List[SourceFile]) -> List[SourceFile]:
        """
        CCS 배치 관련 파일 필터링 (*BAT_SQL.xml)

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SourceFile]: 필터링된 파일 목록
        """
        filtered = []
        for f in source_files:
            if f.extension == ".xml":
                name_upper = f.filename.upper()
                # *BAT_SQL.xml 패턴 필터링
                if name_upper.endswith("BAT_SQL.XML"):
                    filtered.append(f)
                    self.logger.debug(f"CCS 배치 SQL 파일 포함: {f.filename}")

        self.logger.info(f"CCS 배치 SQL 파일 필터링 완료: {len(filtered)}개 파일")
        return filtered

    @override
    def extract_sqls(self, source_files: List[SourceFile]) -> List[SQLExtractionOutput]:
        """
        CCS 배치 전략: *BAT_SQL.xml 파일에서 SQL 추출

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SQLExtractionOutput]: 추출 결과
        """
        results = []

        for sql_xml_file in source_files:
            try:
                # *BAT_SQL.xml 파일에서 SQL 추출
                sql_queries_data = self._extract_sql_from_batch_xml(sql_xml_file.path)

                if sql_queries_data:
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
                    self.logger.debug(
                        f"SQL 추출 완료: {sql_xml_file.filename} - {len(sql_queries)}개 쿼리"
                    )

            except Exception as e:
                self.logger.warning(
                    f"배치 SQL XML 파일 추출 실패: {sql_xml_file.path} - {e}"
                )

        return results

    def _extract_sql_from_batch_xml(self, file_path: Path) -> List[dict]:
        """
        배치 *BAT_SQL.xml 파일에서 SQL 쿼리 추출

        XML 구조: <sql>/<query id="...">SQL</query></sql>

        Args:
            file_path: *BAT_SQL.xml 파일 경로

        Returns:
            List[dict]: 추출된 SQL 쿼리 목록
                각 항목은 {"id": str, "query_type": str, "sql": str, "strategy_specific": dict} 형태
        """
        sql_queries = []

        try:
            # XML 파일 파싱
            tree, error = self.xml_parser.parse_file(file_path)
            if error:
                self.logger.warning(f"XML 파싱 실패: {file_path} - {error}")
                return sql_queries

            root = tree.getroot()

            # <query id="..."> 요소들 찾기
            # XPath: .//query[@id] - query 태그 중 id 속성이 있는 것
            for element in root.xpath(".//query[@id]"):
                query_id = element.get("id", "")
                if not query_id:
                    continue

                # SQL 텍스트 추출 (태그 내 모든 텍스트)
                sql_text = "".join(element.itertext()).strip()
                if not sql_text:
                    continue

                # SQL 주석 제거
                clean_sql = self.xml_parser.remove_sql_comments(sql_text)
                if not clean_sql:
                    continue

                # 쿼리 타입 감지
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

    @override
    def get_class_files_from_sql_query(
        self, sql_query: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Set[str]], Set[str]]:
        """
        SQL 쿼리에서 관련 Java 배치 파일 목록 추출

        파일명 컨벤션을 사용하여 Java 파일을 찾습니다:
            - xxxBAT_SQL.xml → xxxBAT.java
            - 같은 디렉토리의 batvo/ 폴더에서 BATVO 파일 수집

        Args:
            sql_query: SQL 쿼리 정보 딕셔너리

        Returns:
            Tuple[Optional[str], Dict[str, Set[str]], Set[str]]: (method_string, layer_files, all_files) 튜플
                - method_string: 메서드 시그니처 문자열 (예: "SampleBAT.execute")
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

        # SQL XML 파일명에서 _SQL.xml 제거하여 기본 이름 추출
        # 예: BCRecdExtnoRegBAT_SQL.xml → BCRecdExtnoRegBAT
        base_name = sql_xml_path.stem  # BCRecdExtnoRegBAT_SQL
        if base_name.upper().endswith("_SQL"):
            base_name = base_name[:-4]  # BCRecdExtnoRegBAT

        # 1. 파일명 컨벤션으로 BAT.java 파일 찾기
        bat_java_file = self._find_bat_java_file(base_name, sql_xml_path)

        if bat_java_file:
            layer_files["bat"].add(bat_java_file)
            all_files.add(bat_java_file)
            self.logger.debug(f"BAT Java 파일 매핑: {sql_xml_path.name} → {bat_java_file}")

        # 2. batvo/ 디렉토리에서 BATVO 파일 수집
        batvo_files = self._find_batvo_files(sql_xml_path.parent)

        for batvo_file in batvo_files:
            layer_files["batvo"].add(batvo_file)
            all_files.add(batvo_file)

        # method_string 생성: ClassName.queryId
        method_string = None
        query_id = sql_query.get("id", "")

        if bat_java_file and query_id:
            class_name = Path(bat_java_file).stem  # BCRecdExtnoRegBAT
            method_string = f"{class_name}.{query_id}"

        return method_string, layer_files, all_files

    def _find_bat_java_file(self, base_name: str, sql_xml_path: Path) -> Optional[str]:
        """
        파일명 컨벤션으로 BAT.java 파일 찾기

        Args:
            base_name: SQL XML 파일명에서 _SQL.xml을 제거한 기본 이름
            sql_xml_path: SQL XML 파일 경로

        Returns:
            Optional[str]: BAT.java 파일 경로
        """
        sql_xml_dir = sql_xml_path.parent

        # 같은 디렉토리에서 {base_name}.java 파일 찾기
        java_file_path = sql_xml_dir / f"{base_name}.java"
        if java_file_path.exists():
            return str(java_file_path)

        # 대소문자 변형 시도
        for variant in [base_name, base_name.upper(), base_name.lower()]:
            java_file_path = sql_xml_dir / f"{variant}.java"
            if java_file_path.exists():
                return str(java_file_path)

        # 소스 파일 캐시에서 찾기
        for source_file in self.source_files_cache:
            if source_file.extension == ".java":
                file_stem = source_file.path.stem
                if file_stem.upper() == base_name.upper():
                    return str(source_file.path)

        return None

    def _find_batvo_files(self, bat_dir: Path) -> List[str]:
        """
        batvo/ 디렉토리에서 BATVO 파일 수집

        Args:
            bat_dir: BAT 파일이 있는 디렉토리

        Returns:
            List[str]: BATVO 파일 경로 목록
        """
        batvo_files = []

        # batvo/ 디렉토리 찾기
        batvo_dir = bat_dir / "batvo"

        if batvo_dir.exists() and batvo_dir.is_dir():
            # batvo 디렉토리 내의 모든 Java 파일 수집
            for java_file in batvo_dir.glob("*.java"):
                batvo_files.append(str(java_file))
                self.logger.debug(f"BATVO 파일 수집: {java_file.name}")

        # 소스 파일 캐시에서 같은 패키지의 BATVO 파일 찾기
        bat_dir_str = str(bat_dir).lower()
        for source_file in self.source_files_cache:
            if source_file.extension == ".java":
                file_path_str = str(source_file.path).lower()
                # batvo 디렉토리 내의 파일인지 확인
                if "batvo" in file_path_str and bat_dir_str in file_path_str:
                    file_path = str(source_file.path)
                    if file_path not in batvo_files:
                        batvo_files.append(file_path)

        return batvo_files

    def get_layer_name(self) -> str:
        """
        CCS 배치 프레임워크에서의 레이어명 반환

        Returns:
            str: 레이어명 ("bat")
        """
        return "bat"
