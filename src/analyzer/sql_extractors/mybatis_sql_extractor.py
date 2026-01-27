"""
MyBatis SQL Extractor

MyBatis XML Mapper 파일에서 SQL을 추출하는 구현 클래스입니다.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, override

from config.config_manager import Configuration
from models.source_file import SourceFile
from models.sql_extraction_output import SQLExtractionOutput
from parser.xml_mapper_parser import XMLMapperParser

from ..llm_sql_extractor.llm_sql_extractor import LLMSQLExtractor
from ..sql_extractor import SQLExtractor


class MyBatisSQLExtractor(SQLExtractor):
    """
    MyBatis SQL Extractor 구현 클래스

    MyBatis XML Mapper 파일에서 SQL을 추출합니다.
    config.use_llm_parser가 True인 경우 LLMSQLExtractor를 사용합니다.
    """

    def __init__(
        self,
        config: Configuration,
        xml_parser: XMLMapperParser = None,
        java_parse_results: List[dict] = None,
        call_graph_builder = None,
    ):
        """
        MyBatisSQLExtractor 초기화

        Args:
            config: 설정 객체
            xml_parser: XML Mapper 파서
            java_parse_results: Java 파싱 결과 리스트 (사용하지 않지만 호환성을 위해 유지)
            call_graph_builder: CallGraphBuilder 인스턴스 (선택적)
        """
        super().__init__(config=config, xml_parser=xml_parser, java_parse_results=java_parse_results, call_graph_builder=call_graph_builder)
        self.logger = logging.getLogger(__name__)

    def get_layer_name(self) -> str:
        """
        MyBatis 레이어명 반환 (서브클래스에서 오버라이드 가능)

        Returns:
            str: 레이어명 (기본값: "repository")
        """
        return "repository"

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
    def extract_sqls(
        self, source_files: List[SourceFile]
    ) -> List[SQLExtractionOutput]:
        """
        MyBatis 전략: XML Mapper 파일에서 SQL 추출

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SQLExtractionOutput]: 추출 결과
        """
        from models.sql_extraction_output import ExtractedSQLQuery

        results = []
        
        for xml_file in source_files:
            try:
                parse_result = self.xml_parser.parse_mapper_file(xml_file.path)
                if parse_result.get("error"):
                    continue

                # SQL 쿼리들을 strategy 중립적인 형식으로 변환
                sql_queries = []
                for query in parse_result.get("sql_queries", []):
                    # strategy_specific에 MyBatis 특정 정보 저장
                    # 제네릭 타입에서 내부 클래스 타입 추출
                    parameter_type = self._extract_generic_inner_type(query.get("parameter_type"))
                    result_type = self._extract_generic_inner_type(query.get("result_type"))
                    result_map = self._extract_generic_inner_type(query.get("result_map"))

                    strategy_specific = {
                        "namespace": query.get("namespace", ""),
                        "parameter_type": parameter_type,
                        "result_type": result_type,
                        "result_map": result_map,
                        # resultMap 내부 필드 매핑 (SELECT용)
                        "result_field_mappings": query.get("result_field_mappings", []),
                        # SQL 내 #{fieldName} 패턴 (INSERT/UPDATE용)
                        "parameter_field_mappings": query.get("parameter_field_mappings", []),
                    }

                    sql_queries.append(
                        ExtractedSQLQuery(
                            id=query.get("id", ""),
                            query_type=query.get("query_type", "SELECT"),
                            sql=query.get("sql", ""),
                            strategy_specific=strategy_specific,
                        )
                    )

                if sql_queries:
                    results.append(
                        SQLExtractionOutput(file=xml_file, sql_queries=sql_queries)
                    )

            except Exception as e:
                self.logger.warning(f"XML 파일 추출 실패: {xml_file.path} - {e}")

        return results

    @override
    def filter_sql_files(self, source_files: List[SourceFile]) -> List[SourceFile]:
        """
        MyBatis 관련 파일 필터링 (*mapper.xml)

        Args:
            source_files: 소스 파일 목록

        Returns:
            List[SourceFile]: 필터링된 파일 목록
        """
        filtered = []
        for f in source_files:
            name_lower = f.filename.lower()
            # Filter for *mapper.xml
            if f.extension == ".xml" and name_lower.endswith("mapper.xml"):
            # if f.extension == ".xml":
                filtered.append(f)
        return filtered

    @override
    def get_class_files_from_sql_query(
        self, sql_query: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Set[str]], Set[str]]:
        """
        SQL 쿼리에서 관련 클래스 파일 목록 추출

        Args:
            sql_query: SQL 쿼리 정보 딕셔너리

        Returns:
            Tuple[Optional[str], Dict[str, Set[str]], Set[str]]: (method_string, layer_files, all_files) 튜플
                - method_string: 메서드 시그니처 문자열 (예: "UserMapper.getUserById")
                - layer_files: 레이어별 파일 경로 집합을 담은 딕셔너리
                - all_files: 모든 관련 파일 경로 집합
        """
        layer_files: Dict[str, Set[str]] = defaultdict(set)
        all_files: Set[str] = set()

        layer_name = self.get_layer_name()

        strategy_specific = sql_query.get("strategy_specific", {})
        
        # MyBatis: namespace, parameter_type, result_type, result_map 사용
        namespace = strategy_specific.get("namespace", "")
        if namespace:
            interface_file = self._find_class_file(namespace)
            if interface_file:
                layer_files[layer_name].add(interface_file)
                all_files.add(interface_file)

        parameter_type = strategy_specific.get("parameter_type")
        if parameter_type:
            parameter_file = self._find_class_file(parameter_type)
            if parameter_file:
                layer_files[layer_name].add(parameter_file)
                all_files.add(parameter_file)

        result_type = strategy_specific.get("result_type")
        if result_type:
            dao_file = self._find_class_file(result_type)
            if dao_file:
                layer_files[layer_name].add(dao_file)
                all_files.add(dao_file)

        result_map = strategy_specific.get("result_map")
        if result_map:
            result_map_file = self._find_class_file(result_map)
            if result_map_file:
                layer_files[layer_name].add(result_map_file)
                all_files.add(result_map_file)

        # method_string 생성: namespace의 class_name + sql query의 id
        method_string = None
        query_id = sql_query.get("id", "")
        if namespace and query_id:
            # namespace에서 마지막 클래스명 추출
            class_name = namespace.split(".")[-1]
            method_string = f"{class_name}.{query_id}"

        return method_string, layer_files, all_files

