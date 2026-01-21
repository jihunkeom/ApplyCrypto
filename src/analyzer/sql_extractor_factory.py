"""
SQL Extractor Factory

config.sql_wrapping_type에 따라 적절한 SQLExtractor 인스턴스를 생성하는 Factory 클래스입니다.
"""

from typing import Any, List, Optional

from config.config_manager import Configuration
from parser.xml_mapper_parser import XMLMapperParser

from .sql_extractor import SQLExtractor


class SQLExtractorFactory:
    """SQLExtractor 생성을 위한 팩토리 클래스"""

    @staticmethod
    def create(
        config: Configuration,
        xml_parser: Optional[XMLMapperParser] = None,
        java_parse_results: Optional[List[dict]] = None,
        call_graph_builder: Optional[Any] = None,
    ) -> SQLExtractor:
        """
        config.sql_wrapping_type에 따라 적절한 SQLExtractor 인스턴스를 생성합니다.

        Args:
            config: 설정 객체 (sql_wrapping_type 포함)
            xml_parser: XML Mapper 파서 (선택적)
            java_parse_results: Java 파싱 결과 리스트 (선택적)
            call_graph_builder: CallGraphBuilder 인스턴스 (선택적)

        Returns:
            SQLExtractor: 생성된 SQLExtractor 인스턴스

        Raises:
            ValueError: 지원하지 않는 sql_wrapping_type인 경우
        """
        sql_wrapping_type = config.sql_wrapping_type.lower()

        if sql_wrapping_type == "mybatis":
            from .sql_extractors.mybatis_sql_extractor import MyBatisSQLExtractor

            return MyBatisSQLExtractor(
                config=config,
                xml_parser=xml_parser,
                java_parse_results=java_parse_results,
                call_graph_builder=call_graph_builder,
            )

        elif sql_wrapping_type == "mybatis_ccs":
            from .sql_extractors.mybatis_ccs_sql_extractor import MybatisCCSSQLExtractor

            return MybatisCCSSQLExtractor(
                config=config,
                xml_parser=xml_parser,
                java_parse_results=java_parse_results,
                call_graph_builder=call_graph_builder,
            )

        elif sql_wrapping_type == "mybatis_ccs_batch":
            from .sql_extractors.mybatis_ccs_batch_sql_extractor import (
                MyBatisCCSBatchSQLExtractor,
            )

            return MyBatisCCSBatchSQLExtractor(
                config=config,
                xml_parser=xml_parser,
                java_parse_results=java_parse_results,
                call_graph_builder=call_graph_builder,
            )

        elif sql_wrapping_type == "jdbc":
            if "BatBanka" in config.framework_type:
                from .sql_extractors.anyframe_jdbc_bat_sql_extractor import (
                    AnyframeJDBCBatSqlExtractor,
                )

                return AnyframeJDBCBatSqlExtractor(
                    config=config,
                    xml_parser=xml_parser,
                    java_parse_results=java_parse_results,
                    call_graph_builder=call_graph_builder,
                )
            elif "Anyframe" in config.framework_type:
                from .sql_extractors.anyframe_jdbc_sql_extractor import (
                    AnyframeJDBCSQLExtractor,
                )

                return AnyframeJDBCSQLExtractor(
                    config=config,
                    xml_parser=xml_parser,
                    java_parse_results=java_parse_results,
                    call_graph_builder=call_graph_builder,
                )
            else:
                from .sql_extractors.jdbc_sql_extractor import JDBCSQLExtractor

                return JDBCSQLExtractor(
                    config=config,
                    xml_parser=xml_parser,
                    java_parse_results=java_parse_results,
                    call_graph_builder=call_graph_builder,
                )

        elif sql_wrapping_type == "jpa":
            from .sql_extractors.jpa_sql_extractor import JPASQLExtractor

            return JPASQLExtractor(
                config=config,
                xml_parser=xml_parser,
                java_parse_results=java_parse_results,
                call_graph_builder=call_graph_builder,
            )

        else:
            raise ValueError(
                f"지원하지 않는 sql_wrapping_type: {config.sql_wrapping_type}. "
                f"가능한 값: mybatis, mybatis_ccs, mybatis_ccs_batch, jdbc, jpa"
            )
