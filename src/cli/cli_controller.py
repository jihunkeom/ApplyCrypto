"""
CLI Controller 모듈

argparse를 사용하여 CLI 기본 구조를 구축하고, analyze, list, modify 명령어와 각 옵션을 파싱합니다.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    from anytree import Node, RenderTree
except ImportError:
    Node = None
    RenderTree = None

from parser.call_graph_builder import CallGraphBuilder
from parser.java_ast_parser import JavaASTParser
from parser.xml_mapper_parser import XMLMapperParser

from analyzer.db_access_analyzer import DBAccessAnalyzer
from analyzer.sql_extractor import SQLExtractor
from analyzer.sql_parsing_strategy import create_strategy
from collector.source_file_collector import SourceFileCollector
from config.config_manager import Configuration, ConfigurationError
from config.config_manager import load_config as load_global_config
from models.modification_record import ModificationRecord
from models.source_file import SourceFile
from models.table_access_info import TableAccessInfo
from modifier.code_modifier import CodeModifier
from persistence.cache_manager import CacheManager
from persistence.data_persistence_manager import (
    DataPersistenceManager,
    PersistenceError,
)


class CLIController:
    """
    CLI 명령어를 파싱하고 실행하는 컨트롤러 클래스

    주요 기능:
    1. 명령어 정의: analyze, list, modify 명령어 구현
    2. 옵션 파싱: argparse를 사용하여 각 명령의 옵션 정의
    3. 도움말 메시지: 각 명령과 옵션에 대한 명확한 설명 제공
    4. 에러 처리: 잘못된 명령어 및 옵션에 대한 에러 메시지
    5. 진행 상황 표시: 장시간 작업 시 진행률 표시
    6. 로깅: 모든 작업을 로그 파일에 기록
    """

    def __init__(self):
        """CLIController 초기화"""
        self.parser = self._create_parser()
        self.logger = self._setup_logging()
        self.config: Optional[Configuration] = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        argparse 파서 생성 및 서브파서 설정

        Returns:
            argparse.ArgumentParser: 설정된 메인 파서
        """
        # 메인 파서 생성
        parser = argparse.ArgumentParser(
            prog="applycrypto",
            description="Java Spring Boot 프로젝트 암호화 자동 적용 도구",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
예제:
  %(prog)s analyze --config config.json
  %(prog)s list --all
  %(prog)s list --db
  %(prog)s modify --config config.json --dry-run
            """,
        )

        # 서브파서 생성
        subparsers = parser.add_subparsers(
            dest="command", help="사용 가능한 명령어", metavar="COMMAND"
        )

        # analyze 명령어 서브파서
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="프로젝트를 분석하여 소스 파일, Call Graph, DB 접근 정보를 수집합니다",
            description="프로젝트를 분석하여 소스 파일, Call Graph, DB 접근 정보를 수집합니다.",
        )
        analyze_parser.add_argument(
            "--config",
            type=str,
            default="config.json",
            help="설정 파일 경로 (기본값: config.json)",
        )

        # list 명령어 서브파서
        list_parser = subparsers.add_parser(
            "list",
            help="수집된 정보를 조회합니다",
            description="수집된 정보를 조회합니다. 하나 이상의 옵션을 지정할 수 있습니다.",
        )
        list_parser.add_argument(
            "--config",
            type=str,
            default="config.json",
            help="설정 파일 경로 (기본값: config.json)",
        )
        list_group = list_parser.add_mutually_exclusive_group()
        list_group.add_argument(
            "--all", action="store_true", help="수집된 모든 소스 파일 목록을 출력합니다"
        )
        list_group.add_argument(
            "--db", action="store_true", help="테이블별 접근 파일 목록을 출력합니다"
        )
        list_group.add_argument(
            "--modified", action="store_true", help="수정된 파일 목록을 출력합니다"
        )
        list_group.add_argument(
            "--endpoint",
            action="store_true",
            help="REST API 엔드포인트 목록을 출력합니다",
        )
        list_parser.add_argument(
            "--callgraph",
            type=str,
            metavar="ENDPOINT",
            help="특정 엔드포인트의 호출 그래프를 출력합니다",
        )

        # modify 명령어 서브파서
        modify_parser = subparsers.add_parser(
            "modify",
            help="식별된 파일에 암복호화 코드를 삽입합니다",
            description="식별된 파일에 암복호화 코드를 삽입합니다.",
        )
        modify_parser.add_argument(
            "--config",
            type=str,
            default="config.json",
            help="설정 파일 경로 (기본값: config.json)",
        )
        modify_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="실제 파일 수정 없이 미리보기만 수행합니다",
        )
        modify_parser.add_argument(
            "--all",
            action="store_true",
            help="사용자 확인 없이 모든 변경사항을 자동으로 적용합니다",
        )

        return parser

    def _setup_logging(self) -> logging.Logger:
        """
        로깅 설정

        Returns:
            logging.Logger: 설정된 로거
        """
        # 로그 디렉터리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 로그 파일 경로 (타임스탬프 포함)
        log_file = (
            log_dir / f"applycrypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # 로거 설정
        logger = logging.getLogger("applycrypto")
        logger.setLevel(logging.DEBUG)

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        명령줄 인자 파싱

        Args:
            args: 파싱할 인자 리스트 (None이면 sys.argv 사용)

        Returns:
            argparse.Namespace: 파싱된 인자

        Raises:
            SystemExit: 잘못된 인자 또는 명령어가 없을 때
        """
        parsed_args = self.parser.parse_args(args)

        # 명령어가 지정되지 않은 경우 도움말 출력
        if not parsed_args.command:
            self.parser.print_help()
            sys.exit(1)

        return parsed_args

    def load_config(self, config_path: str) -> Configuration:
        """
        설정 파일 로드

        Args:
            config_path: 설정 파일 경로

        Returns:
            Configuration: 로드된 설정 객체

        Raises:
            ConfigurationError: 설정 파일 로드 실패 시
        """
        try:
            self.config = load_global_config(config_path)
            self.logger.info(f"설정 파일 로드 성공: {config_path}")
            return self.config
        except ConfigurationError as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            raise

    def execute(self, args: Optional[List[str]] = None) -> int:
        """
        CLI 명령어 실행

        Args:
            args: 명령줄 인자 리스트 (None이면 sys.argv 사용)

        Returns:
            int: 종료 코드 (0: 성공, 1: 실패, 2: 인자 오류)
        """
        try:
            parsed_args = self.parse_args(args)
            self.logger.info(f"명령어 실행: {parsed_args.command}")

            # 명령어별 핸들러 호출
            if parsed_args.command == "analyze":
                return self._handle_analyze(parsed_args)
            elif parsed_args.command == "list":
                return self._handle_list(parsed_args)
            elif parsed_args.command == "modify":
                return self._handle_modify(parsed_args)
            else:
                self.logger.error(f"알 수 없는 명령어: {parsed_args.command}")
                return 1

        except SystemExit as e:
            # argparse가 발생시킨 SystemExit (잘못된 인자 등)
            return e.code if e.code is not None else 2
        except KeyboardInterrupt:
            self.logger.warning("사용자에 의해 중단되었습니다")
            return 1
        except Exception as e:
            self.logger.exception(f"명령어 실행 중 오류 발생: {e}")
            return 1

    def _handle_analyze(self, args: argparse.Namespace) -> int:
        """
        analyze 명령어 핸들러

        Args:
            args: 파싱된 인자

        Returns:
            int: 종료 코드
        """
        try:
            # 설정 파일 로드
            config = self.load_config(args.config)
            target_project = Path(config.target_project)

            self.logger.info("프로젝트 분석 시작...")
            print("프로젝트 분석을 시작합니다...")

            # Data Persistence Manager 초기화
            persistence_manager = DataPersistenceManager(target_project)
            cache_manager = persistence_manager.cache_manager or CacheManager()

            # 1. 소스 파일 수집
            print("  [1/5] 소스 파일 수집 중...")
            self.logger.info("소스 파일 수집 시작")
            collector = SourceFileCollector(config)
            source_files = list[SourceFile](collector.collect())
            print(f"  ✓ {len(source_files)}개의 소스 파일을 수집했습니다.")
            self.logger.info(f"소스 파일 수집 완료: {len(source_files)}개")

            # 소스 파일 저장
            persistence_manager.save_to_file(
                [f.to_dict() for f in source_files], "source_files.json"
            )

            # 2. Java AST 파싱 및 Call Graph 생성 (파싱 결과 재사용을 위해 먼저 수행)
            print("  [2/5] Java AST 파싱 및 Call Graph 생성 중...")
            self.logger.info("Java AST 파싱 및 Call Graph 생성 시작")
            java_parser = JavaASTParser(cache_manager=cache_manager)
            java_files = [f.path for f in source_files if f.extension == ".java"]

            # Call Graph 생성 (내부에서 모든 Java 파일 파싱)
            call_graph_builder = CallGraphBuilder(
                java_parser=java_parser, cache_manager=cache_manager
            )
            call_graph_builder.build_call_graph(java_files)
            endpoints = call_graph_builder.get_endpoints()

            # 파싱된 결과를 사용하여 Java 파싱 결과 생성
            java_parse_results = []
            for java_file_path in java_files:
                classes = call_graph_builder.get_classes_for_file(java_file_path)
                if classes:
                    # SourceFile 객체 찾기
                    source_file = next(
                        (f for f in source_files if f.path == java_file_path), None
                    )
                    if source_file:
                        java_parse_results.append(
                            {
                                "file": source_file.to_dict(),
                                "classes": [cls.to_dict() for cls in classes],
                            }
                        )

            print(f"  ✓ {len(java_parse_results)}개의 Java 파일을 파싱했습니다.")
            print(f"  ✓ {len(endpoints)}개의 엔드포인트를 식별했습니다.")
            self.logger.info(
                f"Java AST 파싱 및 Call Graph 생성 완료: {len(java_parse_results)}개 파일, {len(endpoints)}개 엔드포인트"
            )

            # 3. SQL Parsing Strategy 초기 생성
            print("  [3/5] SQL 추출 중...")
            self.logger.info("SQL 추출 시작")
            sql_wrapping_type = config.sql_wrapping_type
            sql_strategy = create_strategy(sql_wrapping_type)

            # SQL Extractor 초기화
            xml_parser = XMLMapperParser()
            sql_extractor = SQLExtractor(
                strategy=sql_strategy, xml_parser=xml_parser, java_parser=java_parser
            )

            # SQL 추출 실행
            sql_extraction_results = sql_extractor.extract_from_files(source_files)
            print(f"  ✓ {len(sql_extraction_results)}개의 파일에서 SQL을 추출했습니다.")

            total_sql_queries = sum(
                len(r.get("sql_queries", [])) for r in sql_extraction_results
            )
            print(f"  ✓ 총 {total_sql_queries}개의 SQL 쿼리를 추출했습니다.")
            self.logger.info(
                f"SQL 추출 완료: {len(sql_extraction_results)}개 파일, {total_sql_queries}개 쿼리"
            )

            # Java 파싱 결과 저장
            persistence_manager.save_to_file(
                java_parse_results, "java_parse_results.json"
            )

            # SQL 추출 결과 저장 (xml_parse_results.json 대신)
            persistence_manager.save_to_file(
                sql_extraction_results, "sql_extraction_results.json"
            )

            # Call Graph 저장 (endpoint별 call tree 포함)
            call_graph_data = {
                "endpoints": [
                    ep.to_dict() if hasattr(ep, "to_dict") else str(ep)
                    for ep in endpoints
                ],
                "node_count": call_graph_builder.call_graph.number_of_nodes()
                if call_graph_builder.call_graph
                else 0,
                "edge_count": call_graph_builder.call_graph.number_of_edges()
                if call_graph_builder.call_graph
                else 0,
                "call_trees": call_graph_builder.get_all_call_trees(
                    max_depth=20
                ),  # 모든 엔드포인트의 call tree
            }
            persistence_manager.save_to_file(call_graph_data, "call_graph.json")

            # 5. DB 접근 정보 분석
            print("  [5/5] DB 접근 정보 분석 중...")
            self.logger.info("DB 접근 정보 분석 시작")
            db_analyzer = DBAccessAnalyzer(
                config=config,
                sql_strategy=sql_strategy,
                xml_parser=xml_parser,
                java_parser=java_parser,
                call_graph_builder=call_graph_builder,
            )
            table_access_info_list = db_analyzer.analyze(source_files)
            print(
                f"  ✓ {len(table_access_info_list)}개의 테이블 접근 정보를 분석했습니다."
            )
            self.logger.info(f"DB 접근 정보 분석 완료: {len(table_access_info_list)}개")

            # DB 접근 정보 저장
            persistence_manager.save_to_file(
                [info.to_dict() for info in table_access_info_list],
                "table_access_info.json",
            )

            print("\n분석이 완료되었습니다.")
            print(f"  - 수집된 파일: {len(source_files)}개")
            print(f"  - Java 파일: {len(java_parse_results)}개")
            print(f"  - SQL 추출 파일: {len(sql_extraction_results)}개")
            print(f"  - 엔드포인트: {len(endpoints)}개")
            print(f"  - 테이블 접근 정보: {len(table_access_info_list)}개")
            self.logger.info("프로젝트 분석 완료")
            return 0

        except ConfigurationError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            self.logger.exception(f"analyze 명령어 실행 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)
            return 1

    def _handle_list(self, args: argparse.Namespace) -> int:
        """
        list 명령어 핸들러

        Args:
            args: 파싱된 인자

        Returns:
            int: 종료 코드
        """
        try:
            # 옵션 검증
            has_option = (
                args.all or args.db or args.modified or args.endpoint or args.callgraph
            )
            if not has_option:
                print(
                    "오류: list 명령어에는 하나 이상의 옵션(--all, --db, --modified, --endpoint, --callgraph)이 필요합니다.",
                    file=sys.stderr,
                )
                print("도움말을 보려면: applycrypto list --help", file=sys.stderr)
                return 1

            self.logger.info("정보 조회 시작...")

            # 기본 설정 파일 경로 시도
            config = self.load_config(args.config)
            target_project = Path(config.target_project)

            # DataPersistenceManager 초기화 (output_dir은 현재 작업 디렉터리에 생성됨)
            persistence_manager = DataPersistenceManager(target_project)

            if args.all:
                self._list_all_files(persistence_manager)

            if args.db:
                self._list_db_access(persistence_manager)

            if args.modified:
                self._list_modified_files(persistence_manager)

            if args.endpoint:
                self._list_endpoints(persistence_manager)

            if args.callgraph:
                self._list_callgraph(args.callgraph, persistence_manager)

            self.logger.info("정보 조회 완료")
            return 0

        except Exception as e:
            self.logger.exception(f"list 명령어 실행 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)
            return 1

    def _list_all_files(
        self, persistence_manager: Optional[DataPersistenceManager]
    ) -> None:
        """모든 소스 파일 목록 출력"""
        try:
            if not persistence_manager:
                print(
                    "분석 결과를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return

            source_files_data = persistence_manager.load_from_file(
                "source_files.json", SourceFile
            )
            if not source_files_data:
                print("수집된 소스 파일이 없습니다.")
                return

            source_files = [
                SourceFile.from_dict(f) if isinstance(f, dict) else f
                for f in source_files_data
            ]

            # 테이블 데이터 준비
            table_data = []
            for f in source_files:
                table_data.append(
                    [
                        f.filename,
                        str(f.relative_path),
                        f"{f.size:,} bytes",
                        f.modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                        f.extension,
                    ]
                )

            if tabulate:
                print("\n모든 소스 파일 목록:")
                print(
                    tabulate(
                        table_data,
                        headers=["파일명", "경로", "크기", "수정 시간", "확장자"],
                        tablefmt="grid",
                    )
                )
            else:
                print("\n모든 소스 파일 목록:")
                for row in table_data:
                    print(f"  {row[0]} ({row[1]})")

            print(f"\n총 {len(source_files)}개의 파일")

        except PersistenceError as e:
            print(f"오류: {e}", file=sys.stderr)
        except Exception as e:
            self.logger.exception(f"파일 목록 조회 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)

    def _list_db_access(
        self, persistence_manager: Optional[DataPersistenceManager]
    ) -> None:
        """테이블별 접근 파일 목록 출력"""
        try:
            if not persistence_manager:
                print(
                    "분석 결과를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return

            table_access_data = persistence_manager.load_from_file(
                "table_access_info.json", TableAccessInfo
            )
            if not table_access_data:
                print("테이블 접근 정보가 없습니다.")
                return

            table_access_list = [
                TableAccessInfo.from_dict(t) if isinstance(t, dict) else t
                for t in table_access_data
            ]

            # 테이블별로 그룹화
            table_data = []
            for info in table_access_list:
                table_data.append(
                    [
                        info.table_name,
                        len(info.access_files),
                        ", ".join(
                            [
                                col.get("name", col) if isinstance(col, dict) else col
                                for col in info.columns[:3]
                            ]
                        )
                        + ("..." if len(info.columns) > 3 else ""),
                        info.layer,
                        info.query_type,
                    ]
                )

            if tabulate:
                print("\n테이블별 접근 파일 목록:")
                print(
                    tabulate(
                        table_data,
                        headers=[
                            "테이블명",
                            "접근 파일 수",
                            "칼럼 (일부)",
                            "레이어",
                            "쿼리 타입",
                        ],
                        tablefmt="grid",
                    )
                )
            else:
                print("\n테이블별 접근 파일 목록:")
                for row in table_data:
                    print(f"  {row[0]}: {row[1]}개 파일")

            # 테이블별 접근 파일 경로 상세 출력
            print("\n" + "=" * 80)
            print("테이블별 접근 파일 경로 상세:")
            print("=" * 80)
            for info in table_access_list:
                print(f"\n테이블: {info.table_name} ({len(info.access_files)}개 파일)")
                print(f"  레이어: {info.layer}")
                print(f"  쿼리 타입: {info.query_type}")
                # columns는 이제 객체 배열이므로 이름만 추출
                column_names = [
                    col.get("name", col) if isinstance(col, dict) else col
                    for col in info.columns
                ]
                print(f"  칼럼: {', '.join(column_names) if column_names else 'N/A'}")
                print("  접근 파일:")
                if info.access_files:
                    for file_path in info.access_files:
                        print(f"    - {file_path}")
                else:
                    print("    (접근 파일 없음)")

            print(f"\n총 {len(table_access_list)}개의 테이블")

        except PersistenceError as e:
            print(f"오류: {e}", file=sys.stderr)
        except Exception as e:
            self.logger.exception(f"DB 접근 정보 조회 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)

    def _list_modified_files(
        self, persistence_manager: Optional[DataPersistenceManager]
    ) -> None:
        """수정된 파일 목록 출력"""
        try:
            if not persistence_manager:
                print(
                    "분석 결과를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return

            # 수정 기록 파일 찾기
            try:
                modified_data = persistence_manager.load_from_file(
                    "modification_records.json", ModificationRecord
                )
            except PersistenceError:
                print("수정된 파일이 없습니다.")
                return

            if not modified_data:
                print("수정된 파일이 없습니다.")
                return

            modified_records = [
                ModificationRecord.from_dict(m) if isinstance(m, dict) else m
                for m in modified_data
            ]

            # 테이블 데이터 준비
            table_data = []
            for record in modified_records:
                table_data.append(
                    [
                        Path(record.file_path).name,
                        record.table_name,
                        record.column_name,
                        len(record.modified_methods),
                        record.status,
                        record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )

            if tabulate:
                print("\n수정된 파일 목록:")
                print(
                    tabulate(
                        table_data,
                        headers=[
                            "파일명",
                            "테이블명",
                            "칼럼명",
                            "수정된 메서드 수",
                            "상태",
                            "수정 시간",
                        ],
                        tablefmt="grid",
                    )
                )
            else:
                print("\n수정된 파일 목록:")
                for row in table_data:
                    print(f"  {row[0]} ({row[1]}.{row[2]}) - {row[3]}개 메서드")

            print(f"\n총 {len(modified_records)}개의 수정 기록")

        except PersistenceError as e:
            print(f"오류: {e}", file=sys.stderr)
        except Exception as e:
            self.logger.exception(f"수정 파일 목록 조회 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)

    def _list_endpoints(
        self, persistence_manager: Optional[DataPersistenceManager]
    ) -> None:
        """REST API 엔드포인트 목록 출력"""
        try:
            if not persistence_manager:
                print(
                    "분석 결과를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return

            call_graph_data = persistence_manager.load_from_file("call_graph.json")
            if not call_graph_data or "endpoints" not in call_graph_data:
                print("엔드포인트 정보가 없습니다.")
                return

            endpoints = call_graph_data["endpoints"]
            if not endpoints:
                print("엔드포인트가 없습니다.")
                return

            # Endpoint 객체로 변환
            from parser.call_graph_builder import Endpoint

            endpoint_objects = []
            for ep in endpoints:
                if isinstance(ep, dict):
                    endpoint_objects.append(Endpoint.from_dict(ep))
                elif isinstance(ep, Endpoint):
                    endpoint_objects.append(ep)
                else:
                    # 문자열인 경우 스킵
                    continue

            if not endpoint_objects:
                print("유효한 엔드포인트가 없습니다.")
                return

            # 테이블 데이터 준비
            table_data = []
            for ep in endpoint_objects:
                table_data.append(
                    [ep.http_method, ep.path, ep.method_signature, ep.class_name]
                )

            if tabulate:
                print("\nREST API 엔드포인트 목록:")
                print(
                    tabulate(
                        table_data,
                        headers=["HTTP 메서드", "경로", "메서드 시그니처", "클래스명"],
                        tablefmt="grid",
                    )
                )
            else:
                print("\nREST API 엔드포인트 목록:")
                for row in table_data:
                    print(f"  {row[0]} {row[1]} -> {row[2]} ({row[3]})")

            print(f"\n총 {len(endpoint_objects)}개의 엔드포인트")
            print("\n호출 그래프를 보려면: list --callgraph <method_signature>")
            print("예시: list --callgraph EmpController.login")

        except PersistenceError as e:
            print(f"오류: {e}", file=sys.stderr)
        except Exception as e:
            self.logger.exception(f"엔드포인트 목록 조회 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)

    def _list_callgraph(
        self, endpoint: str, persistence_manager: Optional[DataPersistenceManager]
    ) -> None:
        """특정 엔드포인트의 호출 그래프 출력"""
        try:
            if not persistence_manager:
                print(
                    "분석 결과를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return

            # Call Graph 데이터 로드
            call_graph_data = persistence_manager.load_from_file("call_graph.json")
            if not call_graph_data:
                print("Call Graph 데이터가 없습니다.")
                return

            # 엔드포인트 찾기
            endpoints = call_graph_data.get("endpoints", [])
            from parser.call_graph_builder import Endpoint

            # Endpoint 객체로 변환
            endpoint_objects = []
            for ep in endpoints:
                if isinstance(ep, dict):
                    endpoint_objects.append(Endpoint.from_dict(ep))
                elif isinstance(ep, Endpoint):
                    endpoint_objects.append(ep)

            target_endpoint_obj = None
            method_sig = None

            # 엔드포인트 매칭
            for ep in endpoint_objects:
                method_sig = ep.method_signature
                if (
                    endpoint in method_sig
                    or method_sig.endswith(endpoint)
                    or method_sig == endpoint
                ):
                    target_endpoint_obj = ep
                    break

            if not target_endpoint_obj:
                print(f"엔드포인트 '{endpoint}'를 찾을 수 없습니다.")
                print("\n사용 가능한 엔드포인트 (method_signature 형식):")
                for ep in endpoint_objects[:10]:  # 처음 10개 표시
                    print(f"  - {ep.method_signature}")
                if len(endpoint_objects) > 10:
                    print(f"  ... 외 {len(endpoint_objects) - 10}개")
                print("\n사용법: list --callgraph <method_signature>")
                print("예시: list --callgraph EmpController.login")
                print("     list --callgraph login  (부분 매칭도 가능)")
                return

            cache_manager = persistence_manager.cache_manager or CacheManager()
            java_parser = JavaASTParser(cache_manager=cache_manager)
            call_graph_builder = CallGraphBuilder(
                java_parser=java_parser, cache_manager=cache_manager
            )

            # Call Graph 재생성 (캐시에서 로드)
            try:
                source_files_data = persistence_manager.load_from_file(
                    "source_files.json", SourceFile
                )
                source_files = [
                    SourceFile.from_dict(f) if isinstance(f, dict) else f
                    for f in source_files_data
                ]
                java_files = [f for f in source_files if f.extension == ".java"]
                call_graph_builder.build_call_graph(java_files)
            except Exception as e:
                self.logger.warning(f"Call Graph 재생성 실패: {e}")

            # 호출 트리 출력
            if target_endpoint_obj:
                print(
                    f"\n엔드포인트 '{target_endpoint_obj.method_signature}'의 호출 그래프:"
                )
                print("=" * 60)
                # print_call_tree는 Endpoint 객체를 받음
                call_graph_builder.print_call_tree(
                    target_endpoint_obj, show_layers=True, max_depth=10
                )

        except PersistenceError as e:
            print(f"오류: {e}", file=sys.stderr)
        except Exception as e:
            self.logger.exception(f"호출 그래프 조회 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)

    def _print_diff(self, file_path: str, diff: str):
        """Diff 내용을 컬러로 출력 (간단 구현)"""
        print(f"\n[Diff] {file_path}")
        print("-" * 80)
        for line in diff.splitlines():
            if line.startswith("+"):
                print(f"\033[92m{line}\033[0m")  # Green
            elif line.startswith("-"):
                print(f"\033[91m{line}\033[0m")  # Red
            elif line.startswith("@@"):
                print(f"\033[96m{line}\033[0m")  # Cyan
            else:
                print(line)
        print("-" * 80)

    def _get_user_confirmation(self) -> str:
        """사용자 확인 입력"""
        while True:
            choice = input(
                "\n이 변경사항을 적용하시겠습니까? [y/n/a/q] (y:적용, n:건너뛰기, a:모두적용, q:중단): "
            ).lower()
            if choice in ["y", "n", "a", "q"]:
                return choice
            print("잘못된 입력입니다.")

    def _handle_modify(self, args: argparse.Namespace) -> int:
        """
        modify 명령어 핸들러
        """
        try:
            # 설정 파일 로드
            config = self.load_config(args.config)

            # Type Handler 모드 분기
            if config.type_handler:
                self.logger.info("Type Handler 모드로 수정을 진행합니다.")
                return self._handle_modify_with_type_handler(args, config)

            # Call Chain 모드 분기
            # Call Chain 모드 분기
            if (
                getattr(config, "use_call_chain_mode", False)
                or getattr(config, "diff_gen_type", "") == "call_chain"
            ):
                self.logger.info("Call Chain 모드로 수정을 진행합니다.")
                return self._handle_modify_with_call_chain(args, config)

            # 기존 로직 (직접 코드 수정 방식)
            target_project = Path(config.target_project)

            mode = "미리보기" if args.dry_run else "실제 수정"
            self.logger.info(f"파일 수정 시작 (모드: {mode})...")
            print(f"파일 수정을 시작합니다 (모드: {mode})...")

            # Data Persistence Manager 초기화
            persistence_manager = DataPersistenceManager(target_project)

            # 분석 결과 확인 및 로드
            print("  [1/2] 분석 결과 확인 중...")
            try:
                table_access_info_data = persistence_manager.load_from_file(
                    "table_access_info.json", TableAccessInfo
                )
                if not table_access_info_data:
                    print(
                        "  오류: 테이블 접근 정보를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                    )
                    return 1

                table_access_info_list = []
                for info in table_access_info_data:
                    if isinstance(info, dict):
                        table_access_info_list.append(TableAccessInfo.from_dict(info))
                    elif isinstance(info, TableAccessInfo):
                        table_access_info_list.append(info)

                if not table_access_info_list:
                    print("  수정할 테이블 접근 정보가 없습니다.")
                    return 0

                print(
                    f"  ✓ {len(table_access_info_list)}개의 테이블 접근 정보를 로드했습니다."
                )

            except PersistenceError:
                print(
                    "  오류: 테이블 접근 정보를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return 1

            # CodeModifier 초기화
            code_modifier = CodeModifier(
                config=config, project_root=Path(target_project)
            )

            print("  [2/2] 수정 계획 생성 및 적용 중...")

            total_success = 0
            total_failed = 0
            total_skipped = 0
            apply_all = args.all

            for table_info in table_access_info_list:
                print(f"\n  테이블 '{table_info.table_name}' 처리 중...")

                # 트래킹 시작 (테이블 단위)
                code_modifier.result_tracker.start_tracking()

                # 1. 계획 생성
                try:
                    plans = code_modifier.generate_modification_plans(table_info)
                except Exception as e:
                    self.logger.error(f"계획 생성 실패: {e}")
                    print(f"    ✗ 계획 생성 실패: {e}")
                    total_failed += len(table_info.access_files)  # 대략적인 수치
                    continue

                table_modifications = []

                # 2. 계획 적용 (Interactive loop)
                for plan in plans:
                    if plan.status in ["failed", "skipped"]:
                        # 이미 실패/스킵된 계획은 그대로 적용(기록)
                        res = code_modifier.apply_modification_plan(
                            plan, dry_run=args.dry_run
                        )
                        table_modifications.append(res)
                        if plan.status == "failed":
                            total_failed += 1
                        else:
                            total_skipped += 1
                        continue

                    # Diff 출력 및 사용자 확인
                    if not apply_all:
                        self._print_diff(plan.file_path, plan.unified_diff)
                        choice = self._get_user_confirmation()

                        if choice == "q":
                            print("작업을 중단합니다.")
                            return 0
                        elif choice == "a":
                            apply_all = True
                        elif choice == "n":
                            print("  -> 건너뜀")
                            # 스킵 기록
                            plan.status = "skipped"
                            plan.reason = "User skipped"
                            res = code_modifier.apply_modification_plan(
                                plan, dry_run=args.dry_run
                            )
                            table_modifications.append(res)
                            total_skipped += 1
                            continue

                    # 적용
                    res = code_modifier.apply_modification_plan(
                        plan, dry_run=args.dry_run
                    )
                    table_modifications.append(res)

                    if res.get("status") == "success":
                        print(f"  -> 적용 완료: {Path(plan.file_path).name}")
                        total_success += 1
                    else:
                        print(f"  -> 실패: {res.get('error')}")
                        total_failed += 1

                # 결과 추적 및 저장 (테이블 단위)
                code_modifier.result_tracker.end_tracking()
                code_modifier.result_tracker.update_table_access_info(
                    table_info, table_modifications
                )
                code_modifier.result_tracker.save_modification_history(
                    table_info.table_name, table_modifications
                )

            # 최종 저장
            code_modifier.result_tracker.save_statistics()

            # 수정된 table_access_info 저장
            persistence_manager.save_to_file(
                [info.to_dict() for info in table_access_info_list],
                "table_access_info.json",
            )

            # 통계 출력
            print("\n모든 작업이 완료되었습니다.")
            print(f"  - 성공: {total_success}개")
            print(f"  - 실패: {total_failed}개")
            print(f"  - 건너뜀: {total_skipped}개")

            if args.dry_run:
                print("\n[미리보기 모드] 실제 파일은 수정되지 않았습니다.")

            return 0

        except ConfigurationError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            self.logger.exception(f"modify 명령어 실행 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)
            return 1

    def _handle_modify_with_type_handler(
        self, args: argparse.Namespace, config: Configuration
    ) -> int:
        """
        Type Handler 방식으로 암복호화를 적용하는 핸들러

        Type Handler를 사용하면 Java 비즈니스 로직을 직접 수정하지 않고,
        MyBatis TypeHandler 클래스를 생성하여 XML 매퍼에 등록합니다.

        Args:
            args: 파싱된 인자
            config: 설정 객체

        Returns:
            int: 종료 코드
        """
        try:
            from generator.type_handler_generator import TypeHandlerGenerator

            self.logger.info("Type Handler Generator 초기화...")
            generator = TypeHandlerGenerator(config)

            return generator.execute(dry_run=args.dry_run, apply_all=args.all)

        except ImportError as e:
            self.logger.error(f"Type Handler Generator 모듈을 로드할 수 없습니다: {e}")
            print(
                f"오류: Type Handler Generator 모듈을 로드할 수 없습니다: {e}",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            self.logger.exception(f"Type Handler 수정 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)
            return 1

    def _handle_modify_with_call_chain(
        self, args: argparse.Namespace, config: Configuration
    ) -> int:
        """
        Call Chain 방식으로 암복호화를 적용하는 핸들러

        Call Chain 모드를 사용하면 레이어별 배치 처리 대신
        호출 체인(Controller → Service → Repository) 단위로 LLM을 호출하여
        가장 적절한 레이어에 암복호화 코드를 삽입합니다.

        Args:
            args: 파싱된 인자
            config: 설정 객체

        Returns:
            int: 종료 코드
        """
        try:
            from modifier.call_chain_processor import CallChainProcessor

            target_project = config.target_project

            mode = "미리보기" if args.dry_run else "실제 수정"
            self.logger.info(f"Call Chain 모드로 파일 수정 시작 (모드: {mode})...")
            print(f"Call Chain 모드로 파일 수정을 시작합니다 (모드: {mode})...")

            # Data Persistence Manager 초기화
            persistence_manager = DataPersistenceManager(target_project)

            # 분석 결과 확인 및 로드
            print("  [1/3] 분석 결과 확인 중...")
            try:
                # table_access_info 로드
                table_access_info_data = persistence_manager.load_from_file(
                    "table_access_info.json", TableAccessInfo
                )
                if not table_access_info_data:
                    print(
                        "  오류: 테이블 접근 정보를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                    )
                    return 1

                table_access_info_list = []
                for info in table_access_info_data:
                    if isinstance(info, dict):
                        table_access_info_list.append(TableAccessInfo.from_dict(info))
                    elif isinstance(info, TableAccessInfo):
                        table_access_info_list.append(info)

                if not table_access_info_list:
                    print("  수정할 테이블 접근 정보가 없습니다.")
                    return 0

                print(
                    f"  ✓ {len(table_access_info_list)}개의 테이블 접근 정보를 로드했습니다."
                )

                # call_graph 로드
                call_graph_data = persistence_manager.load_from_file("call_graph.json")
                if not call_graph_data:
                    print(
                        "  오류: Call Graph 정보를 찾을 수 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                    )
                    return 1

                print(
                    f"  ✓ Call Graph 로드 완료 (엔드포인트: {len(call_graph_data.get('endpoints', []))}개)"
                )

            except PersistenceError as e:
                print(f"  오류: 분석 결과를 로드할 수 없습니다: {e}")
                return 1

            # CallChainProcessor 초기화
            print("  [2/3] Call Chain Processor 초기화 중...")
            processor = CallChainProcessor(
                config=config, project_root=Path(target_project)
            )

            # 처리 실행
            print("  [3/3] Call Chain 처리 중...")
            result = processor.process_all(
                table_access_info_list=table_access_info_list,
                call_graph_data=call_graph_data,
                dry_run=args.dry_run,
                apply_all=args.all,
            )

            # 결과 출력
            if result.get("success"):
                stats = result.get("statistics", {})
                print("\n모든 작업이 완료되었습니다.")
                print(f"  - 총 체인 수: {stats.get('total_chains', 0)}개")
                print(f"  - 처리된 체인: {stats.get('processed_chains', 0)}개")
                print(f"  - 성공: {stats.get('success', 0)}개")
                print(f"  - 실패: {stats.get('failed', 0)}개")
                print(f"  - 스킵: {stats.get('skipped', 0)}개")

                if args.dry_run:
                    print("\n[미리보기 모드] 실제 파일은 수정되지 않았습니다.")

                return 0
            else:
                print(f"\n오류: {result.get('error', '알 수 없는 오류')}")
                return 1

        except ImportError as e:
            self.logger.error(f"Call Chain Processor 모듈을 로드할 수 없습니다: {e}")
            print(
                f"오류: Call Chain Processor 모듈을 로드할 수 없습니다: {e}",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            self.logger.exception(f"Call Chain 수정 중 오류: {e}")
            print(f"오류: {e}", file=sys.stderr)
            return 1
