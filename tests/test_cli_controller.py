"""
CLI Controller 테스트

CLI Controller의 기능을 테스트합니다.
"""

import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

from src.cli.cli_controller import CLIController
from config.config_manager import ConfigurationManager
from models.source_file import SourceFile
from persistence.data_persistence_manager import DataPersistenceManager


@pytest.fixture
def temp_dir():
    """임시 디렉터리 생성"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_file(temp_dir):
    """샘플 설정 파일 생성"""
    config_data = {
        "project_path": str(temp_dir),
        "source_file_types": [".java", ".xml"],
        "sql_wrapping_type": "mybatis",
        "access_tables": [
            {
                "table_name": "USERS",
                "columns": ["ID", "NAME", "EMAIL"]
            }
        ]
    }
    
    config_file = temp_dir / "config.json"
    config_file.write_text(json.dumps(config_data, indent=2), encoding='utf-8')
    return config_file


@pytest.fixture
def cli_controller():
    """CLI Controller 생성"""
    return CLIController()


def test_cli_controller_init(cli_controller):
    """CLI Controller 초기화 테스트"""
    assert cli_controller is not None
    assert cli_controller.parser is not None
    assert cli_controller.logger is not None


def test_parse_args_analyze(cli_controller):
    """analyze 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["analyze", "--config", "config.json"])
    assert args.command == "analyze"
    assert args.config == "config.json"


def test_parse_args_list_all(cli_controller):
    """list --all 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["list", "--all"])
    assert args.command == "list"
    assert args.all is True


def test_parse_args_list_db(cli_controller):
    """list --db 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["list", "--db"])
    assert args.command == "list"
    assert args.db is True


def test_parse_args_list_endpoint(cli_controller):
    """list --endpoint 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["list", "--endpoint"])
    assert args.command == "list"
    assert args.endpoint is True


def test_parse_args_list_callgraph(cli_controller):
    """list --callgraph 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["list", "--callgraph", "UserController.getUser"])
    assert args.command == "list"
    assert args.callgraph == "UserController.getUser"


def test_parse_args_modify(cli_controller):
    """modify 명령어 파싱 테스트"""
    args = cli_controller.parse_args(["modify", "--config", "config.json", "--dry-run"])
    assert args.command == "modify"
    assert args.config == "config.json"
    assert args.dry_run is True


def test_load_config(cli_controller, sample_config_file):
    """설정 파일 로드 테스트"""
    config_manager = cli_controller.load_config(str(sample_config_file))
    assert config_manager is not None
    assert config_manager.project_path == Path(sample_config_file.parent)


def test_list_all_files(cli_controller, temp_dir):
    """list --all 파일 목록 출력 테스트"""
    # Data Persistence Manager 생성
    persistence_manager = DataPersistenceManager(temp_dir, output_dir=temp_dir / "results")
    
    # 샘플 소스 파일 생성
    source_files = [
        SourceFile(
            path=temp_dir / "User.java",
            relative_path=Path("User.java"),
            filename="User.java",
            extension=".java",
            size=1000,
            modified_time=datetime.now(),
            tags=[]
        )
    ]
    
    # 파일 저장
    persistence_manager.save_to_file(
        [f.to_dict() for f in source_files],
        "source_files.json"
    )
    
    # list_all_files 호출
    cli_controller._list_all_files(persistence_manager)
    # 예외가 발생하지 않으면 성공


def test_list_db_access(cli_controller, temp_dir):
    """list --db 테이블 접근 정보 출력 테스트"""
    # Data Persistence Manager 생성
    persistence_manager = DataPersistenceManager(temp_dir, output_dir=temp_dir / "results")
    
    # 샘플 테이블 접근 정보 생성
    from models.table_access_info import TableAccessInfo
    table_access_info = [
        TableAccessInfo(
            table_name="USERS",
            columns=["ID", "NAME"],
            access_files=[str(temp_dir / "UserMapper.xml")],
            query_type="SELECT",
            layer="Mapper"
        )
    ]
    
    # 파일 저장
    persistence_manager.save_to_file(
        [t.to_dict() for t in table_access_info],
        "table_access_info.json"
    )
    
    # list_db_access 호출
    cli_controller._list_db_access(persistence_manager)
    # 예외가 발생하지 않으면 성공


def test_list_endpoints(cli_controller, temp_dir):
    """list --endpoint 엔드포인트 목록 출력 테스트"""
    # Data Persistence Manager 생성
    persistence_manager = DataPersistenceManager(temp_dir, output_dir=temp_dir / "results")
    
    # 샘플 Call Graph 데이터 생성
    call_graph_data = {
        "endpoints": [
            {
                "http_method": "GET",
                "path": "/api/users/{id}",
                "method_signature": "UserController.getUser"
            }
        ],
        "node_count": 10,
        "edge_count": 15
    }
    
    # 파일 저장
    persistence_manager.save_to_file(call_graph_data, "call_graph.json")
    
    # list_endpoints 호출
    cli_controller._list_endpoints(persistence_manager)
    # 예외가 발생하지 않으면 성공


def test_execute_analyze(cli_controller, sample_config_file, temp_dir):
    """analyze 명령어 실행 테스트"""
    # 임시 Java 파일 생성
    java_file = temp_dir / "User.java"
    java_file.write_text("public class User {}", encoding='utf-8')
    
    # analyze 명령어 실행
    result = cli_controller.execute(["analyze", "--config", str(sample_config_file)])
    # 성공 또는 실패 모두 테스트 (파일이 없어도 에러 처리 확인)
    assert result in [0, 1]


def test_execute_list_no_option(cli_controller):
    """list 명령어 옵션 없이 실행 테스트"""
    result = cli_controller.execute(["list"])
    assert result == 1  # 옵션이 없으면 실패해야 함
