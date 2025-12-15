"""
Configuration Manager 단위 테스트

다음 시나리오를 검증합니다:
1. 유효한 설정 파일 로드 성공
2. 필수 필드 누락 시 예외 발생
3. 잘못된 JSON 형식 처리
4. 각 설정값의 정확한 파싱
5. 기본값 적용
6. 파일 없음 예외 처리
"""

import pytest
import json
import tempfile
from pathlib import Path
from config.config_manager import ConfigurationManager, ConfigurationError


@pytest.fixture
def valid_config_data():
    """유효한 설정 데이터를 반환하는 픽스처"""
    return {
        "project_path": "/path/to/project",
        "source_file_types": [".java", ".xml"],
        "sql_wrapping_type": "mybatis",
        "access_tables": [
            {
                "table_name": "EMPLOYEE",
                "columns": ["NAME", "JUMIN_NUMBER"]
            },
            {
                "table_name": "CUSTOMER",
                "columns": ["PHONE", "EMAIL"]
            }
        ]
    }


@pytest.fixture
def temp_config_file(valid_config_data):
    """임시 설정 파일을 생성하는 픽스처"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_config_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # 테스트 후 파일 삭제
    Path(temp_path).unlink(missing_ok=True)


def test_load_valid_config(temp_config_file, valid_config_data):
    """유효한 설정 파일 로드 성공 테스트"""
    manager = ConfigurationManager(temp_config_file)
    
    assert manager.project_path == Path(valid_config_data["project_path"])
    assert manager.source_file_types == valid_config_data["source_file_types"]
    assert manager.sql_wrapping_type == valid_config_data["sql_wrapping_type"]
    assert len(manager.access_tables) == 2


def test_missing_required_field():
    """필수 필드 누락 시 예외 발생 테스트"""
    # project_path 누락
    config_data = {
        "source_file_types": [".java"],
        "sql_wrapping_type": "mybatis",
        "access_tables": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ConfigurationError, match="스키마 검증 실패"):
            ConfigurationManager(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_json_format():
    """잘못된 JSON 형식 처리 테스트"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name
    
    try:
        with pytest.raises(ConfigurationError, match="JSON 형식이 올바르지 않습니다"):
            ConfigurationManager(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_not_found():
    """파일 없음 예외 처리 테스트"""
    with pytest.raises(ConfigurationError, match="설정 파일을 찾을 수 없습니다"):
        ConfigurationManager("/nonexistent/path/config.json")


def test_property_access(temp_config_file, valid_config_data):
    """각 설정값의 정확한 파싱 테스트"""
    manager = ConfigurationManager(temp_config_file)
    
    # project_path 프로퍼티 테스트
    assert isinstance(manager.project_path, Path)
    assert str(manager.project_path) == valid_config_data["project_path"]
    
    # source_file_types 프로퍼티 테스트
    assert isinstance(manager.source_file_types, list)
    assert manager.source_file_types == [".java", ".xml"]
    
    # sql_wrapping_type 프로퍼티 테스트
    assert manager.sql_wrapping_type == "mybatis"
    
    # access_tables 프로퍼티 테스트
    assert len(manager.access_tables) == 2
    assert manager.access_tables[0]["table_name"] == "EMPLOYEE"
    assert manager.access_tables[0]["columns"] == ["NAME", "JUMIN_NUMBER"]


def test_get_table_names(temp_config_file):
    """테이블명 목록 조회 테스트"""
    manager = ConfigurationManager(temp_config_file)
    
    table_names = manager.get_table_names()
    assert "EMPLOYEE" in table_names
    assert "CUSTOMER" in table_names
    assert len(table_names) == 2


def test_get_columns_for_table(temp_config_file):
    """특정 테이블의 칼럼 목록 조회 테스트"""
    manager = ConfigurationManager(temp_config_file)
    
    # 존재하는 테이블
    columns = manager.get_columns_for_table("EMPLOYEE")
    assert "NAME" in columns
    assert "JUMIN_NUMBER" in columns
    
    # 존재하지 않는 테이블
    columns = manager.get_columns_for_table("NONEXISTENT")
    assert columns == []


def test_invalid_sql_wrapping_type():
    """잘못된 sql_wrapping_type 값 처리 테스트"""
    config_data = {
        "project_path": "/path/to/project",
        "source_file_types": [".java"],
        "sql_wrapping_type": "invalid_type",
        "access_tables": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ConfigurationError, match="스키마 검증 실패"):
            ConfigurationManager(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_access_tables_structure():
    """잘못된 access_tables 구조 처리 테스트"""
    config_data = {
        "project_path": "/path/to/project",
        "source_file_types": [".java"],
        "sql_wrapping_type": "mybatis",
        "access_tables": [
            {
                "table_name": "EMPLOYEE"
                # columns 필드 누락
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ConfigurationError, match="스키마 검증 실패"):
            ConfigurationManager(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

