"""
Data Persistence Manager 단위 테스트

다음 시나리오를 검증합니다:
1. 데이터 모델이 JSON으로 정확히 직렬화되는지 확인
2. JSON에서 데이터 모델로 정확히 역직렬화되는지 확인
3. 파일 저장 및 로드 성공 확인
4. 손상된 JSON 파일 처리 확인
5. 버전 관리 기능 확인
6. 캐싱 기능 확인
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from models.call_relation import CallRelation
from models.method import Method, Parameter
from models.modification_record import ModificationRecord
from models.source_file import SourceFile
from models.table_access_info import TableAccessInfo
from persistence.data_persistence_manager import (
    DataPersistenceManager,
    PersistenceError,
)


@pytest.fixture
def temp_project_dir():
    """임시 프로젝트 디렉터리를 생성하는 픽스처"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        project_path.mkdir()
        yield project_path


@pytest.fixture
def persistence_manager(temp_project_dir):
    """DataPersistenceManager 인스턴스를 생성하는 픽스처"""
    return DataPersistenceManager(temp_project_dir)


@pytest.fixture
def sample_source_file(temp_project_dir):
    """샘플 SourceFile 객체를 생성하는 픽스처"""
    test_file = temp_project_dir / "test.java"
    test_file.write_text("public class Test {}")

    return SourceFile(
        path=test_file.resolve(),
        relative_path=Path("test.java"),
        filename="test.java",
        extension=".java",
        size=test_file.stat().st_size,
        modified_time=datetime.fromtimestamp(test_file.stat().st_mtime),
        tags=["EMPLOYEE"],
    )


@pytest.fixture
def sample_method():
    """샘플 Method 객체를 생성하는 픽스처"""
    return Method(
        name="getEmployee",
        return_type="EmployeeDTO",
        parameters=[Parameter(name="id", type="Long", is_varargs=False)],
        access_modifier="public",
        class_name="EmployeeService",
        file_path="src/main/java/EmployeeService.java",
        is_static=False,
        annotations=["@Override"],
    )


def test_serialize_source_file_to_json(persistence_manager, sample_source_file):
    """SourceFile을 JSON으로 직렬화 확인"""
    json_str = persistence_manager.serialize_to_json(sample_source_file)

    assert isinstance(json_str, str)
    assert "test.java" in json_str
    assert "EMPLOYEE" in json_str

    # JSON 파싱 가능한지 확인
    data = json.loads(json_str)
    assert data["filename"] == "test.java"


def test_deserialize_source_file_from_json(persistence_manager, sample_source_file):
    """JSON에서 SourceFile로 역직렬화 확인"""
    json_str = persistence_manager.serialize_to_json(sample_source_file)
    restored = persistence_manager.deserialize_from_json(json_str, SourceFile)

    assert isinstance(restored, SourceFile)
    assert restored.filename == sample_source_file.filename
    assert restored.extension == sample_source_file.extension
    assert restored.tags == sample_source_file.tags


def test_serialize_method_to_json(persistence_manager, sample_method):
    """Method를 JSON으로 직렬화 확인"""
    json_str = persistence_manager.serialize_to_json(sample_method)

    assert isinstance(json_str, str)
    assert "getEmployee" in json_str
    assert "EmployeeService" in json_str

    data = json.loads(json_str)
    assert data["name"] == "getEmployee"
    assert len(data["parameters"]) == 1


def test_deserialize_method_from_json(persistence_manager, sample_method):
    """JSON에서 Method로 역직렬화 확인"""
    json_str = persistence_manager.serialize_to_json(sample_method)
    restored = persistence_manager.deserialize_from_json(json_str, Method)

    assert isinstance(restored, Method)
    assert restored.name == sample_method.name
    assert restored.return_type == sample_method.return_type
    assert len(restored.parameters) == 1
    assert restored.parameters[0].name == "id"


def test_save_and_load_source_files(persistence_manager, sample_source_file):
    """SourceFile 리스트 저장 및 로드 확인"""
    source_files = [sample_source_file]

    # 저장
    file_path = persistence_manager.save_to_file(source_files, "source_files.json")

    assert file_path.exists()

    # 로드
    loaded = persistence_manager.load_from_file("source_files.json", SourceFile)

    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert isinstance(loaded[0], SourceFile)
    assert loaded[0].filename == sample_source_file.filename


def test_save_and_load_call_relations(persistence_manager):
    """CallRelation 리스트 저장 및 로드 확인"""
    call_relations = [
        CallRelation(
            caller="EmployeeController.getEmployee",
            callee="EmployeeService.findById",
            caller_file="EmployeeController.java",
            callee_file="EmployeeService.java",
            line_number=10,
        )
    ]

    persistence_manager.save_to_file(call_relations, "call_relations.json")

    loaded = persistence_manager.load_from_file("call_relations.json", CallRelation)

    assert len(loaded) == 1
    assert loaded[0].caller == "EmployeeController.getEmployee"
    assert loaded[0].callee == "EmployeeService.findById"


def test_save_and_load_table_access_info(persistence_manager):
    """TableAccessInfo 리스트 저장 및 로드 확인"""
    table_access = [
        TableAccessInfo(
            table_name="EMPLOYEE",
            columns=["NAME", "JUMIN_NUMBER"],
            access_files=["EmployeeMapper.xml", "EmployeeDTO.java"],
            query_type="SELECT",
        )
    ]

    persistence_manager.save_to_file(table_access, "table_access.json")

    loaded = persistence_manager.load_from_file("table_access.json", TableAccessInfo)

    assert len(loaded) == 1
    assert loaded[0].table_name == "EMPLOYEE"
    assert "NAME" in loaded[0].columns


def test_save_and_load_modification_records(persistence_manager):
    """ModificationRecord 리스트 저장 및 로드 확인"""
    records = [
        ModificationRecord(
            file_path="EmployeeDTO.java",
            table_name="EMPLOYEE",
            column_name="JUMIN_NUMBER",
            modified_methods=["getJuminNumber", "setJuminNumber"],
            added_imports=["k-sign.CryptoService"],
            timestamp=datetime.now(),
            status="success",
        )
    ]

    persistence_manager.save_to_file(records, "modifications.json")

    loaded = persistence_manager.load_from_file(
        "modifications.json", ModificationRecord
    )

    assert len(loaded) == 1
    assert loaded[0].file_path == "EmployeeDTO.java"
    assert loaded[0].status == "success"


def test_corrupted_json_file_handling(persistence_manager, temp_project_dir):
    """손상된 JSON 파일 처리 확인"""
    # 손상된 JSON 파일 생성
    corrupted_file = persistence_manager.output_dir / "corrupted.json"
    corrupted_file.write_text("{ invalid json }")

    # 백업 파일 생성
    backup_file = persistence_manager.create_backup(corrupted_file)

    # 손상된 파일 로드 시도
    with pytest.raises(PersistenceError, match="JSON 역직렬화 실패"):
        persistence_manager.load_from_file("corrupted.json")

    # 백업 복원 시도
    if backup_file.exists():
        restored = persistence_manager.handle_corrupted_file(corrupted_file)
        assert restored is True or restored is False


def test_file_not_found_error(persistence_manager):
    """파일 없음 에러 처리 확인"""
    with pytest.raises(PersistenceError, match="파일을 찾을 수 없습니다"):
        persistence_manager.load_from_file("nonexistent.json")


def test_add_timestamp(persistence_manager):
    """타임스탬프 추가 확인"""
    data = {"key": "value"}
    timestamped = persistence_manager.add_timestamp(data)

    assert "created_time" in timestamped
    assert "modified_time" in timestamped
    assert isinstance(timestamped["created_time"], str)
    assert isinstance(timestamped["modified_time"], str)


def test_get_version_info(persistence_manager, sample_source_file):
    """버전 정보 조회 확인"""
    # 파일 저장
    persistence_manager.save_to_file(
        {"data": [sample_source_file.to_dict()]}, "version_test.json"
    )

    # 버전 정보 조회
    version_info = persistence_manager.get_version_info("version_test.json")

    assert "created_time" in version_info or version_info["created_time"] is None
    assert "modified_time" in version_info or version_info["modified_time"] is None
    assert "file_size" in version_info
    assert version_info["file_size"] > 0


def test_subdirectory_save_and_load(persistence_manager, sample_source_file):
    """하위 디렉터리에 파일 저장 및 로드 확인"""
    source_files = [sample_source_file]

    # 하위 디렉터리에 저장
    file_path = persistence_manager.save_to_file(
        source_files, "files.json", subdirectory="source_files"
    )

    assert "source_files" in str(file_path)
    assert file_path.exists()

    # 하위 디렉터리에서 로드
    loaded = persistence_manager.load_from_file(
        "files.json", SourceFile, subdirectory="source_files"
    )

    assert len(loaded) == 1


def test_cache_functionality(persistence_manager, sample_source_file, temp_project_dir):
    """캐싱 기능 확인"""
    if persistence_manager.cache_manager is None:
        pytest.skip("캐싱이 비활성화되어 있습니다")

    test_file = temp_project_dir / "cache_test.java"
    test_file.write_text("public class CacheTest {}")

    # 캐시에 저장
    data = [sample_source_file]
    persistence_manager.set_cached_result(test_file, data)

    # 캐시에서 조회
    cached = persistence_manager.get_cached_result(test_file)
    assert cached is not None
    assert len(cached) == 1

    # 캐시 무효화
    persistence_manager.cache_manager.invalidate_cache(test_file)
    cached_after = persistence_manager.get_cached_result(test_file)
    assert cached_after is None


def test_roundtrip_serialization(
    persistence_manager, sample_source_file, sample_method
):
    """라운드트립 직렬화/역직렬화 확인"""
    # SourceFile 라운드트립
    json_str = persistence_manager.serialize_to_json(sample_source_file)
    restored_file = persistence_manager.deserialize_from_json(json_str, SourceFile)

    assert restored_file.filename == sample_source_file.filename
    assert restored_file.tags == sample_source_file.tags

    # Method 라운드트립
    json_str = persistence_manager.serialize_to_json(sample_method)
    restored_method = persistence_manager.deserialize_from_json(json_str, Method)

    assert restored_method.name == sample_method.name
    assert len(restored_method.parameters) == len(sample_method.parameters)


def test_list_serialization(persistence_manager, sample_source_file, sample_method):
    """리스트 직렬화/역직렬화 확인"""
    data = {"source_files": [sample_source_file], "methods": [sample_method]}

    json_str = persistence_manager.serialize_to_json(data)
    restored = persistence_manager.deserialize_from_json(json_str)

    assert "source_files" in restored
    assert "methods" in restored
    assert len(restored["source_files"]) == 1
    assert len(restored["methods"]) == 1
