"""
Source File Collector 단위 테스트

다음 시나리오를 검증합니다:
1. 지정된 확장자 파일만 수집
2. 재귀적 탐색이 모든 하위 디렉터리 포함
3. 중복 파일 제외
4. 메타데이터 정확히 추출
5. 빌드 디렉터리 제외
6. 대규모 파일 시뮬레이션 테스트
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from config.config_manager import ConfigurationManager
from collector.source_file_collector import SourceFileCollector
from models.source_file import SourceFile


@pytest.fixture
def temp_project_dir():
    """임시 프로젝트 디렉터리를 생성하는 픽스처"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        project_path.mkdir()
        
        # 테스트 파일 구조 생성
        (project_path / "src" / "main" / "java").mkdir(parents=True)
        (project_path / "src" / "main" / "resources").mkdir(parents=True)
        (project_path / "target").mkdir()  # 빌드 디렉터리 (제외 대상)
        (project_path / ".git").mkdir()  # 버전 관리 디렉터리 (제외 대상)
        
        # Java 파일 생성
        (project_path / "src" / "main" / "java" / "Main.java").write_text("public class Main {}")
        (project_path / "src" / "main" / "java" / "Service.java").write_text("public class Service {}")
        
        # XML 파일 생성
        (project_path / "src" / "main" / "resources" / "mapper.xml").write_text("<mapper></mapper>")
        
        # 제외할 파일들
        (project_path / "target" / "Main.class").write_text("compiled")  # 빌드 결과물
        (project_path / ".git" / "config").write_text("git config")  # 버전 관리 파일
        (project_path / "README.txt").write_text("readme")  # .txt 파일 (수집 대상 아님)
        
        yield project_path


@pytest.fixture
def config_file(temp_project_dir):
    """임시 설정 파일을 생성하는 픽스처"""
    config_data = {
        "project_path": str(temp_project_dir),
        "source_file_types": [".java", ".xml"],
        "sql_wrapping_type": "mybatis",
        "access_tables": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def config_file_with_excludes(temp_project_dir):
    """exclude 설정이 포함된 임시 설정 파일을 생성하는 픽스처"""
    # 테스트용 파일 생성
    (temp_project_dir / "test").mkdir(exist_ok=True)
    (temp_project_dir / "generated").mkdir(exist_ok=True)
    (temp_project_dir / "test" / "TestService.java").write_text("public class TestService {}")
    (temp_project_dir / "test" / "ServiceTest.java").write_text("public class ServiceTest {}")
    (temp_project_dir / "generated" / "Generated.java").write_text("public class Generated {}")
    (temp_project_dir / "src" / "main" / "java" / "MainTest.java").write_text("public class MainTest {}")
    
    config_data = {
        "project_path": str(temp_project_dir),
        "source_file_types": [".java", ".xml"],
        "sql_wrapping_type": "mybatis",
        "access_tables": [],
        "exclude_dirs": ["test", "generated"],
        "exclude_files": ["*Test.java", "*_test.java"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def config_manager(config_file):
    """ConfigurationManager 인스턴스를 생성하는 픽스처"""
    return ConfigurationManager(config_file)


@pytest.fixture
def collector(config_manager):
    """SourceFileCollector 인스턴스를 생성하는 픽스처"""
    return SourceFileCollector(config_manager)


def test_collect_specified_extensions(collector):
    """지정된 확장자 파일만 수집되는지 확인"""
    files = list(collector.collect())
    
    # .java와 .xml 파일만 수집되어야 함
    extensions = {f.extension for f in files}
    assert ".java" in extensions
    assert ".xml" in extensions
    assert ".txt" not in extensions
    assert ".class" not in extensions


def test_recursive_directory_traversal(collector):
    """재귀적 탐색이 모든 하위 디렉터리를 포함하는지 확인"""
    files = list(collector.collect())
    
    # 하위 디렉터리의 파일도 수집되어야 함
    file_paths = [str(f.relative_path) for f in files]
    
    # src/main/java와 src/main/resources의 파일이 모두 수집되어야 함
    assert any("Main.java" in path for path in file_paths)
    assert any("Service.java" in path for path in file_paths)
    assert any("mapper.xml" in path for path in file_paths)


def test_exclude_build_directories(collector):
    """빌드 디렉터리가 제외되는지 확인"""
    files = list(collector.collect())
    
    # target 디렉터리의 파일은 수집되지 않아야 함
    file_paths = [str(f.path) for f in files]
    assert not any("target" in path for path in file_paths)
    
    # .git 디렉터리의 파일도 수집되지 않아야 함
    assert not any(".git" in path for path in file_paths)


def test_metadata_extraction(collector):
    """메타데이터가 정확히 추출되는지 확인"""
    files = list(collector.collect())
    
    assert len(files) > 0
    
    for file in files:
        # 모든 필수 필드가 존재하는지 확인
        assert file.path is not None
        assert file.relative_path is not None
        assert file.filename is not None
        assert file.extension is not None
        assert file.size >= 0
        assert isinstance(file.modified_time, datetime)
        assert isinstance(file.tags, list)
        
        # 경로가 절대 경로인지 확인
        assert file.path.is_absolute()
        
        # 파일명과 확장자가 일치하는지 확인
        assert file.filename.endswith(file.extension)


def test_duplicate_removal(collector):
    """중복 파일이 제외되는지 확인"""
    # 첫 번째 수집
    files1 = list(collector.collect())
    count1 = len(files1)
    
    # 두 번째 수집 (중복 제거되어야 함)
    collector.reset()
    files2 = list(collector.collect())
    count2 = len(files2)
    
    # 같은 개수여야 함
    assert count1 == count2
    
    # 파일 경로가 고유한지 확인
    paths = [f.path for f in files1]
    assert len(paths) == len(set(paths))


def test_cross_platform_path_compatibility(collector):
    """크로스 플랫폼 경로 호환성 확인"""
    files = list(collector.collect())
    
    for file in files:
        # 경로가 정규화되어 있는지 확인
        assert file.path.is_absolute()
        
        # 상대 경로가 올바른지 확인
        assert not file.relative_path.is_absolute() or str(file.relative_path).startswith("/")


def test_generator_pattern(collector):
    """제너레이터 패턴이 올바르게 작동하는지 확인"""
    # 제너레이터로 수집
    generator = collector.collect()
    
    # 첫 번째 파일 가져오기
    first_file = next(generator)
    assert isinstance(first_file, SourceFile)
    
    # 나머지 파일들도 가져올 수 있는지 확인
    remaining_files = list(generator)
    assert len(remaining_files) >= 0


def test_collect_all_method(collector):
    """collect_all() 메서드가 올바르게 작동하는지 확인"""
    files = collector.collect_all()
    
    assert isinstance(files, list)
    assert all(isinstance(f, SourceFile) for f in files)
    
    # 제너레이터와 같은 결과인지 확인
    collector.reset()
    generator_files = list(collector.collect())
    
    assert len(files) == len(generator_files)


def test_invalid_project_path():
    """존재하지 않는 프로젝트 경로 처리 확인"""
    config_data = {
        "project_path": "/nonexistent/path",
        "source_file_types": [".java"],
        "sql_wrapping_type": "mybatis",
        "access_tables": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_path)
        collector = SourceFileCollector(config_manager)
        
        with pytest.raises(ValueError, match="프로젝트 경로가 존재하지 않습니다"):
            list(collector.collect())
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_get_collected_count(collector):
    """수집된 파일 개수 조회 확인"""
    initial_count = collector.get_collected_count()
    assert initial_count == 0
    
    files = list(collector.collect())
    final_count = collector.get_collected_count()
    
    assert final_count == len(files)
    assert final_count > 0


def test_reset_method(collector):
    """reset() 메서드가 올바르게 작동하는지 확인"""
    # 파일 수집
    list(collector.collect())
    count_before = collector.get_collected_count()
    assert count_before > 0
    
    # 리셋
    collector.reset()
    count_after = collector.get_collected_count()
    assert count_after == 0
    
    # 다시 수집 가능한지 확인
    files = list(collector.collect())
    assert len(files) > 0


def test_case_insensitive_extension_filtering(collector):
    """대소문자 구분 없이 확장자 필터링 확인"""
    # 대문자 확장자 파일 생성
    project_path = collector._project_path
    (project_path / "Test.JAVA").write_text("public class Test {}")
    
    files = list(collector.collect())
    
    # 대문자 .JAVA 파일도 수집되어야 함
    extensions = {f.extension.lower() for f in files}
    assert ".java" in extensions


def test_exclude_dirs_from_config(config_file_with_excludes):
    """config.json의 exclude_dirs 설정이 적용되는지 확인"""
    config_manager = ConfigurationManager(config_file_with_excludes)
    collector = SourceFileCollector(config_manager)
    
    files = list(collector.collect())
    file_paths = [str(f.relative_path) for f in files]
    
    # exclude_dirs에 지정된 디렉터리의 파일은 수집되지 않아야 함
    assert not any("test" in path and "TestService" in path for path in file_paths)
    assert not any("test" in path and "ServiceTest" in path for path in file_paths)
    assert not any("generated" in path for path in file_paths)
    
    # 일반 파일은 여전히 수집되어야 함
    assert any("Main.java" in path for path in file_paths)
    assert any("Service.java" in path for path in file_paths)


def test_exclude_files_from_config(config_file_with_excludes):
    """config.json의 exclude_files 패턴이 적용되는지 확인"""
    config_manager = ConfigurationManager(config_file_with_excludes)
    collector = SourceFileCollector(config_manager)
    
    files = list(collector.collect())
    file_paths = [str(f.relative_path) for f in files]
    file_names = [f.filename for f in files]
    
    # exclude_files 패턴에 매칭되는 파일은 수집되지 않아야 함
    assert not any("MainTest.java" in name for name in file_names)
    assert not any("*Test.java" in name for name in file_names)
    
    # 일반 파일은 여전히 수집되어야 함
    assert any("Main.java" in path for path in file_paths)
    assert any("Service.java" in path for path in file_paths)


def test_exclude_dirs_default_behavior(config_file):
    """기본 exclude_dirs가 여전히 작동하는지 확인"""
    config_manager = ConfigurationManager(config_file)
    collector = SourceFileCollector(config_manager)
    
    files = list(collector.collect())
    file_paths = [str(f.path) for f in files]
    
    # 기본 제외 디렉터리(target, .git)는 여전히 제외되어야 함
    assert not any("target" in path for path in file_paths)
    assert not any(".git" in path for path in file_paths)

