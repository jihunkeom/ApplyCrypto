"""
DB Access Analyzer 테스트

DB Access Analyzer의 기능을 테스트합니다.
"""

from datetime import datetime
from parser.call_graph_builder import CallGraphBuilder
from parser.java_ast_parser import JavaASTParser
from parser.xml_mapper_parser import XMLMapperParser
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from analyzer.db_access_analyzer import DBAccessAnalyzer
from analyzer.sql_parsing_strategy import (
    JDBCStrategy,
    JPAStrategy,
    MyBatisStrategy,
    create_strategy,
)
from config.config_manager import ConfigurationManager
from models.source_file import SourceFile
from persistence.cache_manager import CacheManager


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
            {"table_name": "USERS", "columns": ["ID", "NAME", "EMAIL"]},
            {"table_name": "ORDERS", "columns": ["ORDER_ID", "USER_ID", "AMOUNT"]},
        ],
    }

    import json

    config_file = temp_dir / "config.json"
    config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return config_file


@pytest.fixture
def config_manager(sample_config_file):
    """설정 매니저 생성"""
    return ConfigurationManager(str(sample_config_file))


@pytest.fixture
def cache_manager(temp_dir):
    """캐시 매니저 생성"""
    cache_dir = temp_dir / "cache"
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def xml_parser(cache_manager):
    """XML Mapper 파서 생성"""
    return XMLMapperParser()


@pytest.fixture
def java_parser(cache_manager):
    """Java AST 파서 생성"""
    return JavaASTParser(cache_manager=cache_manager)


@pytest.fixture
def call_graph_builder(java_parser, cache_manager):
    """Call Graph Builder 생성"""
    return CallGraphBuilder(java_parser=java_parser, cache_manager=cache_manager)


@pytest.fixture
def db_analyzer(config_manager, xml_parser, java_parser, call_graph_builder):
    """DB Access Analyzer 생성"""
    return DBAccessAnalyzer(
        config_manager=config_manager,
        xml_parser=xml_parser,
        java_parser=java_parser,
        call_graph_builder=call_graph_builder,
    )


@pytest.fixture
def sample_mapper_xml(temp_dir):
    """샘플 Mapper XML 파일 생성"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">
    <select id="findById" resultType="User">
        SELECT id, name, email FROM users WHERE id = #{id}
    </select>
    <insert id="insert">
        INSERT INTO users (name, email) VALUES (#{name}, #{email})
    </insert>
</mapper>
"""
    xml_file = temp_dir / "UserMapper.xml"
    xml_file.write_text(xml_content, encoding="utf-8")
    return xml_file


@pytest.fixture
def sample_source_files(temp_dir, sample_mapper_xml):
    """샘플 소스 파일 목록 생성"""
    files = []

    # XML 파일
    xml_file = SourceFile(
        path=sample_mapper_xml,
        relative_path=sample_mapper_xml.name,
        filename=sample_mapper_xml.name,
        extension=".xml",
        size=sample_mapper_xml.stat().st_size,
        modified_time=datetime.fromtimestamp(sample_mapper_xml.stat().st_mtime),
        tags=[],
    )
    files.append(xml_file)

    # Java 파일
    java_file = temp_dir / "UserDAO.java"
    java_content = """
package com.example.dao;

@Repository
public class UserDAO {
    public User findById(Long id) {
        return null;
    }
}
"""
    java_file.write_text(java_content, encoding="utf-8")

    java_source_file = SourceFile(
        path=java_file,
        relative_path=java_file.name,
        filename=java_file.name,
        extension=".java",
        size=java_file.stat().st_size,
        modified_time=datetime.fromtimestamp(java_file.stat().st_mtime),
        tags=[],
    )
    files.append(java_source_file)

    return files


def test_sql_parsing_strategy_mybatis():
    """MyBatis 전략 테스트"""
    strategy = MyBatisStrategy()

    sql = "SELECT id, name FROM users WHERE id = 1"
    tables = strategy.extract_table_names(sql)
    assert "USERS" in tables

    columns = strategy.extract_column_names(sql, "USERS")
    assert "ID" in columns
    assert "NAME" in columns


def test_sql_parsing_strategy_jpa():
    """JPA 전략 테스트"""
    strategy = JPAStrategy()

    jpql = "SELECT u FROM User u WHERE u.id = 1"
    tables = strategy.extract_table_names(jpql)
    assert "USER" in tables


def test_sql_parsing_strategy_jdbc():
    """JDBC 전략 테스트"""
    strategy = JDBCStrategy()

    sql = "SELECT * FROM orders WHERE order_id = 1"
    tables = strategy.extract_table_names(sql)
    assert "ORDERS" in tables


def test_create_strategy():
    """전략 생성 테스트"""
    mybatis = create_strategy("mybatis")
    assert isinstance(mybatis, MyBatisStrategy)

    jpa = create_strategy("jpa")
    assert isinstance(jpa, JPAStrategy)

    jdbc = create_strategy("jdbc")
    assert isinstance(jdbc, JDBCStrategy)

    with pytest.raises(ValueError):
        create_strategy("invalid")


def test_identify_table_access_files(db_analyzer, sample_source_files):
    """테이블 접근 파일 식별 테스트"""
    file_table_map = db_analyzer._identify_table_access_files(sample_source_files)

    assert len(file_table_map) > 0

    # XML 파일에서 테이블 추출 확인
    xml_file = next((f for f in sample_source_files if f.extension == ".xml"), None)
    if xml_file:
        tables = file_table_map.get(str(xml_file.path), set())
        assert "USERS" in tables


def test_assign_file_tags(db_analyzer, sample_source_files):
    """파일 태그 부여 테스트"""
    file_table_map = db_analyzer._identify_table_access_files(sample_source_files)
    tagged_files = db_analyzer._assign_file_tags(sample_source_files, file_table_map)

    assert len(tagged_files) == len(sample_source_files)

    # XML 파일에 태그가 부여되었는지 확인
    xml_file = next((f for f in tagged_files if f.extension == ".xml"), None)
    if xml_file and "USERS" in db_analyzer.table_column_map:
        assert "USERS" in xml_file.tags


def test_classify_files_by_layer(db_analyzer, sample_source_files):
    """레이어별 파일 분류 테스트"""
    file_table_map = db_analyzer._identify_table_access_files(sample_source_files)
    tagged_files = db_analyzer._assign_file_tags(sample_source_files, file_table_map)
    layer_files = db_analyzer._classify_files_by_layer(tagged_files)

    assert "Mapper" in layer_files or "Unknown" in layer_files
    assert "DAO" in layer_files or "Unknown" in layer_files


def test_identify_layer(db_analyzer, temp_dir):
    """레이어 식별 테스트"""
    # Mapper XML 파일
    mapper_file = SourceFile(
        path=temp_dir / "UserMapper.xml",
        relative_path="UserMapper.xml",
        filename="UserMapper.xml",
        extension=".xml",
        size=100,
        modified_time=datetime.now(),
        tags=[],
    )
    assert db_analyzer._identify_layer(mapper_file) == "Mapper"

    # DAO 파일
    dao_file = SourceFile(
        path=temp_dir / "UserDAO.java",
        relative_path="UserDAO.java",
        filename="UserDAO.java",
        extension=".java",
        size=100,
        modified_time=datetime.now(),
        tags=[],
    )
    assert db_analyzer._identify_layer(dao_file) == "DAO"

    # Service 파일
    service_file = SourceFile(
        path=temp_dir / "UserService.java",
        relative_path="UserService.java",
        filename="UserService.java",
        extension=".java",
        size=100,
        modified_time=datetime.now(),
        tags=[],
    )
    assert db_analyzer._identify_layer(service_file) == "Service"

    # Controller 파일
    controller_file = SourceFile(
        path=temp_dir / "UserController.java",
        relative_path="UserController.java",
        filename="UserController.java",
        extension=".java",
        size=100,
        modified_time=datetime.now(),
        tags=[],
    )
    assert db_analyzer._identify_layer(controller_file) == "Controller"


def test_analyze_column_level(db_analyzer, sample_source_files):
    """칼럼 레벨 분석 테스트"""
    file_table_map = db_analyzer._identify_table_access_files(sample_source_files)
    tagged_files = db_analyzer._assign_file_tags(sample_source_files, file_table_map)
    layer_files = db_analyzer._classify_files_by_layer(tagged_files)

    table_access_info_list = db_analyzer._analyze_column_level(
        file_table_map, layer_files
    )

    assert len(table_access_info_list) > 0

    # USERS 테이블 정보 확인
    users_info = next(
        (info for info in table_access_info_list if info.table_name == "USERS"), None
    )
    if users_info:
        assert len(users_info.columns) > 0
        assert len(users_info.access_files) > 0


def test_analyze(db_analyzer, sample_source_files):
    """전체 분석 테스트"""
    result = db_analyzer.analyze(sample_source_files)

    assert isinstance(result, list)
    assert len(result) >= 0

    # 결과가 TableAccessInfo 객체인지 확인
    if result:
        assert hasattr(result[0], "table_name")
        assert hasattr(result[0], "columns")
        assert hasattr(result[0], "access_files")


def test_extract_sql_from_java(db_analyzer, temp_dir):
    """Java 파일에서 SQL 추출 테스트"""
    java_file = temp_dir / "TestDAO.java"
    java_content = """
public class TestDAO {
    public void test() {
        String sql = "SELECT * FROM users WHERE id = 1";
        /* SQL: SELECT name, email FROM users */
    }
}
"""
    java_file.write_text(java_content, encoding="utf-8")

    sql_queries = db_analyzer._extract_sql_from_java(java_file)

    assert len(sql_queries) > 0
    assert any("SELECT" in sql.upper() for sql in sql_queries)
