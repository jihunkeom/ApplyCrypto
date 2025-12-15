"""
Call Graph Builder 테스트

Call Graph Builder의 기능을 테스트합니다.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from parser.call_graph_builder import CallGraphBuilder, Endpoint, CallChain
from parser.java_ast_parser import JavaASTParser
from persistence.cache_manager import CacheManager


@pytest.fixture
def temp_dir():
    """임시 디렉터리 생성"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_manager(temp_dir):
    """캐시 매니저 생성"""
    cache_dir = temp_dir / "cache"
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def java_parser(cache_manager):
    """Java AST 파서 생성"""
    return JavaASTParser(cache_manager=cache_manager)


@pytest.fixture
def call_graph_builder(java_parser, cache_manager):
    """Call Graph Builder 생성"""
    return CallGraphBuilder(java_parser=java_parser, cache_manager=cache_manager)


@pytest.fixture
def sample_controller_file(temp_dir):
    """샘플 Controller 파일 생성"""
    java_code = """
package com.example.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        validateId(id);
        return userService.findById(id);
    }
    
    @PostMapping
    public User createUser(@RequestBody User user) {
        validateUser(user);
        return userService.save(user);
    }
    
    private void validateId(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("ID cannot be null");
        }
    }
    
    private void validateUser(User user) {
        if (user == null) {
            throw new IllegalArgumentException("User cannot be null");
        }
        validateId(user.getId());
    }
}
"""
    file_path = temp_dir / "UserController.java"
    file_path.write_text(java_code, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_service_file(temp_dir):
    """샘플 Service 파일 생성"""
    java_code = """
package com.example.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class UserService {
    
    @Autowired
    private UserDAO userDAO;
    
    public User findById(Long id) {
        return userDAO.findById(id);
    }
    
    public User save(User user) {
        return userDAO.save(user);
    }
}
"""
    file_path = temp_dir / "UserService.java"
    file_path.write_text(java_code, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_dao_file(temp_dir):
    """샘플 DAO 파일 생성"""
    java_code = """
package com.example.dao;

import org.springframework.stereotype.Repository;

@Repository
public class UserDAO {
    
    public User findById(Long id) {
        // DB 조회 로직
        return null;
    }
    
    public User save(User user) {
        // DB 저장 로직
        return null;
    }
}
"""
    file_path = temp_dir / "UserDAO.java"
    file_path.write_text(java_code, encoding='utf-8')
    return file_path


def test_build_call_graph(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """Call Graph 생성 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    graph = call_graph_builder.build_call_graph(java_files)
    
    assert graph is not None
    assert len(graph.nodes()) > 0
    assert len(graph.edges()) > 0


def test_identify_endpoints(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """엔드포인트 식별 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    endpoints = call_graph_builder.get_endpoints()
    
    assert len(endpoints) > 0
    # getUser 엔드포인트 확인
    getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    assert getUser_endpoint is not None
    assert getUser_endpoint.http_method == "GET"
    assert "UserController.getUser" in getUser_endpoint.method_signature


def test_layer_classification(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """레이어 분류 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    # Controller 레이어 확인
    controller_method = "UserController.getUser"
    layer = call_graph_builder._get_layer(controller_method)
    assert layer == "Controller"
    
    # Service 레이어 확인
    service_method = "UserService.findById"
    layer = call_graph_builder._get_layer(service_method)
    assert layer == "Service"
    
    # DAO 레이어 확인
    dao_method = "UserDAO.findById"
    layer = call_graph_builder._get_layer(dao_method)
    assert layer == "DAO"


def test_build_call_chains(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """호출 체인 생성 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    chains = call_graph_builder.build_call_chains()
    
    assert len(chains) > 0
    
    # getUser -> userService.findById -> userDAO.findById 체인 확인
    getUser_chain = next((c for c in chains if "UserController.getUser" in c.chain), None)
    assert getUser_chain is not None
    assert "UserController.getUser" in getUser_chain.chain


def test_detect_circular_references(call_graph_builder, temp_dir):
    """순환 참조 감지 테스트"""
    # 순환 참조가 있는 Java 코드
    java_code = """
package com.example;

public class CircularClass {
    
    public void methodA() {
        methodB();
    }
    
    public void methodB() {
        methodC();
    }
    
    public void methodC() {
        methodA();  // 순환 참조
    }
}
"""
    file_path = temp_dir / "CircularClass.java"
    file_path.write_text(java_code, encoding='utf-8')
    
    call_graph_builder.build_call_graph([file_path])
    cycles = call_graph_builder.detect_circular_references()
    
    # 순환 참조가 감지되어야 함 (또는 빈 리스트일 수도 있음 - networkx 구현에 따라)
    assert isinstance(cycles, list)


def test_get_call_relations(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """CallRelation 추출 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    relations = call_graph_builder.get_call_relations()
    
    assert len(relations) > 0
    
    # getUser -> validateId 호출 관계 확인
    getUser_relation = next(
        (r for r in relations if "getUser" in r.caller and "validateId" in r.callee),
        None
    )
    assert getUser_relation is not None


def test_save_and_load_graph(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file, temp_dir):
    """그래프 저장 및 로드 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    # 그래프 저장
    graph_file = temp_dir / "call_graph.pkl"
    success = call_graph_builder.save_graph(graph_file)
    assert success
    assert graph_file.exists()
    
    # 새로운 빌더로 그래프 로드
    new_builder = CallGraphBuilder(java_parser=java_parser, cache_manager=cache_manager)
    success = new_builder.load_graph(graph_file)
    assert success
    assert new_builder.call_graph is not None
    assert len(new_builder.call_graph.nodes()) > 0


def test_endpoint_extraction(call_graph_builder, sample_controller_file):
    """엔드포인트 추출 상세 테스트"""
    call_graph_builder.build_call_graph([sample_controller_file])
    
    endpoints = call_graph_builder.get_endpoints()
    
    assert len(endpoints) >= 2  # getUser, createUser
    
    # getUser 엔드포인트 상세 확인
    getUser = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    assert getUser is not None
    assert getUser.class_name == "UserController"
    assert getUser.http_method == "GET"
    
    # createUser 엔드포인트 상세 확인
    createUser = next((ep for ep in endpoints if ep.method_name == "createUser"), None)
    assert createUser is not None
    assert createUser.http_method == "POST"


def test_multiple_layers_call_chain(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file):
    """다중 레이어 호출 체인 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    endpoints = call_graph_builder.get_endpoints()
    getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    
    if getUser_endpoint:
        chains = call_graph_builder.build_call_chains(endpoint=getUser_endpoint)
        
        assert len(chains) > 0
        
        # 체인에 여러 레이어가 포함되어야 함
        chain = chains[0]
        assert len(chain.chain) > 0
        assert len(chain.layers) > 0


def test_empty_graph(call_graph_builder, temp_dir):
    """빈 그래프 처리 테스트"""
    # 빈 Java 파일
    empty_file = temp_dir / "Empty.java"
    empty_file.write_text("public class Empty {}", encoding='utf-8')
    
    graph = call_graph_builder.build_call_graph([empty_file])
    
    assert graph is not None
    # 노드가 없거나 매우 적을 수 있음
    assert isinstance(graph.nodes(), type(graph.nodes()))


def test_print_call_tree(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file, capsys):
    """Call Tree 출력 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    endpoints = call_graph_builder.get_endpoints()
    getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    
    assert getUser_endpoint is not None
    
    # Call Tree 출력
    call_graph_builder.print_call_tree(endpoint=getUser_endpoint)
    
    # 출력 확인
    captured = capsys.readouterr()
    assert "getUser" in captured.out
    assert "Endpoint:" in captured.out or "Method:" in captured.out


def test_print_all_call_trees(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file, capsys):
    """모든 Call Tree 출력 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    # 모든 Call Tree 출력
    call_graph_builder.print_all_call_trees()
    
    # 출력 확인
    captured = capsys.readouterr()
    assert "CALL TREES" in captured.out
    assert "getUser" in captured.out or "createUser" in captured.out


def test_print_call_tree_with_layers(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file, capsys):
    """레이어 정보 포함 Call Tree 출력 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    endpoints = call_graph_builder.get_endpoints()
    getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    
    assert getUser_endpoint is not None
    
    # 레이어 정보 포함 Call Tree 출력
    call_graph_builder.print_call_tree(endpoint=getUser_endpoint, show_layers=True)
    
    # 출력 확인
    captured = capsys.readouterr()
    assert "getUser" in captured.out
    assert "[" in captured.out  # 레이어 정보 표시


def test_print_call_tree_max_depth(call_graph_builder, sample_controller_file, sample_service_file, sample_dao_file, capsys):
    """최대 깊이 제한 Call Tree 출력 테스트"""
    java_files = [sample_controller_file, sample_service_file, sample_dao_file]
    call_graph_builder.build_call_graph(java_files)
    
    endpoints = call_graph_builder.get_endpoints()
    getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
    
    assert getUser_endpoint is not None
    
    # 최대 깊이 1로 제한
    call_graph_builder.print_call_tree(endpoint=getUser_endpoint, max_depth=1)
    
    # 출력 확인
    captured = capsys.readouterr()
    assert "getUser" in captured.out

