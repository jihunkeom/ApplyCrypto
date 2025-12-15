"""
Java AST Parser 테스트

Java AST 파서의 기능을 테스트합니다.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from parser.java_ast_parser import JavaASTParser, ClassInfo
from models.method import Method, Parameter
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
def sample_java_file(temp_dir):
    """샘플 Java 파일 생성"""
    java_code = """
package com.example.demo.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    private String apiVersion = "v1";
    
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
    
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        User user = getUser(id);
        userService.delete(user);
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


def test_parse_file_success(java_parser, sample_java_file):
    """파일 파싱 성공 테스트"""
    tree, error = java_parser.parse_file(sample_java_file)
    
    assert tree is not None
    assert error is None
    assert tree.root_node is not None


def test_parse_file_not_found(java_parser, temp_dir):
    """파일 없음 테스트"""
    non_existent_file = temp_dir / "NonExistent.java"
    tree, error = java_parser.parse_file(non_existent_file)
    
    assert tree is None
    assert error is not None
    assert "파일을 찾을 수 없습니다" in error


def test_extract_class_info(java_parser, sample_java_file):
    """클래스 정보 추출 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    assert len(classes) == 1
    assert classes[0].name == "UserController"
    assert "RestController" in classes[0].annotations
    assert "RequestMapping" in classes[0].annotations


def test_extract_method_info(java_parser, sample_java_file):
    """메서드 정보 추출 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    assert len(classes) > 0
    class_info = classes[0]
    
    # 메서드 개수 확인
    assert len(class_info.methods) >= 5  # getUser, createUser, deleteUser, validateId, validateUser
    
    # getUser 메서드 확인
    getUser_method = next((m for m in class_info.methods if m.name == "getUser"), None)
    assert getUser_method is not None
    assert getUser_method.return_type == "User"
    assert getUser_method.access_modifier == "public"
    assert len(getUser_method.parameters) == 1
    assert getUser_method.parameters[0].name == "id"
    assert getUser_method.parameters[0].type == "Long"


def test_extract_field_info(java_parser, sample_java_file):
    """필드 정보 추출 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    assert len(classes) > 0
    class_info = classes[0]
    
    # 필드 개수 확인
    assert len(class_info.fields) >= 2  # userService, apiVersion
    
    # userService 필드 확인
    user_service_field = next((f for f in class_info.fields if f["name"] == "userService"), None)
    assert user_service_field is not None
    assert user_service_field["type"] == "UserService"
    assert "Autowired" in user_service_field["annotations"]


def test_extract_method_calls(java_parser, sample_java_file):
    """메서드 호출 추출 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    assert len(classes) > 0
    class_info = classes[0]
    
    # getUser 메서드의 호출 확인
    getUser_method = next((m for m in class_info.methods if m.name == "getUser"), None)
    assert getUser_method is not None
    assert len(getUser_method.method_calls) > 0
    assert "validateId" in getUser_method.method_calls or any("validateId" in call for call in getUser_method.method_calls)
    assert any("findById" in call for call in getUser_method.method_calls)


def test_build_call_graph(java_parser, sample_java_file):
    """Call Graph 생성 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    call_graph = java_parser.build_call_graph(classes)
    
    assert len(call_graph) > 0
    # getUser 메서드가 호출하는 메서드 확인
    assert any("getUser" in caller for caller in call_graph.keys())


def test_extract_call_relations(java_parser, sample_java_file):
    """CallRelation 추출 테스트"""
    tree, _ = java_parser.parse_file(sample_java_file)
    classes = java_parser.extract_class_info(tree, sample_java_file)
    
    relations = java_parser.extract_call_relations(classes)
    
    assert len(relations) > 0
    # getUser -> validateId 호출 관계 확인
    assert any("getUser" in rel.caller and "validateId" in rel.callee for rel in relations)


def test_fallback_parse(java_parser, temp_dir):
    """Fallback 파서 테스트"""
    # 간단한 Java 파일 생성
    java_code = """
public class TestClass {
    private String field;
    
    public void method() {
        // test
    }
}
"""
    file_path = temp_dir / "TestClass.java"
    file_path.write_text(java_code, encoding='utf-8')
    
    result = java_parser.fallback_parse(file_path)
    
    assert "error" not in result
    assert "TestClass" in result["classes"]
    assert "method" in result["methods"]
    assert "field" in result["fields"]


def test_cache_functionality(java_parser, sample_java_file, cache_manager):
    """캐시 기능 테스트"""
    # 첫 번째 파싱
    tree1, _ = java_parser.parse_file(sample_java_file)
    assert tree1 is not None
    
    # 두 번째 파싱 (캐시 사용)
    tree2, _ = java_parser.parse_file(sample_java_file)
    assert tree2 is not None
    
    # 캐시에서 조회 확인
    cached = cache_manager.get_cached_result(sample_java_file)
    assert cached is not None


def test_complex_java_code(java_parser, temp_dir):
    """복잡한 Java 코드 파싱 테스트"""
    java_code = """
package com.example;

import java.util.List;
import java.util.Map;

@Controller
public class ComplexController {
    
    @Autowired
    private ServiceA serviceA;
    
    @Autowired
    private ServiceB serviceB;
    
    @GetMapping("/list")
    public List<Item> getItems(@RequestParam String filter) {
        return serviceA.findItems(filter);
    }
    
    @PostMapping("/create")
    public Item createItem(@RequestBody Item item) {
        validate(item);
        return serviceB.save(item);
    }
    
    private void validate(Item item) {
        if (item == null) {
            throw new IllegalArgumentException();
        }
    }
}
"""
    file_path = temp_dir / "ComplexController.java"
    file_path.write_text(java_code, encoding='utf-8')
    
    tree, error = java_parser.parse_file(file_path)
    assert tree is not None
    assert error is None
    
    classes = java_parser.extract_class_info(tree, file_path)
    assert len(classes) == 1
    assert classes[0].name == "ComplexController"
    assert len(classes[0].methods) >= 3

