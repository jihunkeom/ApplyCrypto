"""
Call Tree 출력 예제

엔드포인트부터 시작하는 Call Tree를 터미널에 출력하는 예제입니다.
"""

from parser.call_graph_builder import CallGraphBuilder
from parser.java_ast_parser import JavaASTParser
from persistence.cache_manager import CacheManager
from pathlib import Path

# 캐시 매니저 생성
cache_dir = Path(".cache")
cache_manager = CacheManager(cache_dir=cache_dir)

# Java AST 파서 생성
java_parser = JavaASTParser(cache_manager=cache_manager)

# Call Graph Builder 생성
builder = CallGraphBuilder(java_parser=java_parser, cache_manager=cache_manager)

# 샘플 Java 파일들
controller_code = """
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

service_code = """
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
    
    public void delete(User user) {
        userDAO.delete(user);
    }
}
"""

dao_code = """
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
    
    public void delete(User user) {
        // DB 삭제 로직
    }
}
"""

# 임시 파일 생성
temp_dir = Path("temp_call_tree")
temp_dir.mkdir(exist_ok=True)

controller_file = temp_dir / "UserController.java"
service_file = temp_dir / "UserService.java"
dao_file = temp_dir / "UserDAO.java"

controller_file.write_text(controller_code, encoding='utf-8')
service_file.write_text(service_code, encoding='utf-8')
dao_file.write_text(dao_code, encoding='utf-8')

try:
    print("=" * 60)
    print("Call Tree 출력 예제")
    print("=" * 60)
    
    # Call Graph 생성
    java_files = [controller_file, service_file, dao_file]
    graph = builder.build_call_graph(java_files)
    
    print(f"\n총 {len(graph.nodes())}개의 메서드 노드")
    print(f"총 {len(graph.edges())}개의 호출 관계\n")
    
    # 특정 엔드포인트의 Call Tree 출력
    endpoints = builder.get_endpoints()
    if endpoints:
        getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
        if getUser_endpoint:
            print("\n" + "=" * 60)
            print("특정 엔드포인트의 Call Tree (getUser)")
            print("=" * 60)
            builder.print_call_tree(endpoint=getUser_endpoint, show_layers=True)
        
        createUser_endpoint = next((ep for ep in endpoints if ep.method_name == "createUser"), None)
        if createUser_endpoint:
            print("\n" + "=" * 60)
            print("특정 엔드포인트의 Call Tree (createUser)")
            print("=" * 60)
            builder.print_call_tree(endpoint=createUser_endpoint, show_layers=True)
    
    # 모든 엔드포인트의 Call Tree 출력
    print("\n" + "=" * 60)
    print("모든 엔드포인트의 Call Tree")
    print("=" * 60)
    builder.print_all_call_trees(show_layers=True)

finally:
    # 임시 파일 삭제
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

