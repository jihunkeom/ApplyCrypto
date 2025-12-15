"""
Call Graph Builder 예제

예제 코드를 실행하여 Call Graph Builder의 기능을 확인합니다.
"""


from parser.call_graph_builder import CallGraphBuilder
from parser.java_ast_parser import JavaASTParser
from persistence.cache_manager import CacheManager
from pathlib import Path
from collections import defaultdict

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
}
"""

# 임시 파일 생성
temp_dir = Path("temp_example")
temp_dir.mkdir(exist_ok=True)

controller_file = temp_dir / "UserController.java"
service_file = temp_dir / "UserService.java"
dao_file = temp_dir / "UserDAO.java"

controller_file.write_text(controller_code, encoding='utf-8')
service_file.write_text(service_code, encoding='utf-8')
dao_file.write_text(dao_code, encoding='utf-8')

try:
    print("=" * 60)
    print("Call Graph Builder 예제")
    print("=" * 60)
    
    # Call Graph 생성
    java_files = [controller_file, service_file, dao_file]
    graph = builder.build_call_graph(java_files)
    
    print(f"\n총 {len(graph.nodes())}개의 메서드 노드")
    print(f"총 {len(graph.edges())}개의 호출 관계\n")
    
    # 엔드포인트 정보 출력
    print("=" * 60)
    print("REST API 엔드포인트")
    print("=" * 60)
    endpoints = builder.get_endpoints()
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n[{i}] {endpoint.http_method} {endpoint.path}")
        print(f"    메서드: {endpoint.method_signature}")
        print(f"    클래스: {endpoint.class_name}")
        print(f"    파일: {endpoint.file_path}")
    
    # 호출 체인 생성
    print("\n" + "=" * 60)
    print("호출 체인 (Controller -> Service -> DAO)")
    print("=" * 60)
    
    if endpoints:
        getUser_endpoint = next((ep for ep in endpoints if ep.method_name == "getUser"), None)
        if getUser_endpoint:
            chains = builder.build_call_chains(endpoint=getUser_endpoint)
            
            for i, chain in enumerate(chains[:3], 1):  # 최대 3개만 출력
                print(f"\n[체인 {i}]")
                print(f"  순환 참조: {'예' if chain.is_circular else '아니오'}")
                for j, method in enumerate(chain.chain):
                    layer = chain.layers[j] if j < len(chain.layers) else "Unknown"
                    print(f"  {j+1}. [{layer}] {method}")
    
    # 레이어별 메서드 분류
    print("\n" + "=" * 60)
    print("레이어별 메서드 분류")
    print("=" * 60)
    
    layers = defaultdict(list)
    for node in graph.nodes():
        layer = graph.nodes[node].get("layer", "Unknown")
        layers[layer].append(node)
    
    for layer, methods in sorted(layers.items()):
        print(f"\n[{layer}] ({len(methods)}개)")
        for method in methods[:5]:  # 최대 5개만 출력
            print(f"  - {method}")
        if len(methods) > 5:
            print(f"  ... 외 {len(methods) - 5}개")
    
    # 순환 참조 감지
    print("\n" + "=" * 60)
    print("순환 참조 감지")
    print("=" * 60)
    cycles = builder.detect_circular_references()
    if cycles:
        print(f"\n{len(cycles)}개의 순환 참조가 발견되었습니다:")
        for i, cycle in enumerate(cycles[:3], 1):  # 최대 3개만 출력
            print(f"\n[순환 참조 {i}]")
            print(" -> ".join(cycle))
    else:
        print("\n순환 참조가 없습니다.")
    
    # CallRelation 정보
    print("\n" + "=" * 60)
    print("호출 관계 요약")
    print("=" * 60)
    relations = builder.get_call_relations()
    print(f"\n총 {len(relations)}개의 호출 관계")
    
    # 상위 5개 호출 관계 출력
    print("\n[상위 5개 호출 관계]")
    for i, relation in enumerate(relations[:5], 1):
        print(f"{i}. {relation.caller} -> {relation.callee}")

finally:
    # 임시 파일 삭제
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

