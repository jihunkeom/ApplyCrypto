"""
Java AST Parser 예제

예제 코드를 실행하여 Java AST 파서의 기능을 확인합니다.
"""

from parser.java_ast_parser import JavaASTParser
from persistence.cache_manager import CacheManager
from pathlib import Path

# 캐시 매니저 생성
cache_dir = Path(".cache")
cache_manager = CacheManager(cache_dir=cache_dir)

# Java AST 파서 생성
parser = JavaASTParser(cache_manager=cache_manager)

# Spring Boot Controller 예제 (더 복잡한 버전)
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

# 임시 파일 생성
temp_file = Path("temp_UserController.java")
temp_file.write_text(java_code, encoding='utf-8')

try:
    # 파싱
    tree, error = parser.parse_file(temp_file)
    
    if error:
        print(f"파싱 오류: {error}")
    else:
        # 클래스 정보 추출
        classes = parser.extract_class_info(tree, temp_file)
        
        # 결과 출력
        parser.print_class_info(classes)
        
        # Call Graph 생성 및 출력
        call_graph = parser.build_call_graph(classes)
        parser.print_call_graph(call_graph)
        
        # CallRelation 추출
        relations = parser.extract_call_relations(classes)
        print(f"\n\n총 {len(relations)}개의 호출 관계가 발견되었습니다.")
        
finally:
    # 임시 파일 삭제
    if temp_file.exists():
        temp_file.unlink()

