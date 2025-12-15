"""
DB Access Analyzer 예제

예제 코드를 실행하여 DB Access Analyzer의 기능을 확인합니다.
"""


from analyzer.db_access_analyzer import DBAccessAnalyzer
from config.config_manager import ConfigurationManager
from models.source_file import SourceFile
from parser.xml_mapper_parser import XMLMapperParser
from parser.java_ast_parser import JavaASTParser
from parser.call_graph_builder import CallGraphBuilder
from persistence.cache_manager import CacheManager
from pathlib import Path
from datetime import datetime
import json

# 임시 디렉터리 생성
temp_dir = Path("temp_db_analyzer")
temp_dir.mkdir(exist_ok=True)

# 설정 파일 생성
config_data = {
    "project_path": str(temp_dir),
    "source_file_types": [".java", ".xml"],
    "sql_wrapping_type": "mybatis",
    "access_tables": [
        {
            "table_name": "USERS",
            "columns": ["ID", "NAME", "EMAIL", "CREATED_AT"]
        },
        {
            "table_name": "ORDERS",
            "columns": ["ORDER_ID", "USER_ID", "AMOUNT", "ORDER_DATE"]
        }
    ]
}

config_file = temp_dir / "config.json"
config_file.write_text(json.dumps(config_data, indent=2), encoding='utf-8')

# 샘플 파일 생성
mapper_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">
    <select id="findById" resultType="User">
        SELECT id, name, email, created_at
        FROM users
        WHERE id = #{id}
    </select>
    <insert id="insert">
        INSERT INTO users (name, email, created_at)
        VALUES (#{name}, #{email}, #{createdAt})
    </insert>
    <update id="update">
        UPDATE users
        SET name = #{name}, email = #{email}
        WHERE id = #{id}
    </update>
</mapper>
"""

order_mapper_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.OrderMapper">
    <select id="findByUserId" resultType="Order">
        SELECT order_id, user_id, amount, order_date
        FROM orders
        WHERE user_id = #{userId}
    </select>
    <insert id="insert">
        INSERT INTO orders (user_id, amount, order_date)
        VALUES (#{userId}, #{amount}, #{orderDate})
    </insert>
</mapper>
"""

dao_java_content = """
package com.example.dao;

import org.springframework.stereotype.Repository;

@Repository
public class UserDAO {
    public User findById(Long id) {
        return null;
    }
}
"""

service_java_content = """
package com.example.service;

import org.springframework.stereotype.Service;

@Service
public class UserService {
    public User getUser(Long id) {
        return null;
    }
}
"""

# 파일 생성
mapper_xml = temp_dir / "UserMapper.xml"
order_mapper_xml = temp_dir / "OrderMapper.xml"
dao_java = temp_dir / "UserDAO.java"
service_java = temp_dir / "UserService.java"

mapper_xml.write_text(mapper_xml_content, encoding='utf-8')
order_mapper_xml.write_text(order_mapper_xml_content, encoding='utf-8')
dao_java.write_text(dao_java_content, encoding='utf-8')
service_java.write_text(service_java_content, encoding='utf-8')

try:
    print("=" * 60)
    print("DB Access Analyzer 예제")
    print("=" * 60)
    
    # 설정 매니저 생성
    config_manager = ConfigurationManager(str(config_file))
    
    # 캐시 매니저 생성
    cache_manager = CacheManager(cache_dir=temp_dir / "cache")
    
    # 파서들 생성
    xml_parser = XMLMapperParser()
    java_parser = JavaASTParser(cache_manager=cache_manager)
    call_graph_builder = CallGraphBuilder(java_parser=java_parser, cache_manager=cache_manager)
    
    # DB Access Analyzer 생성
    analyzer = DBAccessAnalyzer(
        config_manager=config_manager,
        xml_parser=xml_parser,
        java_parser=java_parser,
        call_graph_builder=call_graph_builder
    )
    
    # 소스 파일 목록 생성
    source_files = []
    for file_path in [mapper_xml, order_mapper_xml, dao_java, service_java]:
        source_file = SourceFile(
            path=file_path,
            relative_path=file_path.name,
            filename=file_path.name,
            extension=file_path.suffix,
            size=file_path.stat().st_size,
            modified_time=datetime.fromtimestamp(file_path.stat().st_mtime),
            tags=[]
        )
        source_files.append(source_file)
    
    # 분석 수행
    print("\n분석 중...\n")
    table_access_info_list = analyzer.analyze(source_files)
    
    # 결과 출력
    print("=" * 60)
    print("테이블 접근 정보")
    print("=" * 60)
    
    for i, info in enumerate(table_access_info_list, 1):
        print(f"\n[{i}] 테이블: {info.table_name}")
        print(f"    레이어: {info.layer}")
        print(f"    쿼리 타입: {info.query_type}")
        print(f"    칼럼 수: {len(info.columns)}")
        if info.columns:
            print(f"    칼럼: {', '.join(info.columns[:5])}" + (f" ... (총 {len(info.columns)}개)" if len(info.columns) > 5 else ""))
        print(f"    접근 파일 수: {len(info.access_files)}")
        for j, file_path in enumerate(info.access_files[:3], 1):
            print(f"      {j}. {Path(file_path).name}")
        if len(info.access_files) > 3:
            print(f"      ... 외 {len(info.access_files) - 3}개")
    
    # 파일 태그 정보
    print("\n" + "=" * 60)
    print("파일 태그 정보")
    print("=" * 60)
    
    file_table_map = analyzer._identify_table_access_files(source_files)
    tagged_files = analyzer._assign_file_tags(source_files, file_table_map)
    
    for source_file in tagged_files:
        if source_file.tags:
            print(f"\n파일: {source_file.filename}")
            print(f"  태그: {', '.join(source_file.tags)}")
    
    # 레이어별 파일 분류
    print("\n" + "=" * 60)
    print("레이어별 파일 분류")
    print("=" * 60)
    
    layer_files = analyzer._classify_files_by_layer(tagged_files)
    for layer, files in sorted(layer_files.items()):
        print(f"\n[{layer}] ({len(files)}개)")
        for file in files:
            print(f"  - {file.filename}")

finally:
    # 임시 파일 삭제
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

