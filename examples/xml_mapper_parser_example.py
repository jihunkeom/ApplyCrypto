"""
XML Mapper Parser 예제

예제 코드를 실행하여 XML Mapper 파서의 기능을 확인합니다.
"""

from parser.xml_mapper_parser import XMLMapperParser
from pathlib import Path

# XML Mapper 파서 생성
parser = XMLMapperParser()

# 샘플 MyBatis Mapper XML
mapper_xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.UserMapper">
    
    <select id="findById" parameterType="Long" resultType="User">
        SELECT id, name, email, created_at
        FROM users
        WHERE id = #{id}
    </select>
    
    <select id="findAll" resultType="User">
        SELECT * FROM users
    </select>
    
    <insert id="insert" parameterType="User">
        INSERT INTO users (name, email, created_at)
        VALUES (#{name}, #{email}, #{createdAt})
    </insert>
    
    <update id="update" parameterType="User">
        UPDATE users
        SET name = #{name}, email = #{email}
        WHERE id = #{id}
    </update>
    
    <delete id="delete" parameterType="Long">
        DELETE FROM users WHERE id = #{id}
    </delete>
    
    <select id="findOrdersWithUser" resultType="Order">
        SELECT 
            o.id as order_id,
            o.order_date,
            u.name as user_name,
            u.email as user_email
        FROM orders o
        INNER JOIN users u ON o.user_id = u.id
        WHERE o.id = #{orderId}
    </select>
    
</mapper>
"""

# 임시 파일 생성
temp_file = Path("temp_UserMapper.xml")
temp_file.write_text(mapper_xml_content, encoding='utf-8')

try:
    print("=" * 60)
    print("XML Mapper Parser 예제")
    print("=" * 60)
    
    # Mapper 파일 파싱
    result = parser.parse_mapper_file(temp_file)
    
    if result["error"]:
        print(f"오류: {result['error']}")
    else:
        print(f"\n파일: {result['file_path']}")
        print(f"\n총 {len(result['sql_queries'])}개의 SQL 쿼리를 발견했습니다.\n")
        
        # SQL 쿼리 정보 출력
        for i, sql_query in enumerate(result["sql_queries"], 1):
            print(f"[{i}] {sql_query['query_type']} - {sql_query['id']}")
            print(f"    Namespace: {sql_query['namespace']}")
            if sql_query['parameter_type']:
                print(f"    Parameter Type: {sql_query['parameter_type']}")
            if sql_query['result_type']:
                print(f"    Result Type: {sql_query['result_type']}")
            print(f"    SQL: {sql_query['sql'][:100]}...")
            print()
        
        # 메서드 매핑 정보 출력
        print("\n" + "=" * 60)
        print("메서드 매핑 정보")
        print("=" * 60)
        for mapping in result["method_mappings"]:
            print(f"\n메서드: {mapping['method_signature']}")
            print(f"  쿼리 타입: {mapping['query_type']}")
            if mapping['parameters']:
                print(f"  파라미터: {', '.join(mapping['parameters'])}")
        
        # 테이블 접근 정보 출력
        print("\n" + "=" * 60)
        print("테이블 접근 정보")
        print("=" * 60)
        
        # 테이블별로 그룹화
        table_info = {}
        for access_info in result["table_access_info"]:
            table_name = access_info["table_name"]
            if table_name not in table_info:
                table_info[table_name] = {
                    "query_types": set(),
                    "columns": set(),
                    "files": set()
                }
            table_info[table_name]["query_types"].add(access_info["query_type"])
            table_info[table_name]["columns"].update(access_info["columns"])
            table_info[table_name]["files"].update(access_info.get("access_files", []))
        
        for table_name, info in table_info.items():
            print(f"\n테이블: {table_name}")
            print(f"  쿼리 타입: {', '.join(sorted(info['query_types']))}")
            if info["columns"]:
                print(f"  칼럼: {', '.join(sorted(info['columns']))}")
            print(f"  레이어: Mapper")
        
        # 테이블 접근 정보 객체로 추출
        print("\n" + "=" * 60)
        print("TableAccessInfo 객체 추출")
        print("=" * 60)
        table_access_list = parser.extract_table_access_info(temp_file)
        for access_info in table_access_list:
            print(f"\n테이블: {access_info.table_name}")
            print(f"  쿼리 타입: {access_info.query_type}")
            print(f"  칼럼 수: {len(access_info.columns)}")
            print(f"  레이어: {access_info.layer}")
            print(f"  파일: {access_info.access_files[0] if access_info.access_files else 'N/A'}")

finally:
    # 임시 파일 삭제
    if temp_file.exists():
        temp_file.unlink()

