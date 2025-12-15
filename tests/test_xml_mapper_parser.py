"""
XML Mapper Parser 테스트

XML Mapper 파서의 기능을 테스트합니다.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from parser.xml_mapper_parser import XMLMapperParser
from models.table_access_info import TableAccessInfo


@pytest.fixture
def temp_dir():
    """임시 디렉터리 생성"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def xml_parser():
    """XML Mapper 파서 생성"""
    return XMLMapperParser()


@pytest.fixture
def sample_mapper_xml(temp_dir):
    """샘플 MyBatis Mapper XML 파일 생성"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
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
    
</mapper>
"""
    file_path = temp_dir / "UserMapper.xml"
    file_path.write_text(xml_content, encoding='utf-8')
    return file_path


@pytest.fixture
def complex_mapper_xml(temp_dir):
    """복잡한 JOIN 쿼리를 포함한 Mapper XML 파일 생성"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.OrderMapper">
    
    <select id="findOrderWithUser" resultType="Order">
        SELECT 
            o.id as order_id,
            o.order_date,
            u.name as user_name,
            u.email as user_email
        FROM orders o
        INNER JOIN users u ON o.user_id = u.id
        WHERE o.id = #{orderId}
    </select>
    
    <select id="findOrdersByStatus" parameterType="String" resultType="Order">
        <!-- 주석 테스트 -->
        SELECT * FROM orders WHERE status = #{status}
    </select>
    
</mapper>
"""
    file_path = temp_dir / "OrderMapper.xml"
    file_path.write_text(xml_content, encoding='utf-8')
    return file_path


@pytest.fixture
def cdata_mapper_xml(temp_dir):
    """CDATA 섹션을 포함한 Mapper XML 파일 생성"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" 
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.example.mapper.ProductMapper">
    
    <select id="findProducts" resultType="Product">
        <![CDATA[
        SELECT 
            p.id,
            p.name,
            p.price,
            c.name as category_name
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.id
        WHERE p.status = 'ACTIVE'
        ]]>
    </select>
    
</mapper>
"""
    file_path = temp_dir / "ProductMapper.xml"
    file_path.write_text(xml_content, encoding='utf-8')
    return file_path


def test_parse_file_success(xml_parser, sample_mapper_xml):
    """파일 파싱 성공 테스트"""
    tree, error = xml_parser.parse_file(sample_mapper_xml)
    
    assert tree is not None
    assert error is None


def test_parse_file_not_found(xml_parser, temp_dir):
    """파일 없음 테스트"""
    non_existent_file = temp_dir / "NonExistent.xml"
    tree, error = xml_parser.parse_file(non_existent_file)
    
    assert tree is None
    assert error is not None
    assert "파일을 찾을 수 없습니다" in error or "No such file" in error or "오류 발생" in error


def test_extract_sql_tags(xml_parser, sample_mapper_xml):
    """SQL 태그 추출 테스트"""
    tree, _ = xml_parser.parse_file(sample_mapper_xml)
    sql_queries = xml_parser.extract_sql_tags(tree)
    
    assert len(sql_queries) == 5  # findById, findAll, insert, update, delete
    
    # findById 쿼리 확인
    find_by_id = next((q for q in sql_queries if q.id == "findById"), None)
    assert find_by_id is not None
    assert find_by_id.query_type == "SELECT"
    assert find_by_id.parameter_type == "Long"
    assert find_by_id.result_type == "User"
    assert "SELECT" in find_by_id.sql.upper()
    assert "FROM" in find_by_id.sql.upper() and "users" in find_by_id.sql.lower()


def test_remove_sql_comments(xml_parser):
    """SQL 주석 제거 테스트"""
    sql_with_comments = """
    SELECT * FROM users
    -- 한 줄 주석
    WHERE id = 1
    /* 여러 줄
       주석 */
    """
    
    clean_sql = xml_parser.remove_sql_comments(sql_with_comments)
    
    assert "--" not in clean_sql
    assert "/*" not in clean_sql
    assert "*/" not in clean_sql
    assert "SELECT" in clean_sql.upper()
    assert "FROM" in clean_sql.upper() and "users" in clean_sql.lower()


def test_extract_table_names(xml_parser):
    """테이블명 추출 테스트"""
    # SELECT 쿼리
    select_sql = "SELECT * FROM users WHERE id = 1"
    tables = xml_parser.extract_table_names(select_sql)
    assert "users" in tables
    
    # JOIN 쿼리
    join_sql = """
    SELECT * FROM orders o
    INNER JOIN users u ON o.user_id = u.id
    """
    tables = xml_parser.extract_table_names(join_sql)
    assert "orders" in tables
    assert "users" in tables
    
    # INSERT 쿼리
    insert_sql = "INSERT INTO products (name, price) VALUES ('Test', 100)"
    tables = xml_parser.extract_table_names(insert_sql)
    assert "products" in tables
    
    # UPDATE 쿼리
    update_sql = "UPDATE users SET name = 'Test' WHERE id = 1"
    tables = xml_parser.extract_table_names(update_sql)
    assert "users" in tables
    
    # DELETE 쿼리
    delete_sql = "DELETE FROM users WHERE id = 1"
    tables = xml_parser.extract_table_names(delete_sql)
    assert "users" in tables


def test_extract_column_names(xml_parser):
    """칼럼명 추출 테스트"""
    # SELECT 쿼리
    select_sql = "SELECT id, name, email FROM users"
    columns = xml_parser.extract_column_names(select_sql)
    assert "id" in columns
    assert "name" in columns
    assert "email" in columns
    
    # INSERT 쿼리
    insert_sql = "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)"
    columns = xml_parser.extract_column_names(insert_sql)
    assert "name" in columns
    assert "email" in columns
    assert "created_at" in columns
    
    # UPDATE 쿼리
    update_sql = "UPDATE users SET name = ?, email = ? WHERE id = ?"
    columns = xml_parser.extract_column_names(update_sql)
    assert "name" in columns
    assert "email" in columns


def test_extract_mybatis_parameters(xml_parser):
    """MyBatis 파라미터 추출 테스트"""
    sql = """
    SELECT * FROM users
    WHERE id = #{id}
    AND name = #{name}
    AND status = ${status}
    """
    
    parameters = xml_parser.extract_mybatis_parameters(sql)
    
    assert "id" in parameters
    assert "name" in parameters
    assert "status" in parameters


def test_create_method_mapping(xml_parser, sample_mapper_xml):
    """메서드 매핑 생성 테스트"""
    tree, _ = xml_parser.parse_file(sample_mapper_xml)
    sql_queries = xml_parser.extract_sql_tags(tree)
    
    find_by_id = next((q for q in sql_queries if q.id == "findById"), None)
    assert find_by_id is not None
    
    mapping = xml_parser.create_method_mapping(find_by_id)
    
    assert mapping.method_signature == "com.example.mapper.UserMapper.findById"
    assert "id" in mapping.parameters


def test_extract_table_access_info(xml_parser, sample_mapper_xml):
    """테이블 접근 정보 추출 테스트"""
    table_access_list = xml_parser.extract_table_access_info(sample_mapper_xml)
    
    assert len(table_access_list) > 0
    
    # users 테이블 접근 정보 확인
    users_access = next((t for t in table_access_list if t.table_name == "users"), None)
    assert users_access is not None
    assert users_access.query_type in ["SELECT", "INSERT", "UPDATE", "DELETE"]
    assert users_access.layer == "Mapper"
    assert str(sample_mapper_xml) in users_access.access_files


def test_complex_join_query(xml_parser, complex_mapper_xml):
    """복잡한 JOIN 쿼리 처리 테스트"""
    table_access_list = xml_parser.extract_table_access_info(complex_mapper_xml)
    
    # orders와 users 테이블 접근 확인
    table_names = [t.table_name for t in table_access_list]
    assert "orders" in table_names
    assert "users" in table_names


def test_cdata_section(xml_parser, cdata_mapper_xml):
    """CDATA 섹션 처리 테스트"""
    tree, error = xml_parser.parse_file(cdata_mapper_xml)
    assert tree is not None
    assert error is None
    
    sql_queries = xml_parser.extract_sql_tags(tree)
    assert len(sql_queries) > 0
    
    find_products = next((q for q in sql_queries if q.id == "findProducts"), None)
    assert find_products is not None
    assert "SELECT" in find_products.sql.upper()
    assert "FROM" in find_products.sql.upper() and "products" in find_products.sql.lower()


def test_parse_mapper_file(xml_parser, sample_mapper_xml):
    """Mapper 파일 완전 파싱 테스트"""
    result = xml_parser.parse_mapper_file(sample_mapper_xml)
    
    assert result["error"] is None
    assert len(result["sql_queries"]) == 5
    assert len(result["method_mappings"]) == 5
    assert len(result["table_access_info"]) > 0


def test_invalid_xml(xml_parser, temp_dir):
    """잘못된 XML 처리 테스트"""
    invalid_xml = temp_dir / "invalid.xml"
    invalid_xml.write_text("<invalid>unclosed tag", encoding='utf-8')
    
    tree, error = xml_parser.parse_file(invalid_xml)
    
    assert tree is None
    assert error is not None
    assert "XML" in error or "구문" in error

