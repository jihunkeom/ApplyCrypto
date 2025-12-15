"""
Java AST Parser

tree-sitter를 사용하여 Java 소스 코드를 추상 구문 트리(AST)로 파싱하고,
클래스, 메서드, 변수 정보를 추출하는 모듈입니다.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from tree_sitter import Parser, Language, Node, Tree
import tree_sitter_java as tsjava

from models.method import Method, Parameter, LocalVariable
from models.call_relation import CallRelation
from persistence.cache_manager import CacheManager


# Java 언어 설정
JAVA_LANGUAGE = Language(tsjava.language())


@dataclass
class ClassInfo:
    """
    클래스 정보를 저장하는 데이터 모델
    
    Attributes:
        name: 클래스명
        package: 패키지명
        superclass: 부모 클래스명
        interfaces: 구현 인터페이스 목록
        annotations: 어노테이션 목록
        fields: 필드 목록
        methods: 메서드 목록
        file_path: 파일 경로
    """
    name: str
    package: str = ""
    superclass: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    fields: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[Method] = field(default_factory=list)
    file_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        ClassInfo를 딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 딕셔너리 형태의 클래스 정보
        """
        return {
            "name": self.name,
            "package": self.package,
            "superclass": self.superclass,
            "interfaces": self.interfaces,
            "annotations": self.annotations,
            "fields": self.fields,  # 이미 Dict 형태
            "methods": [method.to_dict() if hasattr(method, 'to_dict') else method for method in self.methods],
            "file_path": self.file_path
        }


@dataclass
class FieldInfo:
    """
    필드 정보를 저장하는 데이터 모델
    
    Attributes:
        name: 필드명
        type: 필드 타입
        annotations: 어노테이션 목록
        initial_value: 초기값 (선택적)
        access_modifier: 접근 제어자
        is_static: 정적 필드 여부
        is_final: final 필드 여부
    """
    name: str
    type: str
    annotations: List[str] = field(default_factory=list)
    initial_value: Optional[str] = None
    access_modifier: str = "package"
    is_static: bool = False
    is_final: bool = False


class JavaASTParser:
    """
    Java AST 파서 클래스
    
    tree-sitter를 사용하여 Java 소스 코드를 파싱하고,
    클래스, 메서드, 필드 정보를 추출합니다.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        JavaASTParser 초기화
        
        Args:
            cache_manager: 캐시 매니저 (선택적)
        """
        import logging
        self.parser = Parser(JAVA_LANGUAGE)
        self.logger = logging.getLogger("applycrypto")
        # cache_manager가 없으면 임시 디렉터리에 생성
        if cache_manager is None:
            from tempfile import mkdtemp
            temp_cache_dir = Path(mkdtemp())
            self.cache_manager = CacheManager(temp_cache_dir)
        else:
            self.cache_manager = cache_manager
    
    def parse_file(self, file_path: Path) -> Tuple[Optional[Tree], Optional[str]]:
        """
        Java 파일을 파싱하여 AST로 변환
        
        Args:
            file_path: Java 파일 경로 (Path 객체 또는 문자열)
            
        Returns:
            Tuple[Optional[Tree], Optional[str]]: (AST 트리, 에러 메시지)
        """
        try:
            # Path 객체로 변환 (SourceFile 객체가 전달될 수 있으므로 path 속성 확인)
            if hasattr(file_path, 'path'):
                # SourceFile 객체인 경우 path 속성 사용
                file_path = Path(file_path.path)
            else:
                file_path = Path(file_path)
            
            # 캐시 확인
            cached_ast = self.cache_manager.get_cached_result(file_path)
            if cached_ast:
                return cached_ast, None
            
            # 파일 읽기 (여러 인코딩 시도)
            source_code = None
            encodings = ['utf-8', 'euc-kr', 'cp949', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        source_code = f.read()
                    break  # 성공하면 루프 종료
                except UnicodeDecodeError:
                    continue  # 다음 인코딩 시도
                except Exception as e:
                    # 다른 에러는 로깅하고 계속 시도
                    self.logger.debug(f"인코딩 {encoding} 시도 중 에러: {e}")
                    continue
            
            if source_code is None:
                return None, f"파일을 읽을 수 없습니다: 지원되는 인코딩을 찾을 수 없습니다 (시도한 인코딩: {', '.join(encodings)})"
            
            # 파싱
            tree = self.parser.parse(bytes(source_code, "utf8"))
            
            # 캐시 저장
            self.cache_manager.set_cached_result(file_path, tree)
            
            return tree, None
            
        except FileNotFoundError:
            return None, f"파일을 찾을 수 없습니다: {file_path}"
        except Exception as e:
            return None, f"파싱 중 오류 발생: {str(e)}"
    
    def extract_class_info(self, tree: Tree, file_path: Path) -> List[ClassInfo]:
        """
        AST에서 클래스 정보를 추출
        
        Args:
            tree: AST 트리
            file_path: 파일 경로 (Path 객체 또는 문자열)
            
        Returns:
            List[ClassInfo]: 클래스 정보 목록
        """
        # Path 객체로 변환 (SourceFile 객체가 전달될 수 있으므로 path 속성 확인)
        if hasattr(file_path, 'path'):
            # SourceFile 객체인 경우 path 속성 사용
            file_path = Path(file_path.path)
        else:
            file_path = Path(file_path)
        
        classes = []
        root_node = tree.root_node
        
        # 패키지 정보 추출
        package_name = self._extract_package(root_node)
        
        # 클래스 및 인터페이스 선언 탐색
        for node in self._traverse_tree(root_node):
            if node.type == "class_declaration":
                class_info = self._parse_class_declaration(node, package_name, file_path)
                if class_info:
                    classes.append(class_info)
            elif node.type == "interface_declaration":
                # 인터페이스도 클래스와 동일하게 처리
                class_info = self._parse_class_declaration(node, package_name, file_path)
                if class_info:
                    classes.append(class_info)
        
        return classes
    
    def _extract_package(self, root_node: Node) -> str:
        """
        패키지명 추출
        
        Args:
            root_node: 루트 노드
            
        Returns:
            str: 패키지명
        """
        for child in root_node.children:
            if child.type == "package_declaration":
                for subchild in child.children:
                    if subchild.type == "scoped_identifier":
                        return subchild.text.decode('utf8')
        return ""
    
    def _parse_class_declaration(
        self, 
        node: Node, 
        package_name: str, 
        file_path: Path
    ) -> Optional[ClassInfo]:
        """
        클래스 선언 노드를 파싱하여 ClassInfo 생성
        
        Args:
            node: 클래스 선언 노드
            package_name: 패키지명
            file_path: 파일 경로
            
        Returns:
            Optional[ClassInfo]: 클래스 정보
        """
        class_info = ClassInfo(
            name="",
            package=package_name,
            file_path=str(file_path)
        )
        
        # 클래스 이름 추출
        for child in node.children:
            if child.type == "identifier":
                class_info.name = child.text.decode('utf8')
                break
        
        # 클래스 어노테이션 추출
        for child in node.children:
            if child.type == "modifiers":
                class_info.annotations.extend(self._extract_annotations(child))
        
        # 부모 클래스 및 인터페이스 추출
        for child in node.children:
            if child.type == "superclass":
                for subchild in child.children:
                    if subchild.type in ["type_identifier", "scoped_identifier", "generic_type"]:
                        class_info.superclass = subchild.text.decode('utf8')
                        break
            elif child.type == "interfaces":
                for subchild in child.children:
                    if subchild.type == "type_list":
                        for interface_node in subchild.children:
                            if interface_node.type in ["type_identifier", "scoped_identifier", "generic_type"]:
                                interface_name = interface_node.text.decode('utf8')
                                if interface_name:
                                    class_info.interfaces.append(interface_name)
        
        # 클래스/인터페이스 바디 분석
        for child in node.children:
            if child.type in ["class_body", "interface_body"]:
                for member in child.children:
                    # 필드 추출
                    if member.type == "field_declaration":
                        field_info = self._extract_field_info(member)
                        if field_info:
                            class_info.fields.append({
                                "name": field_info.name,
                                "type": field_info.type,
                                "annotations": field_info.annotations,
                                "initial_value": field_info.initial_value,
                                "access_modifier": field_info.access_modifier,
                                "is_static": field_info.is_static,
                                "is_final": field_info.is_final
                            })
                    
                    # 메서드 추출
                    elif member.type == "method_declaration":
                        method_info = self._extract_method_info(member, class_info.name, file_path)
                        if method_info:
                            class_info.methods.append(method_info)
        
        return class_info if class_info.name else None
    
    def _extract_annotations(self, node: Node) -> List[str]:
        """
        어노테이션 추출
        
        Args:
            node: 노드
            
        Returns:
            List[str]: 어노테이션 목록
        """
        annotations = []
        
        if node.type in ["marker_annotation", "annotation"]:
            for child in node.children:
                if child.type in ["identifier", "scoped_identifier"]:
                    annotation_name = child.text.decode('utf8')
                    # @ 기호 제거
                    if annotation_name.startswith('@'):
                        annotation_name = annotation_name[1:]
                    annotations.append(annotation_name)
        
        for child in node.children:
            annotations.extend(self._extract_annotations(child))
        
        return annotations
    
    def _extract_field_info(self, node: Node) -> Optional[FieldInfo]:
        """
        필드 정보 추출
        
        Args:
            node: 필드 선언 노드
            
        Returns:
            Optional[FieldInfo]: 필드 정보
        """
        field = FieldInfo(name="", type="")
        
        # 필드 어노테이션 및 접근 제어자
        for child in node.children:
            if child.type == "modifiers":
                annotations = self._extract_annotations(child)
                field.annotations.extend(annotations)
                
                # 접근 제어자 추출
                modifier_text = child.text.decode('utf8')
                if 'public' in modifier_text:
                    field.access_modifier = "public"
                elif 'private' in modifier_text:
                    field.access_modifier = "private"
                elif 'protected' in modifier_text:
                    field.access_modifier = "protected"
                
                if 'static' in modifier_text:
                    field.is_static = True
                if 'final' in modifier_text:
                    field.is_final = True
        
        # 필드 타입
        for child in node.children:
            if child.type in ["type_identifier", "generic_type", "integral_type", "floating_point_type", "boolean_type", "void_type"]:
                field.type = child.text.decode('utf8')
                break
        
        # 필드 이름 및 초기값
        for child in node.children:
            if child.type == "variable_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        field.name = subchild.text.decode('utf8')
                    elif subchild.type == "=":
                        # 초기값 추출
                        next_sibling = child.children[child.children.index(subchild) + 1] if child.children.index(subchild) + 1 < len(child.children) else None
                        if next_sibling:
                            field.initial_value = next_sibling.text.decode('utf8')
        
        return field if field.name else None
    
    def _extract_method_info(
        self, 
        node: Node, 
        class_name: str, 
        file_path: Path
    ) -> Optional[Method]:
        """
        메서드 정보 추출
        
        Args:
            node: 메서드 선언 노드
            class_name: 클래스명
            file_path: 파일 경로
            
        Returns:
            Optional[Method]: 메서드 정보
        """
        method = Method(
            name="",
            return_type="void",
            parameters=[],
            class_name=class_name,
            file_path=str(file_path)
        )
        
        # 메서드 어노테이션 및 접근 제어자
        for child in node.children:
            if child.type == "modifiers":
                method.annotations.extend(self._extract_annotations(child))
                
                # 접근 제어자 추출
                modifier_text = child.text.decode('utf8')
                if 'public' in modifier_text:
                    method.access_modifier = "public"
                elif 'private' in modifier_text:
                    method.access_modifier = "private"
                elif 'protected' in modifier_text:
                    method.access_modifier = "protected"
                
                if 'static' in modifier_text:
                    method.is_static = True
                if 'abstract' in modifier_text:
                    method.is_abstract = True
                if 'final' in modifier_text:
                    method.is_final = True
        
        # 반환 타입
        for child in node.children:
            if child.type in ["type_identifier", "generic_type", "void_type", "integral_type", "floating_point_type", "boolean_type"]:
                method.return_type = child.text.decode('utf8')
                break
        
        # 메서드 이름
        for child in node.children:
            if child.type == "identifier":
                method.name = child.text.decode('utf8')
                break
        
        # 파라미터
        for child in node.children:
            if child.type == "formal_parameters":
                method.parameters = self._extract_parameters(child)
        
        # 메서드 블록에서 지역 변수 및 메서드 호출 추출
        for child in node.children:
            if child.type == "block":
                method.local_variables = self._extract_local_variables(child)
                method.method_calls = self._extract_method_calls(child)
        
        return method if method.name else None
    
    def _extract_parameters(self, node: Node) -> List[Parameter]:
        """
        파라미터 추출
        
        Args:
            node: formal_parameters 노드
            
        Returns:
            List[Parameter]: 파라미터 목록
        """
        params = []
        
        for child in node.children:
            if child.type == "formal_parameter":
                param = Parameter(name="", type="")
                
                for subchild in child.children:
                    if subchild.type in ["type_identifier", "generic_type", "integral_type", "floating_point_type", "boolean_type"]:
                        param.type = subchild.text.decode('utf8')
                    elif subchild.type == "identifier":
                        param.name = subchild.text.decode('utf8')
                    elif subchild.type == "...":
                        param.is_varargs = True
                
                if param.name:
                    params.append(param)
        
        return params
    
    def _extract_local_variables(self, node: Node) -> List[LocalVariable]:
        """
        메서드 블록 내 지역 변수 추출
        
        Args:
            node: block 노드 (메서드 블록)
            
        Returns:
            List[LocalVariable]: 지역 변수 목록
        """
        local_vars = []
        
        # block 내부의 모든 노드를 재귀적으로 탐색
        for child in self._traverse_tree(node):
            if child.type == "local_variable_declaration":
                # 지역 변수 선언 노드 처리
                var_type = ""
                var_names = []
                
                # 타입 추출
                for subchild in child.children:
                    if subchild.type in ["type_identifier", "generic_type", "integral_type", 
                                        "floating_point_type", "boolean_type", "void_type"]:
                        var_type = subchild.text.decode('utf8')
                        break
                
                # 변수명 추출 (variable_declarator)
                for subchild in child.children:
                    if subchild.type == "variable_declarator":
                        for var_child in subchild.children:
                            if var_child.type == "identifier":
                                var_name = var_child.text.decode('utf8')
                                if var_name:
                                    var_names.append(var_name)
                
                # 각 변수명에 대해 LocalVariable 생성
                for var_name in var_names:
                    if var_type:  # 타입이 있는 경우만 추가
                        local_vars.append(LocalVariable(name=var_name, type=var_type))
        
        return local_vars
    
    def _extract_method_calls(self, node: Node) -> List[str]:
        """
        메서드 호출 추출 (Call Tree)
        
        Args:
            node: 노드
            
        Returns:
            List[str]: 메서드 호출 목록 (형식: "object.method" 또는 "method")
        """
        calls = []
        
        if node.type == "method_invocation":
            # 여러 개의 '.'이나 '::'로 연결된 경우를 처리
            # 마지막 '.'이나 '::'를 기준으로 method 호출 추출
            # 예: this.vets.findAll() -> vets.findAll()
            
            def extract_method_call_from_node(method_node: Node) -> Optional[str]:
                """method_invocation 노드에서 메서드 호출 문자열 추출 (재귀적)"""
                method_children = list(method_node.children)
                
                # argument_list 찾기 (메서드명은 argument_list 바로 앞)
                method_name = None
                argument_list_idx = -1
                for i, child in enumerate(method_children):
                    if child.type == "argument_list":
                        argument_list_idx = i
                        # argument_list 바로 앞의 identifier가 메서드명
                        if i > 0 and method_children[i - 1].type == "identifier":
                            method_name = method_children[i - 1].text.decode('utf8')
                            if method_name == "vetToVetDto":
                                print(method_name)
                        break
                
                # argument_list가 없는 경우 (예: this.vets.findAll()에서 this 부분)
                # 중첩된 method_invocation이 있으면 그 결과를 반환
                if not method_name:
                    for child in method_children:
                        if child.type == "method_invocation":
                            # 중첩된 method_invocation의 결과를 그대로 반환
                            nested_result = extract_method_call_from_node(child)
                            if nested_result:
                                return nested_result
                    return None
                
                # argument_list 이전의 부분에서 object 부분 추출
                # 여러 개의 '.'이나 '::'로 연결된 경우 처리
                parts = []
                separator = None
                
                # argument_list 이전의 children만 처리
                for i in range(argument_list_idx):
                    child = method_children[i]
                    
                    if child.type == "identifier":
                        identifier_text = child.text.decode('utf8')
                        # this는 제외
                        if identifier_text != "this":
                            parts.append(identifier_text)
                            if identifier_text == "vetToVetDto":
                                print(method_name)
                    elif child.type == "field_access":
                        # field_access 내부를 재귀적으로 처리
                        field_result = extract_from_field_access(child)
                        if field_result:
                            parts.append(field_result)
                    elif child.type == "method_invocation":
                        # 중첩된 method_invocation (예: this.vets.findAll())
                        # 재귀적으로 처리하되, 전체 체인을 parts에 추가
                        # 예: this.vets.findAll() -> vets.findAll()이 nested_result
                        # 하지만 여기서는 vets만 parts에 추가해야 함
                        nested_result = extract_method_call_from_node(child)
                        if nested_result:
                            # nested_result는 "object.method" 형식
                            # 마지막 '.' 또는 '::'를 기준으로 object 부분만 추출
                            if '.' in nested_result:
                                object_part = nested_result.rsplit('.', 1)[0]
                                if object_part:
                                    parts.append(object_part)
                            elif '::' in nested_result:
                                object_part = nested_result.rsplit('::', 1)[0]
                                if object_part:
                                    parts.append(object_part)
                            # method만 있는 경우는 parts에 추가하지 않음
                    elif child.type == ".":
                        separator = "."
                    elif child.type == "::":
                        separator = "::"
                
                # 마지막 '.' 또는 '::'를 기준으로 object.method 추출
                if len(parts) >= 2:
                    # 마지막 두 요소를 사용: parts[-2]는 object, parts[-1]은 무시 (이미 method_name에 있음)
                    # 실제로는 parts의 마지막 전까지가 object
                    object_parts = parts[:-1] if len(parts) > 1 else []
                    if object_parts:
                        object_name = separator.join(object_parts) if separator else object_parts[0]
                        return f"{object_name}{separator or '.'}{method_name}"
                    else:
                        return method_name
                elif len(parts) == 1:
                    # parts[0]가 method_name과 같으면 단순 메서드 호출 (예: myfunc())
                    # parts[0]가 method_name과 다르면 object.method 형식 (예: obj.method())
                    if parts[0] == method_name:
                        # method() 형식
                        return method_name
                    else:
                        # object.method 형식
                        return f"{parts[0]}{separator or '.'}{method_name}"
                else:
                    # method() 형식 (parts가 비어있고 method_name만 있는 경우)
                    return method_name
            
            def extract_from_field_access(field_node: Node) -> Optional[str]:
                """field_access 노드에서 필드 접근 문자열 추출 (재귀적)"""
                field_children = list(field_node.children)
                parts = []
                separator = None
                
                for child in field_children:
                    if child.type == "identifier":
                        identifier_text = child.text.decode('utf8')
                        if identifier_text != "this":
                            parts.append(identifier_text)
                    elif child.type == "field_access":
                        # 중첩된 field_access
                        nested_result = extract_from_field_access(child)
                        if nested_result:
                            parts.append(nested_result)
                    elif child.type == ".":
                        separator = "."
                    elif child.type == "::":
                        separator = "::"
                
                if parts:
                    return separator.join(parts) if separator and len(parts) > 1 else parts[0]
                return None
            
            # method_invocation에서 메서드 호출 추출
            method_call = extract_method_call_from_node(node)
            if method_call:
                calls.append(method_call)
        
        for child in node.children:
            calls.extend(self._extract_method_calls(child))
        
        return calls
    
    def _traverse_tree(self, node: Node):
        """
        트리를 재귀적으로 탐색하는 제너레이터
        
        Args:
            node: 시작 노드
            
        Yields:
            Node: 각 노드
        """
        yield node
        for child in node.children:
            yield from self._traverse_tree(child)
    
    def extract_call_relations(
        self, 
        classes: List[ClassInfo]
    ) -> List[CallRelation]:
        """
        CallRelation 목록 추출
        
        Args:
            classes: 클래스 정보 목록
            
        Returns:
            List[CallRelation]: 호출 관계 목록
        """
        relations = []
        
        for cls in classes:
            for method in cls.methods:
                caller = f"{cls.name}.{method.name}"
                caller_file = method.file_path
                
                for call in method.method_calls:
                    # call 형식이 "object.method"인 경우 callee는 "method"만 사용
                    if '.' in call:
                        callee = call.split('.')[-1]
                    else:
                        callee = call
                    
                    # callee의 파일 경로 찾기 (같은 클래스 내 메서드인 경우)
                    callee_file = caller_file
                    for other_cls in classes:
                        for other_method in other_cls.methods:
                            if other_method.name == callee:
                                callee_file = other_method.file_path
                                break
                    
                    relation = CallRelation(
                        caller=caller,
                        callee=f"{cls.name}.{callee}",
                        caller_file=caller_file,
                        callee_file=callee_file
                    )
                    relations.append(relation)
        
        return relations
    
    def fallback_parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Tree-sitter 파싱 실패 시 정규표현식 기반 Fallback 파서
        
        Args:
            file_path: Java 파일 경로
            
        Returns:
            Dict[str, Any]: 파싱 결과 (클래스명, 메서드명, 필드명)
        """
        # 여러 인코딩 시도
        source_code = None
        encodings = ['utf-8', 'euc-kr', 'cp949', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    source_code = f.read()
                break  # 성공하면 루프 종료
            except UnicodeDecodeError:
                continue  # 다음 인코딩 시도
            except Exception as e:
                # 다른 에러는 마지막 인코딩까지 시도 후 에러 반환
                if encoding == encodings[-1]:
                    return {"error": f"파일 읽기 실패: {str(e)}"}
                continue
        
        if source_code is None:
            return {"error": "파일을 읽을 수 없습니다: 지원되는 인코딩을 찾을 수 없습니다"}
        
        result = {
            "classes": [],
            "methods": [],
            "fields": []
        }
        
        # 클래스명 추출
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, source_code)
        result["classes"] = classes
        
        # 메서드명 추출
        method_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\('
        methods = re.findall(method_pattern, source_code)
        result["methods"] = methods
        
        # 필드명 추출
        field_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*[=;]'
        fields = re.findall(field_pattern, source_code)
        result["fields"] = fields
        
        return result
    
    def print_class_info(self, classes: List[ClassInfo]) -> None:
        """
        클래스 정보를 예제 코드 형식으로 출력
        
        Args:
            classes: 클래스 정보 목록
        """
        for cls in classes:
            print(f"\n{'='*60}")
            print(f"Class: {cls.name}")
            print(f"{'='*60}")
            
            # 클래스 어노테이션
            if cls.annotations:
                print("\n[Class Annotations]")
                for ann in cls.annotations:
                    print(f"  @{ann}")
            
            # 필드 정보
            if cls.fields:
                print("\n[Fields]")
                for field in cls.fields:
                    ann_str = ", ".join([f"@{a}" for a in field["annotations"]]) if field["annotations"] else ""
                    print(f"  {ann_str} {field['type']} {field['name']}")
            
            # 메서드 정보
            if cls.methods:
                print("\n[Methods]")
                for method in cls.methods:
                    # 메서드 시그니처
                    ann_str = " ".join([f"@{a}" for a in method.annotations]) if method.annotations else ""
                    params_str = ", ".join([f"{p.type} {p.name}" for p in method.parameters])
                    print(f"\n  {ann_str}")
                    print(f"  {method.return_type} {method.name}({params_str})")
                    
                    # 메서드 내부 호출
                    if method.method_calls:
                        print(f"    └─ Calls:")
                        for call in method.method_calls:
                            if '.' in call:
                                print(f"       • {call}()")
                            else:
                                print(f"       • {call}()")
    
    def print_call_graph(self, call_graph: Dict[str, List[str]]) -> None:
        """
        Call Graph를 예제 코드 형식으로 출력
        
        Args:
            call_graph: Call Graph 딕셔너리
        """
        print(f"\n\n{'='*60}")
        print("CALL GRAPH")
        print(f"{'='*60}\n")
        
        def print_call_tree(method: str, visited: set = None, indent: int = 0):
            """재귀적으로 호출 트리 출력"""
            if visited is None:
                visited = set()
            
            if method in visited:
                print("  " * indent + f"└─ {method} (recursive/circular)")
                return
            
            visited.add(method)
            print("  " * indent + f"└─ {method}")
            
            if method in call_graph:
                for called in call_graph[method]:
                    print_call_tree(called, visited.copy(), indent + 1)
        
        # 루트 메서드 찾기 (다른 메서드에서 호출되지 않는 메서드)
        all_methods = set(call_graph.keys())
        called_methods = set()
        for calls in call_graph.values():
            called_methods.update(calls)
        
        root_methods = all_methods - called_methods
        
        if root_methods:
            print("Root Methods (entry points):\n")
            for root in root_methods:
                print_call_tree(root)
                print()
        else:
            print("All methods in call graph:\n")
            for method in call_graph.keys():
                print_call_tree(method)
                print()
    
    def extract_jdbc_sql(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        JDBC를 사용하는 Java 파일에서 SQL 쿼리 추출
        
        Args:
            file_path: Java 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 추출된 SQL 쿼리 목록
                각 항목은 {"id": str, "query_type": str, "sql": str, "strategy_specific": dict} 형태
        """
        sql_queries = []
        
        # 파일 읽기
        source_code = None
        encodings = ['utf-8', 'euc-kr', 'cp949', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    source_code = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        if not source_code:
            return sql_queries
        
        # JDBC 패턴 찾기: executeQuery, executeUpdate, prepareStatement 등
        # prepareStatement("SELECT ...") 또는 executeQuery("SELECT ...") 패턴
        jdbc_patterns = [
            # prepareStatement("SQL")
            (r'prepareStatement\s*\(\s*["\']([^"\']+)["\']', "SELECT"),
            # executeQuery("SQL")
            (r'executeQuery\s*\(\s*["\']([^"\']+)["\']', "SELECT"),
            # executeUpdate("SQL")
            (r'executeUpdate\s*\(\s*["\']([^"\']+)["\']', "UPDATE"),
            # execute("SQL")
            (r'execute\s*\(\s*["\']([^"\']+)["\']', "SELECT"),
        ]
        
        # 메서드 내에서 SQL 문자열 찾기
        # 메서드 시그니처 추출
        method_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\('
        methods = re.finditer(method_pattern, source_code)
        
        for method_match in methods:
            method_name = method_match.group(1)
            method_start = method_match.start()
            
            # 메서드 끝 찾기 (다음 메서드 또는 클래스 끝)
            next_method = re.search(r'(?:public|private|protected)?\s+\w+\s+\w+\s*\(', source_code[method_match.end():])
            if next_method:
                method_end = method_match.end() + next_method.start()
            else:
                method_end = len(source_code)
            
            method_body = source_code[method_start:method_end]
            
            # JDBC 패턴 매칭
            for pattern, default_query_type in jdbc_patterns:
                matches = re.finditer(pattern, method_body, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    sql = match.group(1)
                    # SQL 타입 자동 감지
                    query_type = self._detect_query_type(sql)
                    if not query_type:
                        query_type = default_query_type
                    
                    sql_queries.append({
                        "id": method_name,
                        "query_type": query_type,
                        "sql": sql.strip(),
                        "strategy_specific": {
                            "method_name": method_name,
                            "file_path": str(file_path)
                        }
                    })
        
        return sql_queries
    
    def extract_jpa_sql(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        JPA를 사용하는 Java 파일에서 JPQL/Native SQL 쿼리 추출
        
        Args:
            file_path: Java 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 추출된 SQL 쿼리 목록
                각 항목은 {"id": str, "query_type": str, "sql": str, "strategy_specific": dict} 형태
        """
        sql_queries = []
        
        # 파일 읽기
        source_code = None
        encodings = ['utf-8', 'euc-kr', 'cp949', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    source_code = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        if not source_code:
            return sql_queries
        
        # JPA 패턴 찾기
        # @Query("SELECT ...") 어노테이션
        query_annotation_pattern = r'@Query\s*\(\s*value\s*=\s*["\']([^"\']+)["\']'
        query_matches = re.finditer(query_annotation_pattern, source_code, re.IGNORECASE | re.DOTALL)
        
        for match in query_matches:
            sql = match.group(1)
            # 다음 메서드 찾기
            method_start = match.end()
            method_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\('
            method_match = re.search(method_pattern, source_code[method_start:])
            
            method_name = method_match.group(1) if method_match else "unknown"
            query_type = self._detect_query_type(sql)
            if not query_type:
                query_type = "SELECT"
            
            sql_queries.append({
                "id": method_name,
                "query_type": query_type,
                "sql": sql.strip(),
                "strategy_specific": {
                    "method_name": method_name,
                    "file_path": str(file_path),
                    "annotation": "@Query"
                }
            })
        
        # @NamedQuery 어노테이션
        named_query_pattern = r'@NamedQuery\s*\(\s*name\s*=\s*["\']([^"\']+)["\']\s*,\s*query\s*=\s*["\']([^"\']+)["\']'
        named_matches = re.finditer(named_query_pattern, source_code, re.IGNORECASE | re.DOTALL)
        
        for match in named_matches:
            query_name = match.group(1)
            sql = match.group(2)
            query_type = self._detect_query_type(sql)
            if not query_type:
                query_type = "SELECT"
            
            sql_queries.append({
                "id": query_name,
                "query_type": query_type,
                "sql": sql.strip(),
                "strategy_specific": {
                    "query_name": query_name,
                    "file_path": str(file_path),
                    "annotation": "@NamedQuery"
                }
            })
        
        # EntityManager.createQuery("SELECT ...") 또는 createNativeQuery("SELECT ...")
        entity_manager_patterns = [
            (r'createQuery\s*\(\s*["\']([^"\']+)["\']', "SELECT"),
            (r'createNativeQuery\s*\(\s*["\']([^"\']+)["\']', "SELECT"),
        ]
        
        method_pattern = r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\('
        methods = re.finditer(method_pattern, source_code)
        
        for method_match in methods:
            method_name = method_match.group(1)
            method_start = method_match.start()
            
            next_method = re.search(r'(?:public|private|protected)?\s+\w+\s+\w+\s*\(', source_code[method_match.end():])
            if next_method:
                method_end = method_match.end() + next_method.start()
            else:
                method_end = len(source_code)
            
            method_body = source_code[method_start:method_end]
            
            for pattern, default_query_type in entity_manager_patterns:
                matches = re.finditer(pattern, method_body, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    sql = match.group(1)
                    query_type = self._detect_query_type(sql)
                    if not query_type:
                        query_type = default_query_type
                    
                    sql_queries.append({
                        "id": method_name,
                        "query_type": query_type,
                        "sql": sql.strip(),
                        "strategy_specific": {
                            "method_name": method_name,
                            "file_path": str(file_path)
                        }
                    })
        
        return sql_queries
    
    def _detect_query_type(self, sql: str) -> Optional[str]:
        """
        SQL 쿼리 타입 자동 감지
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            Optional[str]: 쿼리 타입 (SELECT, INSERT, UPDATE, DELETE) 또는 None
        """
        sql_upper = sql.strip().upper()
        
        if sql_upper.startswith("SELECT"):
            return "SELECT"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        
        return None

