"""
Call Graph Builder

Java AST 정보를 기반으로 메서드 호출 관계를 추적하여 그래프 구조를 생성하고,
REST API 엔드포인트부터 DAO/Mapper까지 이어지는 호출 체인을 구성하는 모듈입니다.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    import networkx as nx
except ImportError:
    nx = None

from models.call_relation import CallRelation
from models.method import Method
from .java_ast_parser import JavaASTParser, ClassInfo
from persistence.cache_manager import CacheManager


@dataclass
class Endpoint:
    """
    REST API 엔드포인트 정보
    
    Attributes:
        path: 엔드포인트 경로
        http_method: HTTP 메서드 (GET, POST, PUT, DELETE 등)
        method_signature: 메서드 시그니처 (ClassName.methodName)
        class_name: 클래스명
        method_name: 메서드명
        file_path: 파일 경로
    """
    path: str
    http_method: str
    method_signature: str
    class_name: str
    method_name: str
    file_path: str
    
    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            "path": self.path,
            "http_method": self.http_method,
            "method_signature": self.method_signature,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "file_path": self.file_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Endpoint":
        """딕셔너리로부터 Endpoint 객체 생성"""
        return cls(
            path=data["path"],
            http_method=data["http_method"],
            method_signature=data["method_signature"],
            class_name=data["class_name"],
            method_name=data["method_name"],
            file_path=data["file_path"]
        )


@dataclass
class CallChain:
    """
    호출 체인 정보
    
    Attributes:
        chain: 호출 체인 (메서드 시그니처 리스트)
        layers: 각 메서드의 레이어 정보
        is_circular: 순환 참조 여부
    """
    chain: List[str]
    layers: List[str] = field(default_factory=list)
    is_circular: bool = False


class CallGraphBuilder:
    """
    Call Graph Builder 클래스
    
    Java AST 정보를 기반으로 메서드 호출 관계 그래프를 생성하고,
    REST API 엔드포인트부터 시작하는 호출 체인을 구성합니다.
    """
    
    # Spring 및 프레임워크 어노테이션 패턴
    SPRING_ANNOTATIONS = {
        # Spring MVC
        'RequestMapping', 'GetMapping', 'PostMapping', 'PutMapping', 
        'DeleteMapping', 'PatchMapping', 'Controller', 'RestController',
        # Spring Core
        'Service', 'Repository', 'Component', 'Autowired', 'Qualifier',
        # MyBatis
        'Mapper', 'Select', 'Insert', 'Update', 'Delete', 'Param',
        # JPA
        'Entity', 'Table', 'Id', 'GeneratedValue', 'Column', 'OneToMany',
        'ManyToOne', 'OneToOne', 'ManyToMany', 'JoinColumn', 'Query',
        'NamedQuery', 'NamedQueries', 'EntityManager', 'PersistenceContext',
        # JDBC 관련 (직접 사용하는 경우는 어노테이션이 없을 수 있음)
        'Transactional'
    }
    
    # 레이어 분류 패턴 (MyBatis, JDBC, JPA 모두 지원)
    LAYER_PATTERNS = {
        'Controller': ['Controller', 'RestController', 'WebController'],
        'Service': ['Service', 'BusinessService', 'ApplicationService'],
        'Repository': ['Repository', 'JpaRepository', 'CrudRepository', 'DAO', 'Dao', 'JdbcDao', 'JdbcTemplateDao'],
        'Mapper': ['Mapper', 'MyBatisMapper', 'SqlMapper'],
        'Entity': ['Entity', 'Domain', 'Model', 'POJO']
    }
    
    def __init__(self, java_parser: Optional[JavaASTParser] = None, cache_manager: Optional[CacheManager] = None):
        """
        CallGraphBuilder 초기화
        
        Args:
            java_parser: Java AST 파서 (선택적)
            cache_manager: 캐시 매니저 (선택적)
        """
        if nx is None:
            raise ImportError("networkx 라이브러리가 필요합니다. pip install networkx로 설치하세요.")
        
        self.cache_manager = cache_manager
        # java_parser가 없으면 cache_manager를 전달하여 생성
        if java_parser is None:
            if cache_manager is None:
                # 임시 디렉터리 사용
                from tempfile import mkdtemp
                cache_dir = Path(mkdtemp())
                self.cache_manager = CacheManager(cache_dir=cache_dir)
            self.java_parser = JavaASTParser(cache_manager=self.cache_manager)
        else:
            self.java_parser = java_parser
        
        self.logger = logging.getLogger("applycrypto")
        
        # Call Graph (networkx DiGraph)
        self.call_graph: Optional[nx.DiGraph] = None
        
        # 메서드 메타데이터 (메서드 시그니처 -> 메서드 정보)
        self.method_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 클래스 정보 (클래스명 -> ClassInfo)
        self.class_info_map: Dict[str, ClassInfo] = {}
        
        # 파일 경로 -> 클래스 정보 리스트 매핑 (파싱된 정보 재사용용)
        self.file_to_classes_map: Dict[str, List[ClassInfo]] = {}
        
        # 엔드포인트 목록
        self.endpoints: List[Endpoint] = []
    
    def build_call_graph(self, java_files: List[Path]) -> nx.DiGraph:
        """
        Java 파일 목록으로부터 Call Graph 생성
        
        Args:
            java_files: Java 파일 경로 목록
            
        Returns:
            nx.DiGraph: Call Graph
        """
        # 그래프 초기화
        self.call_graph = nx.DiGraph()
        self.method_metadata = {}
        self.class_info_map = {}
        self.file_to_classes_map = {}
        self.endpoints = []
        
        # 모든 Java 파일 파싱
        all_classes: List[ClassInfo] = []
        for file_path in java_files:
            tree, error = self.java_parser.parse_file(file_path)
            if error:
                self.logger.warning(f"파일 파싱 실패: {file_path} - {error}")
                continue
            
            classes = self.java_parser.extract_class_info(tree, file_path)
            all_classes.extend(classes)
            
            # 파일 경로 -> 클래스 정보 매핑 저장 (재사용용)
            file_path_str = str(file_path)
            self.file_to_classes_map[file_path_str] = classes
            
            # 클래스 정보 저장
            for cls in classes:
                self.class_info_map[cls.name] = cls
        
        # 클래스별 필드 정보 수집 (필드명 -> 타입 매핑)
        class_field_map: Dict[str, Dict[str, str]] = {}  # 클래스명 -> {필드명: 타입}
        for cls in all_classes:
            field_map = {}
            for field in cls.fields:
                field_name = field.get("name", "")
                field_type = field.get("type", "")
                if field_name and field_type:
                    # 제네릭 타입 처리 (예: List<User> -> List)
                    if '<' in field_type:
                        field_type = field_type.split('<')[0]
                    field_map[field_name] = field_type
            class_field_map[cls.name] = field_map
        
        # 메서드 호출 관계 추출
        call_relations = []
        for cls in all_classes:
            for method in cls.methods:
                method_signature = f"{cls.name}.{method.name}"
                
                # 메서드 메타데이터 저장
                self.method_metadata[method_signature] = {
                    "class_name": cls.name,
                    "method": method,
                    "file_path": cls.file_path,
                    "package": cls.package,
                    "annotations": method.annotations,
                    "layer": self._classify_layer(cls, method)
                }

                # 현재 클래스의 필드 정보 가져오기
                current_field_map = class_field_map.get(cls.name, {})
                
                # 메서드의 parameters와 local_variables를 가져와서 method_variable_map 구성
                method_variable_map: Dict[str, str] = {}  # 변수명 -> 타입
                
                # 메서드의 parameter들의 type 처리
                for param in method.parameters:
                    param_name = param.name
                    param_type = param.type
                    if param_name and param_type:
                        # 제네릭 타입 처리 (예: List<User> -> List)
                        if '<' in param_type:
                            param_type = param_type.split('<')[0]
                        method_variable_map[param_name] = param_type
                
                # 메서드의 local_variables들의 type 처리
                for local_var in method.local_variables:
                    var_name = local_var.name
                    var_type = local_var.type
                    if var_name and var_type:
                        # 제네릭 타입 처리 (예: List<User> -> List)
                        if '<' in var_type:
                            var_type = var_type.split('<')[0]
                        method_variable_map[var_name] = var_type
                
                # 메서드 호출 관계 추출
                for call in method.method_calls:
                    callee_signature = None
                    callee_file = cls.file_path
                    
                    # call 형식이 "object.method"인 경우 처리
                    if '.' in call:
                        parts = call.split('.')
                        if len(parts) >= 2:
                            # object.method 형식
                            object_name = parts[0]  # 필드명 또는 변수명
                            callee_method = parts[-1]
                            
                        
                            # 필드 변수를 통한 호출인지 확인
                            if object_name in current_field_map:
                                # 필드 타입 찾기
                                field_type = current_field_map[object_name]
                                
                                # 필드 타입이 다른 클래스인 경우 해당 클래스의 메서드로 매핑
                                if field_type in self.class_info_map:
                                    callee_signature = f"{field_type}.{callee_method}"
                                    callee_cls = self.class_info_map[field_type]
                                    callee_file = callee_cls.file_path
                                else:
                                    # 필드 타입 클래스를 찾을 수 없는 경우 필드 타입으로 매핑 시도
                                    callee_signature = f"{field_type}.{callee_method}"
                            elif object_name in method_variable_map:
                                # 메서드 변수(파라미터 또는 리턴 타입)를 통한 호출인지 확인
                                variable_type = method_variable_map[object_name]
                                
                                # 변수 타입이 다른 클래스인 경우 해당 클래스의 메서드로 매핑
                                if variable_type in self.class_info_map:
                                    callee_signature = f"{variable_type}.{callee_method}"
                                    callee_cls = self.class_info_map[variable_type]
                                    callee_file = callee_cls.file_path
                                else:
                                    # 변수 타입 클래스를 찾을 수 없는 경우 변수 타입으로 매핑 시도
                                    callee_signature = f"{variable_type}.{callee_method}"
                            else:
                                # 필드가 아니거나 찾을 수 없는 경우 같은 클래스 내 메서드로 간주
                                # callee_signature = f"{cls.name}.{callee_method}"
                                # 현재 클래스로 대체하지 말고 변수 이름을 signature에 남겨두자.
                                callee_signature = call

                        else:
                            callee_signature = f"{cls.name}.{call}"
                    else:
                        # 같은 클래스 내 메서드 호출
                        callee_signature = f"{cls.name}.{call}"
                    
                    if callee_signature:
                        relation = CallRelation(
                            caller=method_signature,
                            callee=callee_signature,
                            caller_file=cls.file_path,
                            callee_file=callee_file
                        )
                        call_relations.append(relation)
        
        # 그래프에 노드 및 간선 추가
        for relation in call_relations:
            # 노드 추가 (메타데이터 포함)
            if relation.caller not in self.call_graph:
                metadata = self.method_metadata.get(relation.caller, {})
                self.call_graph.add_node(
                    relation.caller,
                    class_name=metadata.get("class_name", ""),
                    file_path=metadata.get("file_path", ""),
                    layer=metadata.get("layer", "Unknown")
                )
            
            if relation.callee not in self.call_graph:
                metadata = self.method_metadata.get(relation.callee, {})
                self.call_graph.add_node(
                    relation.callee,
                    class_name=metadata.get("class_name", ""),
                    file_path=metadata.get("file_path", ""),
                    layer=metadata.get("layer", "Unknown")
                )
            
            # 간선 추가
            self.call_graph.add_edge(relation.caller, relation.callee)
        
        # 엔드포인트 식별
        self._identify_endpoints(all_classes)
        
        return self.call_graph
    
    def _classify_layer(self, cls: ClassInfo, method: Method) -> str:
        """
        클래스와 메서드의 레이어 분류 (MyBatis, JDBC, JPA 모두 지원)
        
        Args:
            cls: 클래스 정보
            method: 메서드 정보
            
        Returns:
            str: 레이어명 (Controller, Service, DAO, Repository, Mapper, Entity, Unknown)
        """
        # 어노테이션 기반 분류 (우선순위 높음)
        all_annotations = cls.annotations + method.annotations
        annotation_lower = [ann.lower() for ann in all_annotations]
        
        # Controller 레이어
        if any('controller' in ann or 'restcontroller' in ann for ann in annotation_lower):
            return 'Controller'
        
        # Service 레이어
        if any('service' in ann for ann in annotation_lower):
            return 'Service'
        
        # MyBatis Mapper 레이어
        if any('mapper' in ann for ann in annotation_lower):
            return 'Mapper'
        
        # JPA Repository 레이어
        if any('repository' in ann for ann in annotation_lower):
            return 'Repository'
        
        # JPA Entity 레이어
        if any('entity' in ann or 'table' in ann for ann in annotation_lower):
            return 'Entity'
        
        # 클래스명 패턴 기반 분류
        class_name = cls.name
        for layer, patterns in self.LAYER_PATTERNS.items():
            for pattern in patterns:
                if pattern in class_name:
                    return layer
        
        # 인터페이스 기반 분류 (MyBatis Mapper 인터페이스 감지)
        if cls.interfaces:
            for interface in cls.interfaces:
                interface_lower = interface.lower()
                # MyBatis Mapper 인터페이스 패턴
                if 'mapper' in interface_lower or 'sqlmapper' in interface_lower:
                    return 'Mapper'
                # JPA Repository 인터페이스 패턴
                if 'repository' in interface_lower or 'jparepository' in interface_lower:
                    return 'Repository'
                # Spring Repository 인터페이스 패턴
                if 'crudrepository' in interface_lower or 'pagerepository' in interface_lower:
                    return 'Repository'
        
        # 패키지 기반 분류
        package = cls.package.lower()
        if 'controller' in package or 'web' in package or 'api' in package:
            return 'Controller'
        elif 'service' in package or 'business' in package:
            return 'Service'
        elif 'mapper' in package or 'mybatis' in package:
            return 'Mapper'
        elif 'repository' in package or 'jpa' in package:
            return 'Repository'
        elif 'dao' in package or 'data' in package:
            return 'DAO'
        elif 'entity' in package or 'domain' in package or 'model' in package or 'beans' in package:
            return 'Entity'
        
        # 필드 기반 추론 (JPA EntityManager, MyBatis SqlSession 등)
        for field in cls.fields:
            field_type = field.get("type", "").lower()
            if 'entitymanager' in field_type or 'entitymanagerfactory' in field_type:
                return 'Repository'  # JPA Repository로 추론
            elif 'sqlsession' in field_type or 'sqlsessiontemplate' in field_type:
                return 'Mapper'  # MyBatis Mapper로 추론
            elif 'jdbctemplate' in field_type or 'datasource' in field_type:
                return 'DAO'  # JDBC DAO로 추론
        
        return 'Unknown'
    
    def _extract_path_from_annotation(self, annotation: str) -> Optional[str]:
        """
        어노테이션 문자열에서 path(value 또는 path 속성) 추출
        
        Args:
            annotation: 어노테이션 문자열 (예: "@GetMapping(\"/users\")" 또는 "@RequestMapping(value=\"/api\")")
            
        Returns:
            Optional[str]: 추출된 path 또는 None
        """
        if not annotation:
            return None
        
        # 어노테이션 전체 텍스트를 가져오기 위해 AST에서 직접 추출 시도
        # 하지만 현재는 문자열만 있으므로 정규표현식으로 파싱
        
        # 패턴 1: @GetMapping("/path") 또는 @GetMapping(value="/path")
        # 패턴 2: @RequestMapping(value="/path") 또는 @RequestMapping(path="/path")
        # 패턴 3: @GetMapping() - path 없음
        
        # value="/path" 또는 path="/path" 또는 "/path" 형식 추출
        patterns = [
            r'value\s*=\s*["\']([^"\']+)["\']',  # value="/path"
            r'path\s*=\s*["\']([^"\']+)["\']',   # path="/path"
            r'\(\s*["\']([^"\']+)["\']\s*\)',    # ("/path")
            r'\(\s*["\']([^"\']+)["\']',          # ("/path" (닫는 괄호 없을 수도)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, annotation)
            if match:
                path = match.group(1)
                if path:
                    return path
        
        return None
    
    def _extract_http_method_from_annotation(self, annotation: str) -> Optional[str]:
        """
        어노테이션에서 HTTP 메서드 추출
        
        Args:
            annotation: 어노테이션 문자열
            
        Returns:
            Optional[str]: HTTP 메서드 (GET, POST, PUT, DELETE, PATCH) 또는 None
        """
        if 'GetMapping' in annotation:
            return 'GET'
        elif 'PostMapping' in annotation:
            return 'POST'
        elif 'PutMapping' in annotation:
            return 'PUT'
        elif 'DeleteMapping' in annotation:
            return 'DELETE'
        elif 'PatchMapping' in annotation:
            return 'PATCH'
        elif 'RequestMapping' in annotation:
            # @RequestMapping(method = RequestMethod.GET) 형식 처리
            method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', annotation, re.IGNORECASE)
            if method_match:
                return method_match.group(1).upper()
            # 기본값은 GET
            return 'GET'
        
        return None
    
    def _get_annotation_text_from_file(self, file_path: str, target_name: str, is_class: bool = True) -> Dict[str, str]:
        """
        파일에서 어노테이션 전체 텍스트 추출
        
        Args:
            file_path: 파일 경로
            target_name: 클래스명 또는 메서드명
            is_class: True면 클래스, False면 메서드
            
        Returns:
            Dict[str, str]: 어노테이션 이름 -> 전체 텍스트 매핑
        """
        annotation_map = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception:
            try:
                with open(file_path, 'r', encoding='euc-kr') as f:
                    source_code = f.read()
            except Exception:
                return annotation_map
        
        if is_class:
            # 클래스 어노테이션 추출
            # class ClassName 또는 public class ClassName 앞의 어노테이션들 찾기
            pattern = rf'(?:@\w+(?:\([^)]*\))?\s*)+class\s+{re.escape(target_name)}\b'
            match = re.search(pattern, source_code, re.MULTILINE | re.DOTALL)
            if match:
                # 매칭된 부분에서 어노테이션 추출
                matched_text = source_code[:match.end()]
                # class 키워드 이전 부분
                before_class = matched_text[:matched_text.rfind('class')]
                # 어노테이션 패턴 찾기
                annotation_pattern = r'@(\w+)(\([^)]*\))?'
                for ann_match in re.finditer(annotation_pattern, before_class):
                    ann_name = ann_match.group(1)
                    ann_full = ann_match.group(0)
                    annotation_map[ann_name] = ann_full
        else:
            # 메서드 어노테이션 추출
            # 메서드 시그니처 앞의 어노테이션들 찾기
            # @GetMapping(...) public ReturnType methodName(...) 패턴
            pattern = rf'(?:@\w+(?:\([^)]*\))?\s*)+(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?\w+\s+{re.escape(target_name)}\s*\('
            match = re.search(pattern, source_code, re.MULTILINE | re.DOTALL)
            if match:
                # 매칭된 부분에서 어노테이션 추출
                matched_text = source_code[:match.end()]
                # 메서드명 이전 부분
                method_name_pos = matched_text.rfind(target_name)
                before_method = matched_text[:method_name_pos]
                # 어노테이션 패턴 찾기
                annotation_pattern = r'@(\w+)(\([^)]*\))?'
                for ann_match in re.finditer(annotation_pattern, before_method):
                    ann_name = ann_match.group(1)
                    ann_full = ann_match.group(0)
                    annotation_map[ann_name] = ann_full
        
        return annotation_map
    
    def _identify_endpoints(self, classes: List[ClassInfo]) -> None:
        """
        REST API 엔드포인트 식별
        
        Args:
            classes: 클래스 정보 목록
        """
        self.endpoints = []
        
        for cls in classes:
            # 클래스 레벨 경로 추출
            class_path = ""
            # 파일에서 클래스 어노테이션 전체 텍스트 가져오기
            class_annotations = self._get_annotation_text_from_file(cls.file_path, cls.name, is_class=True)
            
            for annotation_name in cls.annotations:
                if 'RequestMapping' in annotation_name:
                    # 파일에서 실제 어노테이션 텍스트 가져오기
                    full_annotation = class_annotations.get(annotation_name, annotation_name)
                    extracted_path = self._extract_path_from_annotation(full_annotation)
                    if extracted_path:
                        class_path = extracted_path
                    break
            
            # 메서드 레벨 엔드포인트 식별
            for method in cls.methods:
                endpoint = self._extract_endpoint(cls, method, class_path)
                if endpoint:
                    self.endpoints.append(endpoint)
    
    def _extract_endpoint(
        self, 
        cls: ClassInfo, 
        method: Method, 
        class_path: str
    ) -> Optional[Endpoint]:
        """
        메서드에서 엔드포인트 정보 추출
        
        Args:
            cls: 클래스 정보
            method: 메서드 정보
            class_path: 클래스 레벨 경로
            
        Returns:
            Optional[Endpoint]: 엔드포인트 정보
        """
        http_method = None
        method_path = ""
        
        # 파일에서 메서드 어노테이션 전체 텍스트 가져오기
        method_annotations = self._get_annotation_text_from_file(cls.file_path, method.name, is_class=False)
        
        # 메서드 어노테이션 확인
        for annotation_name in method.annotations:
            # 파일에서 실제 어노테이션 텍스트 가져오기
            full_annotation = method_annotations.get(annotation_name, annotation_name)
            
            # HTTP 메서드 추출
            extracted_method = self._extract_http_method_from_annotation(full_annotation)
            if extracted_method:
                http_method = extracted_method
                # path 추출
                extracted_path = self._extract_path_from_annotation(full_annotation)
                if extracted_path:
                    method_path = extracted_path
                break  # 첫 번째 매칭되는 어노테이션 사용
        
        if http_method:
            # class_path와 method_path 결합
            if class_path and method_path:
                # 둘 다 슬래시로 시작하면 하나 제거
                if class_path.endswith('/') and method_path.startswith('/'):
                    full_path = class_path + method_path[1:]
                elif not class_path.endswith('/') and not method_path.startswith('/'):
                    full_path = class_path + '/' + method_path
                else:
                    full_path = class_path + method_path
            elif class_path:
                full_path = class_path
            elif method_path:
                full_path = method_path
            else:
                full_path = ""
            
            method_signature = f"{cls.name}.{method.name}"
            
            return Endpoint(
                path=full_path,
                http_method=http_method,
                method_signature=method_signature,
                class_name=cls.name,
                method_name=method.name,
                file_path=cls.file_path
            )
        
        return None
    
    def get_endpoints(self) -> List[Endpoint]:
        """
        식별된 엔드포인트 목록 반환
        
        Returns:
            List[Endpoint]: 엔드포인트 목록
        """
        return self.endpoints
    
    def build_call_chains(
        self, 
        endpoint: Optional[Endpoint] = None,
        max_depth: int = 10
    ) -> List[CallChain]:
        """
        호출 체인 생성 (DFS 알고리즘)
        
        Args:
            endpoint: 시작 엔드포인트 (None이면 모든 엔드포인트에서 시작)
            max_depth: 최대 탐색 깊이
            
        Returns:
            List[CallChain]: 호출 체인 목록
        """
        if self.call_graph is None:
            self.logger.error("Call Graph가 생성되지 않았습니다. build_call_graph()를 먼저 호출하세요.")
            return []
        
        chains = []
        
        # 시작점 결정
        if endpoint:
            start_nodes = [endpoint.method_signature]
        else:
            # 모든 엔드포인트에서 시작
            start_nodes = [ep.method_signature for ep in self.endpoints]
        
        # 각 시작점에서 DFS 수행
        for start_node in start_nodes:
            if start_node not in self.call_graph:
                continue
            
            visited_paths: Set[Tuple[str, ...]] = set()
            current_path: List[str] = []
            
            def dfs(node: str, depth: int):
                """DFS 재귀 함수"""
                # 최대 깊이 확인
                if depth > max_depth:
                    return
                
                # 순환 참조 확인
                if node in current_path:
                    # 순환 참조 발견
                    cycle_start = current_path.index(node)
                    cycle = current_path[cycle_start:] + [node]
                    chain = CallChain(
                        chain=current_path + [node],
                        layers=[self._get_layer(m) for m in current_path + [node]],
                        is_circular=True
                    )
                    chains.append(chain)
                    return
                
                # 현재 경로에 추가
                current_path.append(node)
                path_tuple = tuple(current_path)
                
                # 이미 방문한 경로인지 확인
                if path_tuple in visited_paths:
                    current_path.pop()
                    return
                
                visited_paths.add(path_tuple)
                
                # 리프 노드 확인 (더 이상 호출하는 메서드가 없음)
                if node not in self.call_graph or len(list(self.call_graph.successors(node))) == 0:
                    # 호출 체인 완성
                    chain = CallChain(
                        chain=current_path.copy(),
                        layers=[self._get_layer(m) for m in current_path],
                        is_circular=False
                    )
                    chains.append(chain)
                else:
                    # 후속 노드 탐색
                    for successor in self.call_graph.successors(node):
                        dfs(successor, depth + 1)
                
                # 백트래킹
                current_path.pop()
            
            # DFS 시작
            dfs(start_node, 0)
        
        return chains
    
    def _get_layer(self, method_signature: str) -> str:
        """
        메서드의 레이어 정보 조회
        
        Args:
            method_signature: 메서드 시그니처
            
        Returns:
            str: 레이어명
        """
        if method_signature in self.method_metadata:
            return self.method_metadata[method_signature].get("layer", "Unknown")
        elif self.call_graph and method_signature in self.call_graph:
            return self.call_graph.nodes[method_signature].get("layer", "Unknown")
        return "Unknown"
    
    def get_classes_for_file(self, file_path: Path) -> List[ClassInfo]:
        """
        특정 파일의 파싱된 클래스 정보 반환
        
        Args:
            file_path: 파일 경로
            
        Returns:
            List[ClassInfo]: 클래스 정보 리스트 (파싱되지 않았으면 빈 리스트)
        """
        file_path_str = str(file_path)
        return self.file_to_classes_map.get(file_path_str, [])
    
    def get_all_parsed_classes(self) -> Dict[str, List[ClassInfo]]:
        """
        모든 파싱된 파일의 클래스 정보 반환
        
        Returns:
            Dict[str, List[ClassInfo]]: 파일 경로 -> 클래스 정보 리스트 매핑
        """
        return self.file_to_classes_map.copy()
    
    def get_class_by_name(self, class_name: str) -> Optional[ClassInfo]:
        """
        클래스명으로 클래스 정보 조회
        
        Args:
            class_name: 클래스명
            
        Returns:
            Optional[ClassInfo]: 클래스 정보 (없으면 None)
        """
        return self.class_info_map.get(class_name)
    
    def detect_circular_references(self) -> List[List[str]]:
        """
        순환 참조 감지
        
        Returns:
            List[List[str]]: 순환 참조 경로 목록
        """
        if self.call_graph is None:
            return []
        
        # networkx의 강한 연결 요소(Strongly Connected Components) 사용
        cycles = []
        try:
            sccs = list(nx.strongly_connected_components(self.call_graph))
            for scc in sccs:
                if len(scc) > 1:
                    # 순환 참조가 있는 컴포넌트
                    subgraph = self.call_graph.subgraph(scc)
                    # 간단한 순환 경로 찾기
                    for node in scc:
                        try:
                            cycle = nx.find_cycle(subgraph, source=node)
                            if cycle:
                                cycle_path = [edge[0] for edge in cycle] + [cycle[0][1]]
                                cycles.append(cycle_path)
                                break
                        except nx.NetworkXNoCycle:
                            continue
        except Exception as e:
            self.logger.warning(f"순환 참조 감지 중 오류: {e}")
        
        return cycles
    
    def get_call_relations(self) -> List[CallRelation]:
        """
        Call Graph에서 CallRelation 목록 추출
        
        Returns:
            List[CallRelation]: 호출 관계 목록
        """
        if self.call_graph is None:
            return []
        
        relations = []
        for caller, callee in self.call_graph.edges():
            caller_metadata = self.method_metadata.get(caller, {})
            callee_metadata = self.method_metadata.get(callee, {})
            
            relation = CallRelation(
                caller=caller,
                callee=callee,
                caller_file=caller_metadata.get("file_path", ""),
                callee_file=callee_metadata.get("file_path", "")
            )
            relations.append(relation)
        
        return relations
    
    def save_graph(self, file_path: Path) -> bool:
        """
        Call Graph를 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        if self.call_graph is None:
            return False
        
        try:
            import pickle
            # pickle을 사용하여 그래프 저장
            with open(file_path, 'wb') as f:
                pickle.dump(self.call_graph, f)
            return True
        except Exception as e:
            self.logger.error(f"그래프 저장 실패: {e}")
            return False
    
    def load_graph(self, file_path: Path) -> bool:
        """
        파일에서 Call Graph 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            import pickle
            # pickle을 사용하여 그래프 로드
            with open(file_path, 'rb') as f:
                self.call_graph = pickle.load(f)
            return True
        except Exception as e:
            self.logger.error(f"그래프 로드 실패: {e}")
            return False
    
    def print_call_tree(
        self, 
        endpoint: Optional[Endpoint] = None,
        max_depth: int = 10,
        show_layers: bool = True
    ) -> None:
        """
        엔드포인트부터 시작하는 Call Tree를 터미널에 출력
        
        Args:
            endpoint: 시작 엔드포인트 (Endpoint 객체 또는 None이면 모든 엔드포인트에서 시작)
            max_depth: 최대 탐색 깊이
            show_layers: 레이어 정보 표시 여부
        """
        if self.call_graph is None:
            self.logger.error("Call Graph가 생성되지 않았습니다. build_call_graph()를 먼저 호출하세요.")
            return
        
        # 시작점 결정
        if endpoint:
            # Endpoint 객체인 경우 method_signature 사용, 문자열인 경우 그대로 사용
            if isinstance(endpoint, Endpoint):
                start_nodes = [endpoint.method_signature]
            elif isinstance(endpoint, str):
                start_nodes = [endpoint]
            else:
                self.logger.error(f"잘못된 endpoint 타입: {type(endpoint)}")
                return
        else:
            # 모든 엔드포인트에서 시작
            start_nodes = [ep.method_signature for ep in self.endpoints]
        
        if not start_nodes:
            print("출력할 엔드포인트가 없습니다.")
            return
        
        # 각 시작점에서 Call Tree 출력
        for start_node in start_nodes:
            if start_node not in self.call_graph:
                print(f"엔드포인트 '{start_node}'가 Call Graph에 없습니다.")
                continue
            
            # 엔드포인트 정보 출력
            endpoint_info = next((ep for ep in self.endpoints if ep.method_signature == start_node), None)
            if endpoint_info:
                print(f"\n{'='*60}")
                print(f"Endpoint: {endpoint_info.http_method} {endpoint_info.path}")
                print(f"Method: {endpoint_info.method_signature}")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"Method: {start_node}")
                print(f"{'='*60}")
            
            # Call Tree 출력
            visited = set()
            
            def print_node(node: str, prefix: str = "", is_last: bool = True, depth: int = 0):
                """
                재귀적으로 노드를 출력하는 내부 함수
                
                Args:
                    node: 현재 노드
                    prefix: 접두사 (들여쓰기용)
                    is_last: 마지막 자식 노드 여부
                    depth: 현재 깊이
                """
                # 최대 깊이 확인
                if depth > max_depth:
                    return
                
                # 순환 참조 확인
                if node in visited:
                    layer_info = f" [{self._get_layer(node)}]" if show_layers else ""
                    print(f"{prefix}└─ {node}{layer_info} (recursive/circular)")
                    return
                
                visited.add(node)
                
                # 노드 출력
                layer_info = f" [{self._get_layer(node)}]" if show_layers else ""
                connector = "└─ " if is_last else "├─ "
                print(f"{prefix}{connector}{node}{layer_info}")
                
                # 자식 노드 가져오기
                if node in self.call_graph:
                    successors = list(self.call_graph.successors(node))
                    if successors:
                        # 다음 레벨 접두사 계산
                        extension = "   " if is_last else "│  "
                        for i, successor in enumerate(successors):
                            is_last_child = (i == len(successors) - 1)
                            new_prefix = prefix + extension
                            print_node(successor, new_prefix, is_last_child, depth + 1)
                
                visited.remove(node)
            
            # 루트 노드부터 시작
            print_node(start_node, "", True, 0)
            print()
    
    def get_call_tree(
        self,
        endpoint: Endpoint,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        엔드포인트부터 시작하는 Call Tree를 딕셔너리 형태로 반환
        
        Args:
            endpoint: 시작 엔드포인트
            max_depth: 최대 탐색 깊이
            
        Returns:
            Dict[str, Any]: Call Tree 구조 (JSON 직렬화 가능)
        """
        if self.call_graph is None:
            self.logger.error("Call Graph가 생성되지 않았습니다. build_call_graph()를 먼저 호출하세요.")
            return {}
        
        start_node = endpoint.method_signature
        if start_node not in self.call_graph:
            self.logger.warning(f"엔드포인트 '{start_node}'가 Call Graph에 없습니다.")
            return {}
        
        visited_in_path = set()
        
        def build_tree_node(node: str, depth: int) -> Dict[str, Any]:
            """
            재귀적으로 트리 노드를 구성하는 내부 함수
            
            Args:
                node: 현재 노드
                depth: 현재 깊이
                
            Returns:
                Dict[str, Any]: 노드 정보 딕셔너리
            """
            # 최대 깊이 확인
            if depth > max_depth:
                return None
            
            # 순환 참조 확인
            is_circular = node in visited_in_path
            if is_circular:
                return {
                    "method_signature": node,
                    "layer": self._get_layer(node),
                    "is_circular": True,
                    "children": []
                }
            
            visited_in_path.add(node)
            
            # 노드 정보 구성
            node_info: Dict[str, Any] = {
                "method_signature": node,
                "layer": self._get_layer(node),
                "is_circular": False,
                "children": []
            }
            
            # 메서드 메타데이터 추가
            if node in self.method_metadata:
                metadata = self.method_metadata[node]
                node_info["class_name"] = metadata.get("class_name", "")
                node_info["file_path"] = metadata.get("file_path", "")
            
            # 자식 노드 가져오기
            if node in self.call_graph:
                successors = list(self.call_graph.successors(node))
                for successor in successors:
                    child_node = build_tree_node(successor, depth + 1)
                    if child_node is not None:
                        node_info["children"].append(child_node)
            
            visited_in_path.remove(node)
            
            return node_info
        
        # 루트 노드부터 시작
        tree = build_tree_node(start_node, 0)
        
        # 엔드포인트 정보 추가
        if tree:
            tree["endpoint"] = {
                "path": endpoint.path,
                "http_method": endpoint.http_method,
                "method_signature": endpoint.method_signature,
                "class_name": endpoint.class_name,
                "method_name": endpoint.method_name,
                "file_path": endpoint.file_path
            }
        
        return tree if tree else {}
    
    def get_all_call_trees(
        self,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        모든 엔드포인트의 Call Tree를 딕셔너리 형태로 반환
        
        Args:
            max_depth: 최대 탐색 깊이
            
        Returns:
            List[Dict[str, Any]]: 각 엔드포인트의 Call Tree 리스트
        """
        call_trees = []
        for endpoint in self.endpoints:
            tree = self.get_call_tree(endpoint, max_depth)
            if tree:
                call_trees.append(tree)
        return call_trees
    
    def print_all_call_trees(
        self,
        max_depth: int = 10,
        show_layers: bool = True
    ) -> None:
        """
        모든 엔드포인트의 Call Tree를 터미널에 출력
        
        Args:
            max_depth: 최대 탐색 깊이
            show_layers: 레이어 정보 표시 여부
        """
        if not self.endpoints:
            print("엔드포인트가 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print("CALL TREES (모든 엔드포인트)")
        print(f"{'='*60}\n")
        
        for endpoint in self.endpoints:
            self.print_call_tree(endpoint, max_depth, show_layers)

