"""
Call Graph Builder

Java AST 정보를 기반으로 메서드 호출 관계를 추적하여 그래프 구조를 생성하고,
REST API 엔드포인트부터 DAO/Mapper까지 이어지는 호출 체인을 구성하는 모듈입니다.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

from models.call_relation import CallRelation
from models.endpoint import Endpoint
from models.method import Method
from persistence.cache_manager import CacheManager

from .java_ast_parser import ClassInfo, JavaASTParser
from .endpoint_strategy import EndpointExtractionStrategy


class CallGraphBuilder:
    """
    Call Graph Builder 클래스

    Java AST 정보를 기반으로 메서드 호출 관계 그래프를 생성하고,
    REST API 엔드포인트부터 시작하는 호출 체인을 구성합니다.
    """

    def __init__(
        self,
        java_parser: Optional[JavaASTParser] = None,
        cache_manager: Optional[CacheManager] = None,
        endpoint_strategy: Optional[EndpointExtractionStrategy] = None,
    ):
        """
        CallGraphBuilder 초기화

        Args:
            java_parser: Java AST 파서 (선택적)
            cache_manager: 캐시 매니저 (선택적)
            endpoint_strategy: 엔드포인트 추출 전략 (선택적, 없으면 기본값 사용)
        """
        if nx is None:
            raise ImportError(
                "networkx 라이브러리가 필요합니다. pip install networkx로 설치하세요."
            )

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

        # EndpointExtractionStrategy 설정
        self.endpoint_strategy = endpoint_strategy

        self.logger = logging.getLogger("applycrypto")

        # Call Graph (networkx DiGraph)
        self.call_graph: Optional[nx.DiGraph] = None

        # 메서드 메타데이터 (메서드 시그니처 -> 메서드 정보)
        self.method_metadata: Dict[str, Dict[str, Any]] = {}

        # 클래스 정보 (클래스명 -> ClassInfo)
        self.class_name_to_info: Dict[str, ClassInfo] = {}

        # 파일 경로 -> 클래스 정보 리스트 매핑 (파싱된 정보 재사용용)
        self.file_to_classes_map: Dict[str, List[ClassInfo]] = {}

        # 클래스 정보 맵 (클래스명 -> ClassInfo 리스트) - DBAccessAnalyzer에서 사용
        self.class_info_map: Dict[str, List[Dict[str, Any]]] = {}

        # 엔드포인트 목록
        self.endpoints: List[Endpoint] = []

    def get_class_info_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        클래스 정보 맵 반환 (DBAccessAnalyzer에서 사용하는 형태)

        Returns:
            Dict[str, List[Dict[str, Any]]]: 클래스명 -> ClassInfo 리스트 매핑
        """
        return self.class_info_map

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
        self.class_name_to_info = {}
        self.file_to_classes_map = {}
        self.class_info_map = defaultdict(list)
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
                self.class_name_to_info[cls.name] = cls

        # class_info_map 생성 (DBAccessAnalyzer에서 사용할 형태)
        self.class_info_map = self._build_class_info_map(all_classes)

        # 클래스별 필드 정보 수집 (필드명 -> 타입 매핑)
        class_field_map: Dict[str, Dict[str, str]] = {}  # 클래스명 -> {필드명: 타입}
        for cls in all_classes:
            field_map = {}
            for class_field_info in cls.fields:
                field_name = class_field_info.get("name", "")
                field_type = class_field_info.get("type", "")
                if field_name and field_type:
                    # 제네릭 타입 처리 (예: List<User> -> List)
                    if "<" in field_type:
                        field_type = field_type.split("<")[0]
                    field_map[field_name] = field_type
            class_field_map[cls.name] = field_map

        # 메서드 호출 관계 추출
        call_relations = []
        for cls in all_classes:
            for method in cls.methods:
                method_signature = f"{cls.name}.{method.name}"

                # 메서드 메타데이터 저장
                # Strategy가 있으면 Strategy의 classify_layer 사용, 없으면 기본값
                if self.endpoint_strategy:
                    layer = self.endpoint_strategy.classify_layer(cls, method)
                else:
                    layer = "Unknown"  # 기본값

                self.method_metadata[method_signature] = {
                    "class_name": cls.name,
                    "method": method,
                    "file_path": cls.file_path,
                    "package": cls.package,
                    "annotations": method.annotations,
                    "layer": layer,
                    "class_info": cls,  # _get_layer에서 사용하기 위해 추가
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
                        if "<" in param_type:
                            param_type = param_type.split("<")[0]
                        method_variable_map[param_name] = param_type

                # 메서드의 local_variables들의 type 처리
                for local_var in method.local_variables:
                    var_name = local_var.name
                    var_type = local_var.type
                    if var_name and var_type:
                        # 제네릭 타입 처리 (예: List<User> -> List)
                        if "<" in var_type:
                            var_type = var_type.split("<")[0]
                        method_variable_map[var_name] = var_type

                # 메서드 호출 관계 추출
                for call in method.method_calls:
                    callee_signature = None
                    callee_file = cls.file_path

                    # call 형식이 "object.method"인 경우 처리
                    if "." in call:
                        parts = call.split(".")
                        if len(parts) >= 2:
                            # object.method 형식
                            object_name = parts[0]  # 필드명 또는 변수명
                            callee_method = parts[-1]

                            # 필드 변수를 통한 호출인지 확인
                            if object_name in current_field_map:
                                # 필드 타입 찾기
                                field_type = current_field_map[object_name]

                                # 필드 타입이 다른 클래스인 경우 해당 클래스의 메서드로 매핑
                                if field_type in self.class_name_to_info:
                                    callee_signature = f"{field_type}.{callee_method}"
                                    callee_cls = self.class_name_to_info[field_type]
                                    callee_file = callee_cls.file_path
                                else:
                                    # 필드 타입 클래스를 찾을 수 없는 경우 필드 타입으로 매핑 시도
                                    callee_signature = f"{field_type}.{callee_method}"
                            elif object_name in method_variable_map:
                                # 메서드 변수(파라미터 또는 리턴 타입)를 통한 호출인지 확인
                                variable_type = method_variable_map[object_name]

                                # 변수 타입이 다른 클래스인 경우 해당 클래스의 메서드로 매핑
                                if variable_type in self.class_name_to_info:
                                    callee_signature = (
                                        f"{variable_type}.{callee_method}"
                                    )
                                    callee_cls = self.class_name_to_info[variable_type]
                                    callee_file = callee_cls.file_path
                                else:
                                    # 변수 타입 클래스를 찾을 수 없는 경우 변수 타입으로 매핑 시도
                                    callee_signature = (
                                        f"{variable_type}.{callee_method}"
                                    )
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
                            callee_file=callee_file,
                        )
                        call_relations.append(relation)
                        # callee가 인터페이스인 경우, 이를 구현하는 클래스를 찾아 추가 CallRelation 생성
                        # callee_signature 형식: "ClassName.methodName"
                        if "." in callee_signature:
                            callee_class_name = callee_signature.split(".")[0]

                            # callee 클래스가 인터페이스인지 확인
                            if callee_class_name in self.class_name_to_info:
                                callee_class_info = self.class_name_to_info[
                                    callee_class_name
                                ]
                                if callee_class_info.is_interface_class:
                                    # 해당 인터페이스를 구현하는 클래스 찾기
                                    for impl_cls in all_classes:
                                        # 구현 클래스의 interfaces 목록에 callee 인터페이스가 있는지 확인
                                        # 인터페이스 이름이 단순 이름일 수도 있고, 패키지 포함 전체 이름일 수도 있음
                                        interface_found = False
                                        for interface_name in impl_cls.interfaces:
                                            # 단순 이름 비교
                                            if interface_name == callee_class_name:
                                                interface_found = True
                                                break
                                            # 패키지 포함 전체 이름 비교
                                            if "." in interface_name:
                                                simple_interface_name = (
                                                    interface_name.split(".")[-1]
                                                )
                                                if (
                                                    simple_interface_name
                                                    == callee_class_name
                                                ):
                                                    interface_found = True
                                                    break
                                            # callee_class_name이 패키지 포함 전체 이름인 경우
                                            if "." in callee_class_name:
                                                simple_callee_name = (
                                                    callee_class_name.split(".")[-1]
                                                )
                                                if (
                                                    interface_name == simple_callee_name
                                                    or interface_name
                                                    == callee_class_name
                                                ):
                                                    interface_found = True
                                                    break

                                        if interface_found:
                                            # 구현 클래스를 callee로 하는 추가 CallRelation 생성
                                            callee_method_name = callee_signature.split(
                                                "."
                                            )[-1]
                                            impl_callee_signature = (
                                                f"{impl_cls.name}.{callee_method_name}"
                                            )

                                            impl_relation = CallRelation(
                                                caller=method_signature,
                                                callee=impl_callee_signature,
                                                caller_file=cls.file_path,
                                                callee_file=impl_cls.file_path,
                                            )
                                            call_relations.append(impl_relation)

        # 그래프에 노드 및 간선 추가
        for relation in call_relations:
            # 노드 추가 (메타데이터 포함)
            if relation.caller not in self.call_graph:
                metadata = self.method_metadata.get(relation.caller, {})
                self.call_graph.add_node(
                    relation.caller,
                    class_name=metadata.get("class_name", ""),
                    file_path=metadata.get("file_path", ""),
                    layer=metadata.get("layer", "Unknown"),
                )

            if relation.callee not in self.call_graph:
                metadata = self.method_metadata.get(relation.callee, {})
                self.call_graph.add_node(
                    relation.callee,
                    class_name=metadata.get("class_name", ""),
                    file_path=metadata.get("file_path", ""),
                    layer=metadata.get("layer", "Unknown"),
                )

            # 간선 추가
            self.call_graph.add_edge(relation.caller, relation.callee)

        # 엔드포인트 식별
        if self.endpoint_strategy:
            self.endpoints = self.endpoint_strategy.extract_endpoints_from_classes(
                all_classes
            )
        else:
            # Strategy가 없으면 빈 리스트 (하위 호환성)
            self.endpoints = []
            self.logger.warning(
                "EndpointExtractionStrategy가 설정되지 않아 엔드포인트를 추출할 수 없습니다."
            )

        # 인터페이스 엔드포인트에 대한 구현 클래스의 call relation 추가
        for endpoint in self.endpoints:
            endpoint_class_name = endpoint.class_name
            endpoint_method_name = endpoint.method_name
            
            # 엔드포인트가 인터페이스 클래스인지 확인
            if endpoint_class_name in self.class_name_to_info:
                endpoint_class_info = self.class_name_to_info[endpoint_class_name]
                
                if endpoint_class_info.is_interface_class:
                    # 해당 인터페이스를 구현하는 클래스 찾기
                    for impl_cls in all_classes:
                        # 구현 클래스의 interfaces 목록에 엔드포인트 인터페이스가 있는지 확인
                        interface_found = False
                        for interface_name in impl_cls.interfaces:
                            # 단순 이름 비교
                            if interface_name == endpoint_class_name:
                                interface_found = True
                                break
                            # 패키지 포함 전체 이름 비교
                            if "." in interface_name:
                                simple_interface_name = interface_name.split(".")[-1]
                                if simple_interface_name == endpoint_class_name:
                                    interface_found = True
                                    break
                            # endpoint_class_name이 패키지 포함 전체 이름인 경우
                            if "." in endpoint_class_name:
                                simple_endpoint_name = endpoint_class_name.split(".")[-1]
                                if interface_name == simple_endpoint_name or interface_name == endpoint_class_name:
                                    interface_found = True
                                    break
                        
                        if interface_found:
                            # 구현 클래스의 메서드 시그니처 구성
                            impl_method_signature = f"{impl_cls.name}.{endpoint_method_name}"
                            
                            # 구현 클래스의 메서드가 실제로 존재하는지 확인
                            method_exists = False
                            for method in impl_cls.methods:
                                if method.name == endpoint_method_name:
                                    method_exists = True
                                    break
                            
                            if not method_exists:
                                continue
                            
                            # endpoint를 caller로, 구현 클래스 메서드를 callee로 하는 새로운 relation 추가
                            # endpoint 노드가 없으면 추가
                            if endpoint.method_signature not in self.call_graph:
                                # 엔드포인트 메서드의 메타데이터 가져오기 또는 생성
                                endpoint_metadata = self.method_metadata.get(
                                    endpoint.method_signature,
                                    {
                                        "class_name": endpoint.class_name,
                                        "file_path": endpoint.file_path,
                                        "layer": "Endpoint",
                                    }
                                )
                                self.call_graph.add_node(
                                    endpoint.method_signature,
                                    class_name=endpoint_metadata.get("class_name", endpoint.class_name),
                                    file_path=endpoint_metadata.get("file_path", endpoint.file_path),
                                    layer=endpoint_metadata.get("layer", "Endpoint"),
                                )
                            
                            # 구현 클래스 메서드 노드가 없으면 추가
                            if impl_method_signature not in self.call_graph:
                                impl_metadata = self.method_metadata.get(
                                    impl_method_signature,
                                    {
                                        "class_name": impl_cls.name,
                                        "file_path": impl_cls.file_path,
                                        "layer": "Unknown",
                                    }
                                )
                                self.call_graph.add_node(
                                    impl_method_signature,
                                    class_name=impl_metadata.get("class_name", impl_cls.name),
                                    file_path=impl_metadata.get("file_path", impl_cls.file_path),
                                    layer=impl_metadata.get("layer", "Unknown"),
                                )
                            
                            # 이미 존재하는 relation인지 확인 후 추가
                            if not self.call_graph.has_edge(endpoint.method_signature, impl_method_signature):
                                self.call_graph.add_edge(
                                    endpoint.method_signature,
                                    impl_method_signature
                                )
                                
                                self.logger.debug(
                                    f"인터페이스 엔드포인트 연결 추가: "
                                    f"{endpoint.method_signature} -> {impl_method_signature}"
                                )

        return self.call_graph

    def _build_class_info_map(
        self, all_classes: List[ClassInfo]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        클래스 정보를 딕셔너리 리스트 형태로 변환하여 class_info_map 생성

        Args:
            all_classes: 모든 클래스 정보 리스트

        Returns:
            Dict[str, List[Dict[str, Any]]]: 클래스명 -> ClassInfo 리스트 매핑
        """
        class_info_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for cls in all_classes:
            file_path_str = str(cls.file_path)
            full_class_name = f"{cls.package}.{cls.name}" if cls.package else cls.name

            # 클래스 정보 딕셔너리 생성
            class_info_dict = {
                "class_name": cls.name,
                "package": cls.package,
                "full_class_name": full_class_name,
                "file_path": file_path_str,
            }

            # 단순 클래스명으로 저장
            class_info_map[cls.name].append(class_info_dict)

            # 전체 클래스명으로도 저장 (패키지가 있는 경우)
            if cls.package:
                class_info_map[full_class_name].append(class_info_dict)

        return dict(class_info_map)

    def get_endpoints(self) -> List[Endpoint]:
        """
        식별된 엔드포인트 목록 반환

        Returns:
            List[Endpoint]: 엔드포인트 목록
        """
        return self.endpoints

    def get_endpoint_method_signatures(self) -> Set[str]:
        """
        모든 엔드포인트의 method signature를 Set으로 반환합니다.

        Returns:
            Set[str]: 엔드포인트 method signature 집합
        """
        return {ep.method_signature for ep in self.endpoints}

    def restore_from_call_trees(
        self,
        call_trees: List[Dict[str, Any]],
        endpoints: List[Endpoint],
        method_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        저장된 call_trees 데이터로부터 call_graph를 복원합니다.

        Args:
            call_trees: 저장된 call tree 리스트
            endpoints: 엔드포인트 리스트
            method_metadata: 메서드 메타데이터 (선택적, call_trees에서 추출 가능)
        """
        if nx is None:
            raise ImportError("networkx가 설치되어 있지 않습니다.")

        # 그래프 초기화
        self.call_graph = nx.DiGraph()
        self.endpoints = endpoints
        self.method_metadata = method_metadata or {}

        def extract_edges_from_tree(node: Dict[str, Any], parent: Optional[str] = None):
            """
            재귀적으로 트리에서 edges를 추출하는 내부 함수

            Args:
                node: 현재 노드
                parent: 부모 노드의 method_signature
            """
            method_sig = node.get("method_signature")
            if not method_sig:
                return

            # 노드 추가
            self.call_graph.add_node(method_sig)

            # 메타데이터 업데이트 (call_trees에 있는 정보 사용)
            if method_sig not in self.method_metadata:
                self.method_metadata[method_sig] = {}
            if "class_name" in node:
                self.method_metadata[method_sig]["class_name"] = node["class_name"]
            if "file_path" in node:
                self.method_metadata[method_sig]["file_path"] = node["file_path"]
            if "layer" in node:
                self.method_metadata[method_sig]["layer"] = node["layer"]
            if "arguments" in node:
                self.method_metadata[method_sig]["arguments"] = node["arguments"]

            # 부모에서 현재 노드로 edge 추가
            if parent:
                self.call_graph.add_edge(parent, method_sig)

            # 자식 노드 처리
            children = node.get("children", [])
            for child in children:
                extract_edges_from_tree(child, method_sig)

        # 각 call_tree에서 edges 추출
        for tree in call_trees:
            # endpoint 정보가 있으면 사용
            endpoint_info = tree.get("endpoint", {})
            if endpoint_info:
                method_sig = endpoint_info.get("method_signature")
                if method_sig:
                    self.call_graph.add_node(method_sig)
                    if method_sig not in self.method_metadata:
                        self.method_metadata[method_sig] = {}
                    if "class_name" in endpoint_info:
                        self.method_metadata[method_sig]["class_name"] = endpoint_info[
                            "class_name"
                        ]
                    if "file_path" in endpoint_info:
                        self.method_metadata[method_sig]["file_path"] = endpoint_info[
                            "file_path"
                        ]

            # 트리 구조에서 edges 추출
            if "method_signature" in tree:
                extract_edges_from_tree(tree)

        self.logger.info(
            f"Call Graph 복원 완료: {self.call_graph.number_of_nodes()}개 노드, "
            f"{self.call_graph.number_of_edges()}개 엣지"
        )

    def _get_layer(self, method_signature: str) -> str:
        """
        메서드의 레이어 정보 조회

        Args:
            method_signature: 메서드 시그니처

        Returns:
            str: 레이어명
        """
        if method_signature in self.method_metadata:
            metadata = self.method_metadata[method_signature]
            # 이미 저장된 layer가 있으면 사용
            if "layer" in metadata:
                return metadata.get("layer", "Unknown")
            # Strategy가 있고 class_info와 method가 있으면 재분류
            elif (
                self.endpoint_strategy
                and "class_info" in metadata
                and "method" in metadata
            ):
                cls = metadata["class_info"]
                method = metadata["method"]
                layer = self.endpoint_strategy.classify_layer(cls, method)
                # 메타데이터 업데이트
                metadata["layer"] = layer
                return layer
            return "Unknown"
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
        return self.class_name_to_info.get(class_name)

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
                callee_file=callee_metadata.get("file_path", ""),
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
            with open(file_path, "wb") as f:
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
            with open(file_path, "rb") as f:
                self.call_graph = pickle.load(f)
            return True
        except Exception as e:
            self.logger.error(f"그래프 로드 실패: {e}")
            return False

    def print_call_tree(
        self,
        endpoint: Optional[Endpoint] = None,
        max_depth: int = 10,
        show_layers: bool = True,
    ) -> None:
        """
        엔드포인트부터 시작하는 Call Tree를 터미널에 출력

        Args:
            endpoint: 시작 엔드포인트 (Endpoint 객체 또는 None이면 모든 엔드포인트에서 시작)
            max_depth: 최대 탐색 깊이
            show_layers: 레이어 정보 표시 여부
        """
        if self.call_graph is None:
            self.logger.error(
                "Call Graph가 생성되지 않았습니다. build_call_graph()를 먼저 호출하세요."
            )
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
            endpoint_info = next(
                (ep for ep in self.endpoints if ep.method_signature == start_node), None
            )
            if endpoint_info:
                print(f"\n{'=' * 60}")
                print(f"Endpoint: {endpoint_info.http_method} {endpoint_info.path}")
                print(f"Method: {endpoint_info.method_signature}")
                print(f"{'=' * 60}")
            else:
                print(f"\n{'=' * 60}")
                print(f"Method: {start_node}")
                print(f"{'=' * 60}")

            # Call Tree 출력
            visited = set()

            def print_node(
                node: str, prefix: str = "", is_last: bool = True, depth: int = 0
            ):
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
                            is_last_child = i == len(successors) - 1
                            new_prefix = prefix + extension
                            print_node(successor, new_prefix, is_last_child, depth + 1)

                visited.remove(node)

            # 루트 노드부터 시작
            print_node(start_node, "", True, 0)
            print()

    def get_call_tree(self, endpoint: Endpoint, max_depth: int = 10) -> Dict[str, Any]:
        """
        엔드포인트부터 시작하는 Call Tree를 딕셔너리 형태로 반환

        Args:
            endpoint: 시작 엔드포인트
            max_depth: 최대 탐색 깊이

        Returns:
            Dict[str, Any]: Call Tree 구조 (JSON 직렬화 가능)
        """
        if self.call_graph is None:
            self.logger.error(
                "Call Graph가 생성되지 않았습니다. build_call_graph()를 먼저 호출하세요."
            )
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
                    "children": [],
                }

            visited_in_path.add(node)

            # 노드 정보 구성
            node_info: Dict[str, Any] = {
                "method_signature": node,
                "layer": self._get_layer(node),
                "is_circular": False,
                "children": [],
            }

            # 메서드 메타데이터 추가
            if node in self.method_metadata:
                metadata = self.method_metadata[node]
                node_info["class_name"] = metadata.get("class_name", "")
                node_info["file_path"] = metadata.get("file_path", "")

                # 메서드 인자 정보 추가
                if "method" in metadata:
                    method = metadata["method"]
                    node_info["arguments"] = [
                        {"name": p.name, "type": p.type} for p in method.parameters
                    ]
                elif "arguments" in metadata:
                    node_info["arguments"] = metadata["arguments"]

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
                "file_path": endpoint.file_path,
            }

        return tree if tree else {}

    def get_all_call_trees(self, max_depth: int = 10) -> List[Dict[str, Any]]:
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
        self, max_depth: int = 10, show_layers: bool = True
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

        print(f"\n{'=' * 60}")
        print("CALL TREES (모든 엔드포인트)")
        print(f"{'=' * 60}\n")

        for endpoint in self.endpoints:
            self.print_call_tree(endpoint, max_depth, show_layers)
