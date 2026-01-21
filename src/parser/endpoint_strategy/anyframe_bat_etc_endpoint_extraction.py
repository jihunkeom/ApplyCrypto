import logging
from typing import List, Optional
from models.method import Method
from parser.java_ast_parser import ClassInfo
from models.endpoint import Endpoint

from .endpoint_extraction_strategy import EndpointExtractionStrategy

logger = logging.getLogger(__name__)


class AnyframeBatEtcEndpointExtraction(EndpointExtractionStrategy):
    """
    Spring Batch Quarts 프레임워크 엔드포인트 추출 전략

    Spring Batch Quarts 구조 패턴을 사용하여 엔드포인트를 추출합니다.
    """

    LAYER_PATTERNS = {
        "Job": ["Job"],
        "Service": ["Service", "ServiceImpl", "Stub"],
        "Repository": [
            "Repository",
            "JpaRepository",
            "CrudRepository",
            "DAO",
            "Dao",
            "DAOImpl",
            "DaoImpl",
            "JdbcDao",
            "JdbcTemplateDao",
        ],
        "Mapper": ["Mapper", "MyBatisMapper", "SqlMapper"],
        "Entity": ["Entity", "Domain", "Model", "POJO"],
        "VO": ["VO", "DTO"],
    }

    def extract_endpoints_from_classes(self, classes: List[ClassInfo]):
        """
        클래스 목록에서 모든 엔드포인트를 추출합니다.

        Args:
            classes: 클래스 정보 목록

        Returns:
            List[Endpoint]: 추출된 엔드포인트 목록
        """
        endpoints = []

        for cls in classes:
            # 메서드 레벨 엔드포인트 식별
            for method in cls.methods:
                endpoint = self.extract_endpoint(cls, method, "")
                if endpoint:
                    endpoints.append(endpoint)
        return endpoints

    def extract_endpoint(
        self,
        cls: ClassInfo,
        method: Method,
        class_path: str,
    ) -> Optional[Endpoint]:
        """
        메서드에서 엔드포인트 정보 추출 (Spring Batch Quartz 기반)

        Args:
            cls: 클래스 정보
            method: 메서드 정보
            class_path: 클래스 레벨 경로 (사용되지 않음)

        Returns:
            Optional[Endpoint]: 엔드포인트 정보
        """
        # Spring Batch Quartz의 Job 파일은 execute 메서드 포함. BAT 파일도 execute 메서드를 포함
        if (not cls.name.endswith("Job") or method.name != "execute") and (
            not cls.name.endswith("BAT") or method.name != "execute"
        ):
            return None

        # Endpoint 객체 생성
        method_signature = f"{cls.name}.{method.name}"
        return Endpoint(
            path=cls.file_path,
            http_method=None,
            method_signature=method_signature,
            class_name=cls.name,
            method_name=method.name,
            file_path=cls.file_path,
        )

    def classify_layer(self, cls: ClassInfo, method: Method) -> str:
        """
        클래스와 메서드의 레이어 분류

        Args:
            cls: 클래스 정보
            method: 메서드 정보

        Returns:
            str: 레이어명
        """
        # 어노테이션 기반 분류 (우선순위 높음)
        all_annotations = cls.annotations + method.annotations
        annotation_lower = [ann.lower() for ann in all_annotations]

        # Job 레이어
        if any("job" in ann for ann in annotation_lower):
            return "Job"

        # Service 레이어
        if any("service" in ann for ann in annotation_lower):
            return "Service"

        # Mybatis Mapper 레이어
        if any("mapper" in ann for ann in annotation_lower):
            return "Mapper"

        # JPA Repository 레이어
        if any("repository" in ann for ann in annotation_lower):
            return "Repository"

        # JPA Entity 레이어
        if any("entity" in ann or "table" in ann for ann in annotation_lower):
            return "Entity"

        # VO 레이어
        if any("vo" in ann for ann in annotation_lower):
            return "VO"

        # 클래스명 패턴 기반 분류
        class_name = cls.name
        for layer, patterns in self.LAYER_PATTERNS.items():
            for pattern in patterns:
                # 패턴이 클래스명에 포함되어 있는지 확인
                if pattern in class_name:
                    return layer

        # 인터페이스 기반 분류 (MyBatis Mapper 인터페이스 감지)
        if cls.interfaces:
            for interface in cls.interfaces:
                interface_lower = interface.lower()
                # MyBatis Mapper 인터페이스 패턴
                if "mapper" in interface_lower or "sqlmapper" in interface_lower:
                    return "Mapper"
                # JPA Repository 인터페이스 패턴
                if (
                    "repository" in interface_lower
                    or "jparepository" in interface_lower
                ):
                    return "Repository"
                # Spring Repository 인터페이스 패턴
                if (
                    "crudrepository" in interface_lower
                    or "pagerepository" in interface_lower
                ):
                    return "Repository"

        # 패키지 기반 분류
        package = cls.package.lower() if cls.package else ""
        if "job" in package:
            return "Job"
        elif "vo" in package:
            return "VO"
        elif "service" in package or "business" not in package:
            return "Service"
        elif "mapper" in package or "mybatis" not in package:
            return "Mapper"
        elif "repository" in package or "jpa" in package:
            return "Repository"
        elif "dao" in package or "data" in package:
            return "DAO"
        elif (
            "entity" in package
            or "domain" in package
            or "model" in package
            or "beans" in package
        ):
            return "Entity"

        # 필드 기반 추론 (JPA EntityManager, MyBatis SqlSession 등)
        for class_field_info in cls.fields:
            field_type = class_field_info.get("type", "").lower()
            if "entitymanager" in field_type or "entitymanagerfactory" in field_type:
                return "Repository"  # JPA Repository로 추론
            elif "sqlsession" in field_type or "sqlsessiontemplate" in field_type:
                return "Mapper"  # MyBatis Mapper 로 추론
            elif "jdbctemplate" in field_type or "datasource" in field_type:
                return "DAO"  # JDBC DAO로 추론

        return "Unknown"
