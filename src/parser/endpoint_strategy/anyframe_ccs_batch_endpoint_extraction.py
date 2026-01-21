"""
Anyframe CCS Batch Endpoint Extraction Strategy

CCS 배치 프로그램의 엔드포인트(진입점)를 추출하는 전략 클래스입니다.

특징:
    - 엔드포인트 조건: 클래스명이 *BAT로 끝나고 execute 메서드가 존재
    - HTTP endpoint가 아닌 배치 Job의 진입점으로 처리
    - 레이어 분류: BAT, BATVO
"""

import logging
from typing import List, Optional

from models.endpoint import Endpoint
from models.method import Method
from parser.java_ast_parser import ClassInfo

from .endpoint_extraction_strategy import EndpointExtractionStrategy

logger = logging.getLogger(__name__)


class AnyframeCCSBatchEndpointExtraction(EndpointExtractionStrategy):
    """
    CCS 배치 프로그램 엔드포인트 추출 전략

    배치 클래스(*BAT.java)의 execute 메서드를 엔드포인트로 추출합니다.
    """

    # 레이어 패턴 정의
    LAYER_PATTERNS = {
        "BAT": ["BAT"],
        "BATVO": ["BATVO", "BatVO"],
    }

    def extract_endpoints_from_classes(
        self, classes: List[ClassInfo]
    ) -> List[Endpoint]:
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

        logger.info(f"CCS 배치 엔드포인트 추출 완료: {len(endpoints)}개")
        return endpoints

    def extract_endpoint(
        self,
        cls: ClassInfo,
        method: Method,
        class_path: str,
    ) -> Optional[Endpoint]:
        """
        메서드에서 엔드포인트 정보 추출 (CCS 배치 기반)

        엔드포인트 조건:
            - 클래스명이 *BAT로 끝남
            - 메서드명이 execute

        Args:
            cls: 클래스 정보
            method: 메서드 정보
            class_path: 클래스 레벨 경로 (배치에서는 사용되지 않음)

        Returns:
            Optional[Endpoint]: 엔드포인트 정보
        """
        # CCS 배치의 BAT 파일은 execute 메서드를 포함
        if not cls.name.endswith("BAT") or method.name != "execute":
            return None

        # Endpoint 객체 생성
        method_signature = f"{cls.name}.{method.name}"
        return Endpoint(
            path=cls.file_path,  # 배치는 HTTP path가 없으므로 파일 경로 사용
            http_method=None,  # 배치는 HTTP 메서드가 없음
            method_signature=method_signature,
            class_name=cls.name,
            method_name=method.name,
            file_path=cls.file_path,
        )

    def extract_path_from_annotation(self, annotation: str) -> Optional[str]:
        """
        어노테이션에서 path 추출 (배치에서는 사용되지 않음)

        Args:
            annotation: 어노테이션 문자열

        Returns:
            Optional[str]: None (배치는 HTTP path 없음)
        """
        return None

    def extract_http_method_from_annotation(
        self, annotation: str
    ) -> Optional[str]:
        """
        어노테이션에서 HTTP 메서드 추출 (배치에서는 사용되지 않음)

        Args:
            annotation: 어노테이션 문자열

        Returns:
            Optional[str]: None (배치는 HTTP 메서드 없음)
        """
        return None

    def classify_layer(self, cls: ClassInfo, method: Method) -> str:
        """
        클래스와 메서드의 레이어 분류

        Args:
            cls: 클래스 정보
            method: 메서드 정보

        Returns:
            str: 레이어명 (BAT, BATVO, Unknown)
        """
        class_name = cls.name

        # 클래스명 패턴 기반 분류
        for layer, patterns in self.LAYER_PATTERNS.items():
            for pattern in patterns:
                # 패턴이 클래스명에 포함되어 있는지 확인
                if class_name.endswith(pattern):
                    return layer

        # 패키지 기반 분류
        package = cls.package.lower() if cls.package else ""
        if "bat" in package:
            if "batvo" in package or "vo" in package:
                return "BATVO"
            return "BAT"

        return "Unknown"

    def get_class_level_path(self, cls: ClassInfo) -> str:
        """
        클래스 레벨 경로 추출 (배치에서는 빈 문자열 반환)

        Args:
            cls: 클래스 정보

        Returns:
            str: 빈 문자열 (배치는 HTTP path 없음)
        """
        return ""
