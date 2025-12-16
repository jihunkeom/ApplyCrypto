"""
Method 데이터 모델

Java 메서드 정보를 저장하는 데이터 모델입니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Parameter:
    """
    메서드 파라미터 정보

    Attributes:
        name: 파라미터 이름
        type: 파라미터 타입
        is_varargs: 가변 인자 여부
    """

    name: str
    type: str
    is_varargs: bool = False


@dataclass
class LocalVariable:
    """
    메서드 내부 지역 변수 정보

    Attributes:
        name: 변수 이름
        type: 변수 타입
    """

    name: str
    type: str


@dataclass
class Method:
    """
    Java 메서드 정보를 저장하는 데이터 모델

    Attributes:
        name: 메서드명
        return_type: 반환 타입
        parameters: 파라미터 목록
        local_variables: 메서드 내부 지역 변수 목록
        access_modifier: 접근 제어자 (public, private, protected, package)
        class_name: 소속 클래스명
        file_path: 파일 경로
        is_static: 정적 메서드 여부
        is_abstract: 추상 메서드 여부
        annotations: 어노테이션 목록
        exceptions: 예외 선언 목록
    """

    name: str
    return_type: str
    parameters: List[Parameter]
    local_variables: List[LocalVariable] = field(default_factory=list)
    access_modifier: str = "package"
    class_name: Optional[str] = None
    file_path: str = ""
    is_static: bool = False
    is_abstract: bool = False
    is_final: bool = False
    annotations: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    method_calls: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": [
                {"name": p.name, "type": p.type, "is_varargs": p.is_varargs}
                for p in self.parameters
            ],
            "local_variables": [
                {"name": v.name, "type": v.type} for v in self.local_variables
            ],
            "access_modifier": self.access_modifier,
            "class_name": self.class_name,
            "file_path": self.file_path,
            "is_static": self.is_static,
            "is_abstract": self.is_abstract,
            "is_final": self.is_final,
            "annotations": self.annotations,
            "exceptions": self.exceptions,
            "method_calls": self.method_calls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Method":
        """딕셔너리로부터 Method 객체 생성"""
        return cls(
            name=data["name"],
            return_type=data["return_type"],
            parameters=[
                Parameter(
                    name=p["name"],
                    type=p["type"],
                    is_varargs=p.get("is_varargs", False),
                )
                for p in data.get("parameters", [])
            ],
            local_variables=[
                LocalVariable(name=v["name"], type=v["type"])
                for v in data.get("local_variables", [])
            ],
            access_modifier=data.get("access_modifier", "package"),
            class_name=data.get("class_name"),
            file_path=data.get("file_path", ""),
            is_static=data.get("is_static", False),
            is_abstract=data.get("is_abstract", False),
            is_final=data.get("is_final", False),
            annotations=data.get("annotations", []),
            exceptions=data.get("exceptions", []),
            method_calls=data.get("method_calls", []),
        )
