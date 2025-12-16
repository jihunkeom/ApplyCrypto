"""
커스텀 JSON 디코더 모듈

JSON 문자열을 읽을 때 datetime, Path 등 특수 타입으로 복원하는 디코더입니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class CustomJSONDecoder:
    """
    커스텀 JSON 디코더 클래스

    JSON 문자열을 읽을 때 특수 타입으로 복원합니다:
    - ISO 8601 형식 문자열: datetime 객체로 변환
    - 경로 문자열: Path 객체로 변환
    """

    @staticmethod
    def decode_datetime(value: str) -> datetime:
        """
        ISO 8601 형식 문자열을 datetime 객체로 변환

        Args:
            value: ISO 8601 형식 문자열

        Returns:
            datetime 객체
        """
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # 다른 형식 시도
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"날짜 형식을 파싱할 수 없습니다: {value}")

    @staticmethod
    def decode_path(value: str) -> Path:
        """
        문자열을 Path 객체로 변환

        Args:
            value: 경로 문자열

        Returns:
            Path 객체
        """
        return Path(value)

    @staticmethod
    def decode_dict(data: Dict[str, Any], model_class: type = None) -> Dict[str, Any]:
        """
        딕셔너리를 재귀적으로 디코딩

        Args:
            data: 디코딩할 딕셔너리
            model_class: 복원할 모델 클래스 (from_dict 메서드가 있는 경우)

        Returns:
            디코딩된 딕셔너리
        """
        if model_class and hasattr(model_class, "from_dict"):
            return model_class.from_dict(data)

        result = {}
        for key, value in data.items():
            result[key] = CustomJSONDecoder.decode_value(value)
        return result

    @staticmethod
    def decode_value(value: Any) -> Any:
        """
        값을 재귀적으로 디코딩

        Args:
            value: 디코딩할 값

        Returns:
            디코딩된 값
        """
        if isinstance(value, dict):
            # datetime 필드 확인 (일반적인 필드명 패턴)
            if (
                "timestamp" in value
                or "modified_time" in value
                or "created_time" in value
            ):
                for time_key in ["timestamp", "modified_time", "created_time"]:
                    if time_key in value and isinstance(value[time_key], str):
                        try:
                            value[time_key] = CustomJSONDecoder.decode_datetime(
                                value[time_key]
                            )
                        except ValueError:
                            pass

            # path 필드 확인
            if "path" in value or "file_path" in value or "relative_path" in value:
                for path_key in [
                    "path",
                    "file_path",
                    "relative_path",
                    "caller_file",
                    "callee_file",
                ]:
                    if path_key in value and isinstance(value[path_key], str):
                        # 경로처럼 보이는 문자열인지 확인
                        if "/" in value[path_key] or "\\" in value[path_key]:
                            value[path_key] = CustomJSONDecoder.decode_path(
                                value[path_key]
                            )

            # 재귀적으로 딕셔너리 처리
            return {k: CustomJSONDecoder.decode_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [CustomJSONDecoder.decode_value(item) for item in value]

        elif isinstance(value, str):
            # ISO 8601 형식 날짜 문자열인지 확인
            if len(value) >= 19 and ("T" in value or "-" in value):
                try:
                    return CustomJSONDecoder.decode_datetime(value)
                except ValueError:
                    pass

        return value
