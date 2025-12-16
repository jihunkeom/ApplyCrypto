"""
Configuration Manager 모듈

JSON 설정 파일을 로드하고 검증하는 Configuration Manager를 구현합니다.
프로젝트 경로, 파일 타입, SQL Wrapping 타입, 암호화 대상 테이블/칼럼 정보를 파싱하고 스키마 검증을 수행합니다.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
from jsonschema import ValidationError, validate


class ConfigurationError(Exception):
    """설정 관련 에러를 나타내는 사용자 정의 예외 클래스"""

    pass


class ConfigurationManager:
    """
    JSON 설정 파일을 로드하고 검증하는 Configuration Manager 클래스

    주요 기능:
    1. JSON 설정 파일 로드 및 파싱
    2. 스키마 검증을 통한 필수 필드 검증
    3. 타입 안전한 설정값 접근 인터페이스 제공
    4. 에러 처리 및 기본값 설정
    """

    # JSON 스키마 정의: 필수 필드와 타입 검증
    CONFIG_SCHEMA = {
        "type": "object",
        "required": [
            "target_project",
            "source_file_types",
            "sql_wrapping_type",
            "access_tables",
        ],
        "properties": {
            "target_project": {
                "type": "string",
                "description": "대상 프로젝트 루트 경로",
            },
            "source_file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "수집할 소스 파일 확장자 목록",
            },
            "sql_wrapping_type": {
                "type": "string",
                "enum": ["mybatis", "jdbc", "jpa"],
                "description": "SQL Wrapping 타입",
            },
            "access_tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["table_name", "columns"],
                    "properties": {
                        "table_name": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "new_column": {"type": "boolean"},
                                        },
                                        "required": ["name"],
                                    },
                                ]
                            },
                        },
                    },
                },
                "description": "암호화 대상 테이블 및 칼럼 정보",
            },
            "llm_provider": {
                "type": "string",
                "enum": ["watsonx_ai", "claude_ai", "openai", "mock"],
                "description": "사용할 LLM 프로바이더 (기본값: watsonx_ai)",
            },
            "exclude_dirs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "제외할 디렉터리 이름 목록 (예: ['test', 'generated'])",
            },
            "exclude_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "제외할 파일 패턴 목록 (glob 패턴 지원, 예: ['*Test.java', '*_test.java'])",
            },
        },
    }

    def __init__(self, config_file_path: str):
        """
        ConfigurationManager 초기화

        Args:
            config_file_path: 설정 파일 경로 (문자열)

        Raises:
            ConfigurationError: 파일이 없거나, JSON 파싱 실패, 또는 스키마 검증 실패 시
        """
        self._config_file_path = Path(config_file_path)
        self._config_data: Optional[Dict[str, Any]] = None

        # 설정 파일 로드 및 검증 수행
        self._load_config()
        self._validate_schema()

    def _load_config(self) -> None:
        """
        JSON 설정 파일을 로드하고 파싱합니다.

        Raises:
            ConfigurationError: 파일이 없거나 JSON 파싱 실패 시
        """
        # 파일 존재 여부 확인
        if not self._config_file_path.exists():
            raise ConfigurationError(
                f"설정 파일을 찾을 수 없습니다: {self._config_file_path}"
            )

        # 파일 읽기 및 JSON 파싱
        try:
            with open(self._config_file_path, "r", encoding="utf-8") as f:
                self._config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"설정 파일의 JSON 형식이 올바르지 않습니다: {e}")
        except IOError as e:
            raise ConfigurationError(f"설정 파일을 읽는 중 오류가 발생했습니다: {e}")

    def _validate_schema(self) -> None:
        """
        jsonschema를 사용하여 설정 데이터의 스키마를 검증합니다.

        필수 필드:
        - target_project: 대상 프로젝트 경로
        - source_file_types: 소스 파일 타입 목록
        - sql_wrapping_type: SQL Wrapping 타입
        - access_tables: 접근 테이블 목록

        Raises:
            ConfigurationError: 스키마 검증 실패 시
        """
        if self._config_data is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")

        try:
            validate(instance=self._config_data, schema=self.CONFIG_SCHEMA)
        except ValidationError as e:
            error_path = ".".join(str(p) for p in e.path)
            raise ConfigurationError(
                f"설정 파일 스키마 검증 실패: {error_path} - {e.message}"
            )

    @property
    def target_project(self) -> Path:
        """
        대상 프로젝트 경로를 반환합니다.

        Returns:
            Path: 대상 프로젝트 루트 경로
        """
        if self._config_data is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")

        return Path(self._config_data["target_project"])

    @property
    def project_path(self) -> Path:
        """
        프로젝트 경로를 반환합니다. (하위 호환성을 위해 유지)

        Returns:
            Path: 대상 프로젝트 루트 경로
        """
        return self.target_project

    @property
    def source_file_types(self) -> List[str]:
        """
        수집할 소스 파일 확장자 목록을 반환합니다.

        Returns:
            List[str]: 파일 확장자 목록 (예: [".java", ".xml"])
        """
        if self._config_data is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")

        return self._config_data.get("source_file_types", [])

    @property
    def sql_wrapping_type(self) -> str:
        """
        SQL Wrapping 타입을 반환합니다.

        Returns:
            str: SQL Wrapping 타입 ("mybatis", "jdbc", "jpa" 중 하나)
        """
        if self._config_data is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")

        return self._config_data.get("sql_wrapping_type", "mybatis")

    @property
    def access_tables(self) -> List[Dict[str, Any]]:
        """
        암호화 대상 테이블 및 칼럼 정보를 반환합니다.

        Returns:
            List[Dict[str, Any]]: 테이블 정보 목록
                각 항목은 {"table_name": str, "columns": List[str]} 형태
        """
        if self._config_data is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")

        return self._config_data.get("access_tables", [])

    def get_table_names(self) -> List[str]:
        """
        암호화 대상 테이블명 목록을 반환합니다.

        Returns:
            List[str]: 테이블명 목록
        """
        return [table["table_name"] for table in self.access_tables]

    def get_columns_for_table(self, table_name: str) -> List[str]:
        """
        특정 테이블의 암호화 대상 칼럼 목록을 반환합니다.

        Args:
            table_name: 테이블명

        Returns:
            List[str]: 칼럼명 목록 (테이블이 없으면 빈 리스트)
        """
        for table in self.access_tables:
            if table["table_name"] == table_name:
                return table.get("columns", [])
        return []

    @property
    def exclude_dirs(self) -> List[str]:
        """
        제외할 디렉터리 이름 목록을 반환합니다.

        Returns:
            List[str]: 디렉터리 이름 목록 (없으면 빈 리스트)
        """
        if self._config_data is None:
            return []

        return self._config_data.get("exclude_dirs", [])

    @property
    def exclude_files(self) -> List[str]:
        """
        제외할 파일 패턴 목록을 반환합니다.

        Returns:
            List[str]: 파일 패턴 목록 (glob 패턴 지원, 없으면 빈 리스트)
        """
        if self._config_data is None:
            return []

        return self._config_data.get("exclude_files", [])

    @property
    def llm_provider(self) -> str:
        """
        LLM 프로바이더 이름을 반환합니다.

        Returns:
            str: LLM 프로바이더 이름 ("watsonx_ai", "watsonx", "openai" 중 하나, 기본값: "watsonx_ai")
        """
        if self._config_data is None:
            return "watsonx_ai"

        return self._config_data.get("llm_provider", "watsonx_ai")

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값을 가져옵니다.

        Args:
            key: 설정 키
            default: 기본값 (키가 없을 경우 반환)

        Returns:
            Any: 설정값 또는 기본값
        """
        if self._config_data is None:
            return default

        return self._config_data.get(key, default)
