"""
Configuration Manager 모듈

JSON 설정 파일을 로드하고 검증하는 Configuration Manager를 구현합니다.
프로젝트 경로, 파일 타입, SQL Wrapping 타입, 암호화 대상 테이블/칼럼 정보를 파싱하고 스키마 검증을 수행합니다.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError


class ConfigurationError(Exception):
    """설정 관련 에러를 나타내는 사용자 정의 예외 클래스"""

    pass


class ColumnDetail(BaseModel):
    name: str = Field(..., description="컬럼 이름")
    new_column: Optional[bool] = Field(None, description="새로운 컬럼 여부")
    column_type: Optional[Literal["dob", "ssn", "name", "sex"]] = Field(None, description="컬럼 타입 (dob: 생년월일, ssn: 주민번호, name: 이름, sex: 성별)")


class AccessTable(BaseModel):
    table_name: str = Field(..., description="테이블 이름")
    columns: List[Union[str, ColumnDetail]] = Field(..., description="컬럼 목록")


class Configuration(BaseModel):
    target_project: str = Field(..., description="대상 프로젝트 루트 경로")
    type_handler: bool = Field(False, description="Type Handler 사용 여부")
    type_handler_package: Optional[str] = Field(None, description="Type Handler 패키지 이름")
    type_handler_output_dir: Optional[str] = Field(None, description="Type Handler 출력 디렉터리")
    source_file_types: List[str] = Field(..., description="수집할 소스 파일 확장자 목록")
    sql_wrapping_type: Literal["mybatis", "jdbc", "jpa"] = Field(..., description="SQL Wrapping 타입")
    access_tables: List[AccessTable] = Field(..., description="암호화 대상 테이블 및 칼럼 정보")
    llm_provider: Literal["watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"] = Field("watsonx_ai", description="사용할 LLM 프로바이더")
    exclude_dirs: List[str] = Field(default_factory=list, description="제외할 디렉터리 이름 목록")
    exclude_files: List[str] = Field(default_factory=list, description="제외할 파일 패턴 목록")


class ConfigurationManager:
    """
    JSON 설정 파일을 로드하고 검증하는 Configuration Manager 클래스

    주요 기능:
    1. JSON 설정 파일 로드 및 파싱
    2. Pydantic을 통한 필수 필드 및 타입 검증
    3. 타입 안전한 설정값 접근 인터페이스 제공
    4. 에러 처리 및 기본값 설정
    """

    def __init__(self, config_file_path: str):
        """
        ConfigurationManager 초기화

        Args:
            config_file_path: 설정 파일 경로 (문자열)

        Raises:
            ConfigurationError: 파일이 없거나, JSON 파싱 실패, 또는 스키마 검증 실패 시
        """
        self._config_file_path = Path(config_file_path)
        self.config: Optional[Configuration] = None

        # 설정 파일 로드 및 검증 수행
        self._load_and_validate_config()

    def _load_and_validate_config(self) -> None:
        """
        JSON 설정 파일을 로드하고 Pydantic을 사용하여 검증합니다.

        Raises:
            ConfigurationError: 파일이 없거나 JSON 파싱 실패, 또는 검증 실패 시
        """
        # 파일 존재 여부 확인
        if not self._config_file_path.exists():
            raise ConfigurationError(
                f"설정 파일을 찾을 수 없습니다: {self._config_file_path}"
            )

        # 파일 읽기 및 JSON 파싱
        try:
            with open(self._config_file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                self.config = Configuration(**config_data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"설정 파일의 JSON 형식이 올바르지 않습니다: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"설정 파일 스키마 검증 실패: {e}")
        except IOError as e:
            raise ConfigurationError(f"설정 파일을 읽는 중 오류가 발생했습니다: {e}")

    def get_table_names(self) -> List[str]:
        """
        암호화 대상 테이블명 목록을 반환합니다.

        Returns:
            List[str]: 테이블명 목록
        """
        if self.config is None:
            raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")
        return [table.table_name for table in self.config.access_tables]

    def get_columns_for_table(self, table_name: str) -> List[Union[str, ColumnDetail]]:
        """
        특정 테이블의 암호화 대상 칼럼 목록을 반환합니다.

        Args:
            table_name: 테이블명

        Returns:
            List[Union[str, ColumnDetail]]: 칼럼명 목록 (테이블이 없으면 빈 리스트)
        """
        if self.config is None:
             raise ConfigurationError("설정 데이터가 로드되지 않았습니다.")
        for table in self.config.access_tables:
            if table.table_name == table_name:
                return table.columns
        return []

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값을 가져옵니다.

        Args:
            key: 설정 키
            default: 기본값 (키가 없을 경우 반환)

        Returns:
            Any: 설정값 또는 기본값
        """
        if self.config is None:
            return default

        return getattr(self.config, key, default)
