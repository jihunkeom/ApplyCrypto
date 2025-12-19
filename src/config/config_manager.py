"""
Configuration Manager 모듈

JSON 설정 파일을 로드하고 검증하는 Configuration Manager를 구현합니다.
프로젝트 경로, 파일 타입, SQL Wrapping 타입, 암호화 대상 테이블/칼럼 정보를 파싱하고 스키마 검증을 수행합니다.
"""

import json
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError


class ConfigurationError(Exception):
    """설정 관련 에러를 나타내는 사용자 정의 예외 클래스"""

    pass


class ColumnDetail(BaseModel):
    name: str = Field(..., description="컬럼 이름")
    new_column: Optional[bool] = Field(None, description="새로운 컬럼 여부")
    column_type: Optional[Literal["dob", "ssn", "name", "sex"]] = Field(
        None,
        description="컬럼 타입 (dob: 생년월일, ssn: 주민번호, name: 이름, sex: 성별)",
    )
    encryption_code: Optional[str] = Field(None, description="암호화 코드")


class AccessTable(BaseModel):
    table_name: str = Field(..., description="테이블 이름")
    columns: List[Union[str, ColumnDetail]] = Field(..., description="컬럼 목록")


class TypeHandlerConfig(BaseModel):
    package: str = Field(..., description="Type Handler 패키지 이름")
    output_dir: str = Field(..., description="Type Handler 출력 디렉터리")


class Configuration(BaseModel):
    target_project: str = Field(..., description="대상 프로젝트 루트 경로")
    type_handler: Optional[TypeHandlerConfig] = Field(
        None, description="Type Handler 설정"
    )
    source_file_types: List[str] = Field(
        ..., description="수집할 소스 파일 확장자 목록"
    )
    sql_wrapping_type: Literal["mybatis", "jdbc", "jpa"] = Field(
        ..., description="SQL Wrapping 타입"
    )
    access_tables: List[AccessTable] = Field(
        ..., description="암호화 대상 테이블 및 칼럼 정보"
    )
    diff_gen_type: str = Field(..., description="Diff Generator 타입")
    llm_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field("watsonx_ai", description="사용할 LLM 프로바이더")
    exclude_dirs: List[str] = Field(
        default_factory=list, description="제외할 디렉터리 이름 목록"
    )
    exclude_files: List[str] = Field(
        default_factory=list, description="제외할 파일 패턴 목록"
    )
    max_tokens_per_batch: int = Field(2000, description="한번에 처리할 최대 토큰 수")
    max_workers: int = Field(4, description="병렬 처리 워커 수")
    max_retries: int = Field(3, description="최대 재시도 횟수")

    def get_table_names(self) -> List[str]:
        """
        암호화 대상 테이블명 목록을 반환합니다.

        Returns:
            List[str]: 테이블명 목록
        """
        return [table.table_name for table in self.access_tables]

    def get_columns_for_table(self, table_name: str) -> List[Union[str, ColumnDetail]]:
        """
        특정 테이블의 암호화 대상 칼럼 목록을 반환합니다.

        Args:
            table_name: 테이블명 (대소문자 구분 없음)

        Returns:
            List[Union[str, ColumnDetail]]: 칼럼명 목록 (테이블이 없으면 빈 리스트)
        """
        for table in self.access_tables:
            if table.table_name == table_name:
                return table.columns
        return []


# 전역 설정 인스턴스
_config: Optional[Configuration] = None


def load_config(config_file_path: str) -> Configuration:
    """
    설정 파일을 로드하고 전역 설정 인스턴스를 설정합니다.

    Args:
        config_file_path: 설정 파일 경로

    Returns:
        Configuration: 로드된 설정 객체

    Raises:
        ConfigurationError: 설정 로드 실패 시
    """
    global _config

    path = Path(config_file_path)
    if not path.exists():
        raise ConfigurationError(f"설정 파일을 찾을 수 없습니다: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            _config = Configuration(**config_data)
            return _config
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"설정 파일의 JSON 형식이 올바르지 않습니다: {e}")
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = " -> ".join(map(str, error["loc"]))
            msg = error["msg"]
            # 주요 에러 메시지 한글화
            if error["type"] == "missing":
                msg = "필수 항목이 누락되었습니다"
            elif "valid value" in msg:
                msg = f"유효한 값이 아닙니다 ({msg})"

            error_messages.append(f"  - 필드: {loc}, 원인: {msg}")

        formatted_error = "\n".join(error_messages)
        raise ConfigurationError(f"설정 파일 검증 실패:\n{formatted_error}")
    except IOError as e:
        raise ConfigurationError(f"설정 파일을 읽는 중 오류가 발생했습니다: {e}")


def get_config() -> Configuration:
    """
    로드된 전역 설정 객체를 반환합니다.

    Returns:
        Configuration: 설정 객체

    Raises:
        ConfigurationError: 설정이 로드되지 않은 경우
    """
    if _config is None:
        raise ConfigurationError(
            "설정이 아직 로드되지 않았습니다. load_config()를 먼저 호출하세요."
        )
    return _config
