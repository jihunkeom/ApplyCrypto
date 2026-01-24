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
    column_type: Optional[Literal["dob", "rrn", "name", "sex"]] = Field(
        None,
        description="컬럼 타입 (dob: 생년월일, rrn: 주민등록번호, name: 이름, sex: 성별)",
    )
    encryption_code: Optional[str] = Field(None, description="암호화 코드 (예: P017, P018, P019)")


class AccessTable(BaseModel):
    table_name: str = Field(..., description="테이블 이름")
    columns: List[Union[str, ColumnDetail]] = Field(..., description="컬럼 목록")


class TypeHandlerConfig(BaseModel):
    package: str = Field(..., description="Type Handler 패키지 이름")
    output_dir: str = Field(..., description="Type Handler 출력 디렉터리")


class MultiStepExecutionConfig(BaseModel):
    """TwoStep/ThreeStep 공통 실행 옵션

    mode 값에 따른 동작:
    - "full": Planning + Execution 전체 실행 (기본값)
    - "plan_only": Planning까지만 실행하고 종료, 결과는 JSON으로 저장
    - "execution_only": 이전 Planning 결과를 사용하여 Execution만 실행 (plan_timestamp 필수)
    """

    mode: Literal["full", "plan_only", "execution_only"] = Field(
        "full",
        description="실행 모드. 'full': 전체 실행, 'plan_only': Planning까지만, "
        "'execution_only': Execution만 실행",
    )
    plan_timestamp: Optional[str] = Field(
        None,
        description="execution_only 모드에서 사용할 이전 Planning 결과의 timestamp. "
        "예: '20250123_143052'. 해당 timestamp 폴더 내 모든 plan을 실행.",
    )


class TwoStepConfig(BaseModel):
    """2-Step LLM 협업 설정 (Planning + Execution)"""

    planning_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field(..., description="Planning 단계에서 사용할 LLM 프로바이더")
    planning_model: Optional[str] = Field(
        None, description="Planning 단계에서 사용할 모델 ID (예: gpt-oss-120b)"
    )
    execution_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field(..., description="Execution 단계에서 사용할 LLM 프로바이더")
    execution_model: Optional[str] = Field(
        None, description="Execution 단계에서 사용할 모델 ID (예: codestral-2508)"
    )
    execution_options: Optional[MultiStepExecutionConfig] = Field(
        None,
        description="실행 옵션 (mode, plan_timestamp)",
    )


class ThreeStepConfig(BaseModel):
    """3-Step LLM 협업 설정 (VO Extraction + Planning + Execution)

    1단계 (VO Extraction): VO 파일 + SQL 쿼리 → vo_info JSON 추출
    2단계 (Planning): vo_info + 소스코드 + call chain → modification_instructions
    3단계 (Execution): modification_instructions + 소스코드 → 수정된 코드

    1/2단계는 분석용 모델 (reasoning 우수), 3단계는 코드 생성용 모델 사용
    """

    analysis_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field(..., description="1/2단계 (VO Extraction + Planning)에서 사용할 LLM 프로바이더")
    analysis_model: Optional[str] = Field(
        None, description="1/2단계에서 사용할 모델 ID (예: gpt-oss-120b)"
    )
    execution_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field(..., description="3단계 (Execution)에서 사용할 LLM 프로바이더")
    execution_model: Optional[str] = Field(
        None, description="3단계에서 사용할 모델 ID (예: codestral-2508)"
    )
    execution_options: Optional[MultiStepExecutionConfig] = Field(
        None,
        description="실행 옵션 (mode, plan_timestamp)",
    )


class Configuration(BaseModel):
    target_project: str = Field(..., description="대상 프로젝트 루트 경로")
    type_handler: Optional[TypeHandlerConfig] = Field(
        None, description="Type Handler 설정"
    )
    source_file_types: List[str] = Field(
        ..., description="수집할 소스 파일 확장자 목록"
    )
    framework_type: Literal[
        "SpringMVC",
        "AnyframeSarangOn",
        "AnyframeOld",
        "AnyframeEtc",
        "AnyframeCCS",
        "SpringBatQrts",
        "AnyframeBatSarangOn",
        "AnyframeBatEtc",
        "anyframe_ccs_batch",
    ] = Field(
        "SpringMVC", description="프레임워크 타입"
    )
    sql_wrapping_type: Literal["mybatis", "mybatis_ccs", "mybatis_ccs_batch", "jdbc", "jpa"] = Field(
        ..., description="SQL Wrapping 타입"
    )
    access_tables: List[AccessTable] = Field(
        ..., description="암호화 대상 테이블 및 칼럼 정보"
    )
    modification_type: Literal[
        "TypeHandler", "ControllerOrService", "ServiceImplOrBiz", "TwoStep", "ThreeStep"
    ] = Field(..., description="코드 수정 타입")
    two_step_config: Optional[TwoStepConfig] = Field(
        None, description="2-Step LLM 협업 설정 (modification_type이 TwoStep일 때 필수)"
    )
    three_step_config: Optional[ThreeStepConfig] = Field(
        None, description="3-Step LLM 협업 설정 (modification_type이 ThreeStep일 때 필수)"
    )
    llm_provider: Literal[
        "watsonx_ai", "claude_ai", "openai", "mock", "watsonx_ai_on_prem"
    ] = Field("watsonx_ai", description="사용할 LLM 프로바이더")
    exclude_dirs: List[str] = Field(
        default_factory=list, description="제외할 디렉터리 이름 목록"
    )
    exclude_files: List[str] = Field(
        default_factory=list, description="제외할 파일 패턴 목록"
    )
    use_call_chain_mode: bool = Field(False, description="Call Chain 모드 사용 여부")
    use_llm_parser: bool = Field(False, description="LLM 파서 사용 여부")
    max_tokens_per_batch: int = Field(8000, description="한번에 처리할 최대 토큰 수")
    max_workers: int = Field(4, description="병렬 처리 워커 수")
    max_retries: int = Field(3, description="최대 재시도 횟수")
    generate_full_source: bool = Field(
        False,
        description="전체 소스 코드를 포함할지 여부 (true: 전체 코드, false: 관련 부분만)",
    )

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
            
            # 하위 호환성: 마이그레이션 유틸리티를 사용하여 자동 변환
            from .config_migration import ConfigMigration
            
            migrator = ConfigMigration(str(path))
            migration_result = migrator.migrate(update_file=False, backup=False)
            
            if migration_result["migrated"]:
                # 마이그레이션이 필요한 경우 자동으로 변환
                for key, value in migration_result["new_values"].items():
                    config_data[key] = value
                
                # 경고 메시지 출력
                print(
                    f"\n[경고] config.json에서 마이그레이션이 필요한 필드가 발견되었습니다."
                )
                for change in migration_result["changes"]:
                    print(f"  - {change}")
                print()
                
                # 사용자에게 마이그레이션 여부 확인
                while True:
                    try:
                        response = input(
                            "config.json 파일을 자동으로 마이그레이션하시겠습니까? (yes/no): "
                        ).strip().lower()
                        
                        if response in ["yes", "y"]:
                            # 마이그레이션 실행
                            from .config_migration import migrate_config_file
                            
                            print("\n[정보] config.json 파일을 마이그레이션하는 중...")
                            migrate_result = migrate_config_file(
                                str(path),
                                update_file=True,
                                backup=True,
                                save_log=True,
                            )
                            
                            if migrate_result["backup_path"]:
                                print(f"[정보] 백업 파일이 생성되었습니다: {migrate_result['backup_path']}")
                            
                            print("[정보] 마이그레이션이 완료되었습니다.\n")
                            
                            # 업데이트된 파일 다시 읽기
                            with open(path, "r", encoding="utf-8") as f:
                                config_data = json.load(f)
                            break
                            
                        elif response in ["no", "n"]:
                            print(
                                "\n[권장] 나중에 config.json을 업데이트하여 구식 필드를 제거하고 "
                                f"새로운 필드를 사용하세요. 자동 업데이트를 원하시면 "
                                f"다음 명령어를 사용하세요:\n"
                                f"  from config import migrate_config_file\n"
                                f"  migrate_config_file('{path}', update_file=True, backup=True)\n"
                            )
                            break
                        else:
                            print("  'yes' 또는 'no'를 입력해주세요.")
                    except (EOFError, KeyboardInterrupt):
                        # 사용자가 Ctrl+C를 누르거나 입력이 중단된 경우
                        print("\n[정보] 마이그레이션이 취소되었습니다. 현재 설정으로 계속 진행합니다.\n")
                        break
            
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
