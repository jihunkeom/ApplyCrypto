"""
Type Handler Generator 모듈

MyBatis Type Handler를 자동 생성하여 암복호화를 적용하는 모듈입니다.
Java 비즈니스 로직을 직접 수정하지 않고, Type Handler 클래스를 생성하고
XML 매퍼에 typeHandler 속성을 등록합니다.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config_manager import Configuration
from models.table_access_info import TableAccessInfo
from modifier.error_handler import ErrorHandler
from modifier.llm.llm_factory import create_llm_provider
from modifier.llm.llm_provider import LLMProvider
from modifier.result_tracker import ResultTracker
from persistence.data_persistence_manager import DataPersistenceManager

logger = logging.getLogger("applycrypto.type_handler_generator")


class TypeHandlerGenerator:
    """
    Type Handler Generator 클래스

    MyBatis Type Handler를 자동 생성하여 암복호화를 적용합니다.

    주요 기능:
    1. 템플릿 기반 Type Handler Java 클래스 생성
    2. LLM을 활용한 XML 매퍼 파일 typeHandler 속성 추가
    3. 생성 결과 추적 및 저장
    """

    # encryption_code에서 클래스명 생성 헬퍼
    @staticmethod
    def _encryption_code_to_class_name(encryption_code: str) -> str:
        """
        encryption_code를 TypeHandler 클래스명으로 변환
        예: K_SIGN_JUMIN -> JuminTypeHandler
            K_SIGN_NAME -> NameTypeHandler
        """
        # K_SIGN_ 접두사 제거
        code = encryption_code.replace("K_SIGN_", "").replace("k_sign_", "")
        # CamelCase로 변환
        parts = code.lower().split("_")
        class_name = "".join(part.capitalize() for part in parts) + "TypeHandler"
        return class_name

    # encryption_code별 TypeHandler 템플릿
    TYPE_HANDLER_JAVA_TEMPLATE = """package {package_name};

import java.sql.CallableStatement;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;
import org.apache.ibatis.type.MappedTypes;

import k_sign.CryptoService;

/**
 * {description} 암복호화 TypeHandler
 * 
 * encryption_code: {encryption_code}
 */
@MappedTypes(String.class)
public class {class_name} extends BaseTypeHandler<String> {{

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, String parameter, JdbcType jdbcType) throws SQLException {{
        ps.setString(i, CryptoService.encrypt(parameter, CryptoService.{encryption_code}));
    }}

    @Override
    public String getNullableResult(ResultSet rs, String columnName) throws SQLException {{
        String encryptedValue = rs.getString(columnName);
        if (encryptedValue == null) {{
            return null;
        }}
        return CryptoService.decrypt(encryptedValue, CryptoService.{encryption_code});
    }}

    @Override
    public String getNullableResult(ResultSet rs, int columnIndex) throws SQLException {{
        String encryptedValue = rs.getString(columnIndex);
        if (encryptedValue == null) {{
            return null;
        }}
        return CryptoService.decrypt(encryptedValue, CryptoService.{encryption_code});
    }}

    @Override
    public String getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {{
        String encryptedValue = cs.getString(columnIndex);
        if (encryptedValue == null) {{
            return null;
        }}
        return CryptoService.decrypt(encryptedValue, CryptoService.{encryption_code});
    }}

    /**
     * 암호화 유틸리티 메서드
     */
    public static String encrypt(String plainText) {{
        return CryptoService.encrypt(plainText, CryptoService.{encryption_code});
    }}

    /**
     * 복호화 유틸리티 메서드
     */
    public static String decrypt(String encryptedText) {{
        return CryptoService.decrypt(encryptedText, CryptoService.{encryption_code});
    }}
}}
"""

    XML_MODIFICATION_PROMPT_TEMPLATE = """
## Task
You are a MyBatis expert. Modify the following XML mapper file to use TypeHandler for encryption/decryption.

## Target Columns and their TypeHandlers (from table '{table_name}'):
{columns_info}

## Original XML Content
```xml
{xml_content}
```

## Requirements
1. Add typeHandler attribute to <result> tags that map to target columns, using the specific TypeHandler class for each column
2. Add typeHandler attribute to parameter mappings (#{{columnName}}) for INSERT/UPDATE statements, using the specific TypeHandler class for each column
3. Do NOT modify any other parts of the XML
4. Preserve all formatting and comments
5. Each column has its own specific TypeHandler class - use the correct one for each column

## Output Format
Return the complete modified XML content.
If no modifications are needed, return the original XML as-is with a comment at the top: <!-- NO_MODIFICATION_NEEDED -->

```xml
(modified XML content here)
```
"""

    def __init__(
        self,
        config: Configuration,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        TypeHandlerGenerator 초기화

        Args:
            config: 설정 객체
            llm_provider: LLM 프로바이더 (선택적)
        """
        self.config = config
        self.project_root = Path(config.target_project)

        # LLM 프로바이더 초기화
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = create_llm_provider(provider_name=config.llm_provider)

        # 컴포넌트 초기화
        self.error_handler = ErrorHandler(max_retries=config.max_retries)
        self.result_tracker = ResultTracker(self.project_root)
        self.persistence_manager = DataPersistenceManager(self.project_root)

        # Type Handler 기본 설정
        type_handler_config = config.type_handler
        if type_handler_config:
            self.type_handler_package = type_handler_config.package
            self.type_handler_output_dir = type_handler_config.output_dir
        else:
            self.type_handler_package = "com.example.typehandler"
            self.type_handler_output_dir = "src/main/java"

        logger.info(
            f"TypeHandlerGenerator 초기화 완료: {self.llm_provider.get_provider_name()}"
        )

    def execute(self, dry_run: bool = False, apply_all: bool = False) -> int:
        """
        Type Handler 생성 및 XML 수정 실행

        Args:
            dry_run: 실제 파일 수정 없이 미리보기만 수행
            apply_all: 사용자 확인 없이 모든 변경사항 적용

        Returns:
            int: 종료 코드 (0: 성공, 1: 실패)
        """
        try:
            mode = "미리보기" if dry_run else "실제 수정"
            logger.info(f"Type Handler 생성 시작 (모드: {mode})...")
            print(f"Type Handler 방식으로 암복호화를 적용합니다 (모드: {mode})...")

            # 1. 분석 결과 로드
            print("  [1/4] 분석 결과 로드 중...")
            table_access_info_list = self._load_table_access_info()
            if not table_access_info_list:
                print(
                    "  오류: 테이블 접근 정보가 없습니다. 먼저 'analyze' 명령어를 실행하세요."
                )
                return 1
            print(
                f"  ✓ {len(table_access_info_list)}개의 테이블 접근 정보를 로드했습니다."
            )

            # SQL 추출 결과 로드 확인
            sql_results = self._load_sql_extraction_results()
            if sql_results:
                print(f"  ✓ {len(sql_results)}개의 XML 매퍼 파일 정보를 로드했습니다.")

            # 2. 필요한 encryption_code 수집
            print("\n  [2/4] 필요한 TypeHandler 분석 중...")
            required_encryption_codes = self._collect_required_encryption_codes(
                table_access_info_list
            )
            print(f"  ✓ 필요한 TypeHandler: {', '.join(required_encryption_codes)}")

            # 통계
            total_handlers_created = 0
            total_xml_modified = 0
            total_skipped = 0
            total_failed = 0

            # 3. encryption_code별 TypeHandler 클래스 생성
            print("\n  [3/4] TypeHandler 클래스 생성 중...")
            generated_handlers = {}  # encryption_code -> full_class_name 매핑

            for encryption_code in required_encryption_codes:
                handler_result = self._generate_type_handler_for_encryption_code(
                    encryption_code, dry_run, apply_all
                )

                if handler_result["status"] == "success":
                    total_handlers_created += 1
                    generated_handlers[encryption_code] = handler_result[
                        "full_class_name"
                    ]
                    print(f"    ✓ {handler_result['class_name']} 생성 완료")
                elif handler_result["status"] == "skipped":
                    total_skipped += 1
                    # 이미 존재하는 경우에도 매핑 정보 저장
                    if "full_class_name" in handler_result:
                        generated_handlers[encryption_code] = handler_result[
                            "full_class_name"
                        ]
                    print(
                        f"    - {handler_result.get('class_name', encryption_code)} 건너뜀: {handler_result.get('reason', '')}"
                    )
                else:
                    total_failed += 1
                    print(
                        f"    ✗ {encryption_code} TypeHandler 생성 실패: {handler_result.get('error', '')}"
                    )

            # 4. 테이블별 XML 매퍼 수정
            print("\n  [4/4] XML 매퍼 파일 수정 중...")
            for table_info in table_access_info_list:
                print(f"    - 테이블 '{table_info.table_name}' XML 수정 중...")

                xml_results = self._modify_xml_mappers(
                    table_info,
                    generated_handlers,  # column_type -> TypeHandler 매핑 전달
                    dry_run,
                    apply_all,
                )

                for result in xml_results:
                    if result["status"] == "success":
                        total_xml_modified += 1
                        print(
                            f"      ✓ XML 수정 완료: {Path(result['file_path']).name}"
                        )
                    elif result["status"] == "skipped":
                        print(
                            f"      - XML 수정 건너뜀: {Path(result['file_path']).name}"
                        )
                    else:
                        print(f"      ✗ XML 수정 실패: {result.get('error', '')}")

            # 5. 결과 저장
            self.result_tracker.save_statistics()

            # 6. 최종 통계 출력
            print("\n" + "=" * 60)
            print("Type Handler 생성 완료!")
            print("=" * 60)
            print(f"  - 생성된 Type Handler: {total_handlers_created}개")
            print(f"  - 수정된 XML 파일: {total_xml_modified}개")
            print(f"  - 건너뜀: {total_skipped}개")
            print(f"  - 실패: {total_failed}개")

            if dry_run:
                print("\n[미리보기 모드] 실제 파일은 수정되지 않았습니다.")

            return 0

        except Exception as e:
            logger.exception(f"Type Handler 생성 중 오류: {e}")
            print(f"오류: {e}")
            return 1

    def _collect_required_encryption_codes(
        self, table_access_info_list: List[TableAccessInfo]
    ) -> List[str]:
        """테이블 접근 정보에서 필요한 encryption_code 목록 수집"""
        encryption_codes = set()
        for table_info in table_access_info_list:
            for col in table_info.columns:
                if isinstance(col, dict):
                    encryption_code = col.get("encryption_code", "")
                    if encryption_code:
                        encryption_codes.add(encryption_code)
        return sorted(list(encryption_codes))

    def _generate_type_handler_for_encryption_code(
        self, encryption_code: str, dry_run: bool, apply_all: bool
    ) -> Dict[str, Any]:
        """
        특정 encryption_code에 대한 TypeHandler 클래스 생성

        Args:
            encryption_code: 암호화 코드 (예: K_SIGN_JUMIN, K_SIGN_NAME)
            dry_run: 미리보기 모드
            apply_all: 자동 적용 모드

        Returns:
            Dict[str, Any]: 생성 결과
        """
        if not encryption_code:
            return {
                "status": "failed",
                "error": "encryption_code가 비어있습니다.",
            }

        # encryption_code에서 클래스명 생성
        class_name = self._encryption_code_to_class_name(encryption_code)
        description = f"{encryption_code} 암복호화"
        full_class_name = f"{self.type_handler_package}.{class_name}"

        # 파일 저장 경로 결정
        output_path = self._get_type_handler_output_path(class_name)

        # 이미 존재하는지 확인
        if output_path.exists() and not dry_run:
            return {
                "status": "skipped",
                "reason": "이미 존재함",
                "class_name": class_name,
                "full_class_name": full_class_name,
            }

        try:
            # 템플릿으로 Java 코드 생성
            java_code = self.TYPE_HANDLER_JAVA_TEMPLATE.format(
                package_name=self.type_handler_package,
                class_name=class_name,
                encryption_code=encryption_code,
                description=description,
            )

            # 미리보기 출력
            if not apply_all and not dry_run:
                print(f"\n--- TypeHandler 클래스 미리보기: {class_name} ---")
                print(f"\033[92m{java_code}\033[0m")  # Green
                print(f"--- 저장 경로: {output_path} ---\n")

                choice = self._get_user_confirmation()
                if choice == "n":
                    return {
                        "status": "skipped",
                        "reason": "사용자가 건너뛰기를 선택함",
                        "class_name": class_name,
                        "full_class_name": full_class_name,
                    }
                elif choice == "q":
                    raise KeyboardInterrupt("사용자가 중단을 선택함")

            # 파일 저장
            if not dry_run:
                self._save_type_handler_file(java_code, output_path)

            return {
                "status": "success",
                "class_name": class_name,
                "full_class_name": full_class_name,
                "output_path": str(output_path),
                "java_code": java_code,
            }

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"TypeHandler 생성 실패 ({encryption_code}): {e}")
            return {
                "status": "failed",
                "error": str(e),
                "class_name": class_name,
            }

    def _load_table_access_info(self) -> List[TableAccessInfo]:
        """분석 결과에서 테이블 접근 정보 로드 (config의 encryption_code 병합)"""
        try:
            data = self.persistence_manager.load_from_file(
                "table_access_info.json", TableAccessInfo
            )
            if not data:
                return []

            result = []
            for info in data:
                if isinstance(info, dict):
                    table_info = TableAccessInfo.from_dict(info)
                elif isinstance(info, TableAccessInfo):
                    table_info = info
                else:
                    continue

                # config_manager에서 encryption_code 정보 병합
                table_info = self._merge_encryption_codes(table_info)
                result.append(table_info)

            return result

        except Exception as e:
            logger.error(f"테이블 접근 정보 로드 실패: {e}")
            return []

    def _merge_encryption_codes(self, table_info: TableAccessInfo) -> TableAccessInfo:
        """config_manager에서 encryption_code 정보를 가져와 TableAccessInfo에 병합"""
        # config에서 해당 테이블의 컬럼 정보 가져오기
        config_columns = {}
        for table in self.config.access_tables:
            if table.table_name.lower() == table_info.table_name.lower():
                for col in table.columns:
                    if not isinstance(col, str):
                        # ColumnDetail object
                        col_name = col.name.lower()
                        column_type = col.column_type if col.column_type else ""
                        if col_name:
                            config_columns[col_name] = column_type.lower()
                break

        # table_info의 columns에 encryption_code 추가
        updated_columns = []
        for col in table_info.columns:
            if isinstance(col, dict):
                col_name = col.get("name", "").lower()
                # config에서 encryption_code 가져오기
                if col_name in config_columns:
                    col["encryption_code"] = config_columns[col_name]
                updated_columns.append(col)
            elif isinstance(col, str):
                # 문자열인 경우 dict로 변환
                col_lower = col.lower()
                updated_columns.append(
                    {
                        "name": col,
                        "new_column": False,
                        "encryption_code": config_columns.get(col_lower, ""),
                    }
                )
            else:
                updated_columns.append(col)

        table_info.columns = updated_columns
        return table_info

    def _load_sql_extraction_results(self) -> List[Dict[str, Any]]:
        """sql_extraction_results.json에서 SQL 추출 결과 로드"""
        try:
            data = self.persistence_manager.load_from_file(
                "sql_extraction_results.json"
            )
            return data if data else []
        except Exception as e:
            logger.error(f"SQL 추출 결과 로드 실패: {e}")
            return []

    def _find_xml_files_for_table(self, table_name: str) -> List[Path]:
        """
        sql_extraction_results.json을 참고하여 특정 테이블과 관련된 XML 파일 탐색

        Args:
            table_name: 테이블명

        Returns:
            List[Path]: 관련 XML 파일 경로 목록
        """
        xml_files = []
        table_name_lower = table_name.lower()

        # SQL 추출 결과에서 테이블을 사용하는 XML 파일 찾기
        sql_results = self._load_sql_extraction_results()

        for result in sql_results:
            file_info = result.get("file", {})
            file_path = file_info.get("path", "")

            # XML 파일이 아니면 스킵
            if not str(file_path).endswith(".xml"):
                continue

            sql_queries = result.get("sql_queries", [])

            # 해당 파일의 SQL 쿼리 중에 타겟 테이블을 사용하는 쿼리가 있는지 확인
            for query in sql_queries:
                sql = query.get("sql", "").lower()
                # 테이블명이 SQL에 포함되어 있는지 확인
                if table_name_lower in sql:
                    xml_path = Path(file_path)
                    if xml_path.exists() and xml_path not in xml_files:
                        xml_files.append(xml_path)
                        logger.info(
                            f"테이블 '{table_name}' 관련 XML 파일 발견: {xml_path.name}"
                        )
                    break  # 이 파일에서 이미 찾았으므로 다음 파일로

        return xml_files

    def _find_xml_mapper_files(self) -> List[Path]:
        """
        프로젝트에서 모든 XML 매퍼 파일 탐색 (sql_extraction_results.json 활용)
        """
        xml_files = []

        # SQL 추출 결과에서 XML 파일 목록 추출
        sql_results = self._load_sql_extraction_results()

        for result in sql_results:
            file_info = result.get("file", {})
            file_path = file_info.get("path", "")

            # XML 파일이면 추가
            if file_path.endswith(".xml"):
                xml_path = Path(file_path)
                if xml_path.exists():
                    xml_files.append(xml_path)

        return xml_files

    def _modify_xml_mappers(
        self,
        table_info: TableAccessInfo,
        type_handler_mapping: Dict[str, str],
        dry_run: bool,
        apply_all: bool,
    ) -> List[Dict[str, Any]]:
        """
        관련 XML 매퍼 파일들을 수정하여 typeHandler 속성 추가

        Args:
            table_info: 테이블 접근 정보
            type_handler_mapping: column_type -> TypeHandler 전체 클래스명 매핑
            dry_run: 미리보기 모드
            apply_all: 자동 적용 모드

        Returns:
            List[Dict[str, Any]]: 수정 결과 목록
        """
        results = []

        # sql_extraction_results.json에서 테이블 관련 XML 파일 찾기
        related_xml_files = self._find_xml_files_for_table(table_info.table_name)

        if not related_xml_files:
            logger.warning(
                f"테이블 '{table_info.table_name}'과 관련된 XML 파일을 찾을 수 없습니다."
            )
            return results

        for xml_file in related_xml_files:
            result = self._modify_single_xml_mapper(
                xml_file,
                table_info,
                type_handler_mapping,
                dry_run,
                apply_all,
            )
            results.append(result)

        return results

    def _modify_single_xml_mapper(
        self,
        xml_file: Path,
        table_info: TableAccessInfo,
        type_handler_mapping: Dict[str, str],
        dry_run: bool,
        apply_all: bool,
    ) -> Dict[str, Any]:
        """
        단일 XML 매퍼 파일 수정

        Args:
            xml_file: XML 파일 경로
            table_info: 테이블 접근 정보
            type_handler_mapping: column_type -> TypeHandler 전체 클래스명 매핑
            dry_run: 미리보기 모드
            apply_all: 자동 적용 모드

        Returns:
            Dict[str, Any]: 수정 결과
        """
        try:
            # 원본 XML 읽기
            with open(xml_file, "r", encoding="utf-8") as f:
                original_content = f.read()

            # 컬럼 정보 포맷팅 (TypeHandler 매핑 포함)
            columns_info = self._format_columns_info_with_handler(
                table_info.columns, type_handler_mapping
            )

            # XML 내용의 중괄호를 이스케이프 (MyBatis #{param} 문법 때문에 필요)
            escaped_xml_content = original_content.replace("{", "{{").replace("}", "}}")

            # 프롬프트 생성
            prompt = self.XML_MODIFICATION_PROMPT_TEMPLATE.format(
                table_name=table_info.table_name,
                columns_info=columns_info,
                xml_content=escaped_xml_content,
            )

            # print(prompt)

            # LLM 호출
            response, error = self.error_handler.retry_with_backoff(
                self.llm_provider.call, prompt
            )

            if error:
                return {
                    "status": "failed",
                    "file_path": str(xml_file),
                    "error": str(error),
                }

            # 응답에서 XML 코드 추출
            modified_xml = self._extract_xml_code(response)
            if not modified_xml:
                return {
                    "status": "failed",
                    "file_path": str(xml_file),
                    "error": "LLM 응답에서 XML 코드를 추출할 수 없습니다.",
                }

            # 수정 필요 없음 체크
            if "NO_MODIFICATION_NEEDED" in modified_xml:
                return {
                    "status": "skipped",
                    "file_path": str(xml_file),
                    "reason": "수정이 필요하지 않음",
                }

            # 변경사항 확인
            if original_content.strip() == modified_xml.strip():
                return {
                    "status": "skipped",
                    "file_path": str(xml_file),
                    "reason": "변경사항 없음",
                }

            # 미리보기 및 사용자 확인
            if not apply_all and not dry_run:
                self._print_xml_diff(xml_file, original_content, modified_xml)
                choice = self._get_user_confirmation()
                if choice == "n":
                    return {
                        "status": "skipped",
                        "file_path": str(xml_file),
                        "reason": "사용자가 건너뛰기를 선택함",
                    }
                elif choice == "q":
                    raise KeyboardInterrupt("사용자가 중단을 선택함")

            # 파일 저장
            if not dry_run:
                # 백업 생성
                self.error_handler.backup_file(xml_file)

                with open(xml_file, "w", encoding="utf-8") as f:
                    f.write(modified_xml)

                logger.info(f"XML 파일 수정 완료: {xml_file}")
                print(f"XML 파일 수정 완료: {xml_file}")

            return {
                "status": "success",
                "file_path": str(xml_file),
                "original": original_content,
                "modified": modified_xml,
            }

        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback

            logger.error(f"XML 수정 실패: {xml_file} - {e}")
            logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
            return {
                "status": "failed",
                "file_path": str(xml_file),
                "error": str(e),
            }

    def _format_columns_info_with_handler(
        self, columns: List[Any], type_handler_mapping: Dict[str, str]
    ) -> str:
        """컬럼 정보를 TypeHandler 매핑과 함께 포맷팅 (XML 수정용)"""
        lines = []
        for col in columns:
            if isinstance(col, dict):
                col_name = col.get("name", str(col))
                encryption_code = col.get("encryption_code", "")

                if encryption_code and encryption_code in type_handler_mapping:
                    handler_class = type_handler_mapping[encryption_code]
                    lines.append(f'  - {col_name} -> typeHandler="{handler_class}"')
                else:
                    lines.append(f"  - {col_name} (no TypeHandler)")
            else:
                lines.append(f"  - {col}")
        return "\n".join(lines)

    def _get_type_handler_output_path(self, class_name: str) -> Path:
        """Type Handler 파일 저장 경로 결정"""
        # 패키지 경로로 변환
        package_path = self.type_handler_package.replace(".", "/")
        output_dir = self.project_root / self.type_handler_output_dir / package_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{class_name}.java"

    def _save_type_handler_file(self, java_code: str, output_path: Path) -> None:
        """Type Handler Java 파일 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(java_code)
        logger.info(f"Type Handler 파일 저장: {output_path}")

    def _extract_xml_code(self, response: Dict[str, Any]) -> Optional[str]:
        """LLM 응답에서 XML 코드 추출"""
        content = response.get("content", "")
        if not content:
            return None

        # ```xml ... ``` 블록 추출
        pattern = r"```xml\s*(.*?)\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 코드 블록 없이 직접 반환된 경우
        if content.strip().startswith("<?xml") or content.strip().startswith("<"):
            return content.strip()

        return None

    def _print_xml_diff(self, file_path: Path, original: str, modified: str) -> None:
        """XML 변경사항 출력"""
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
        )

        print(f"\n[Diff] {file_path}")
        print("-" * 80)
        for line in diff:
            if line.startswith("+"):
                print(f"\033[92m{line}\033[0m", end="")  # Green
            elif line.startswith("-"):
                print(f"\033[91m{line}\033[0m", end="")  # Red
            elif line.startswith("@@"):
                print(f"\033[96m{line}\033[0m", end="")  # Cyan
            else:
                print(line, end="")
        print("-" * 80)

    def _get_user_confirmation(self) -> str:
        """사용자 확인 입력"""
        while True:
            choice = input(
                "\n이 변경사항을 적용하시겠습니까? [y/n/a/q] "
                "(y:적용, n:건너뛰기, a:모두적용, q:중단): "
            ).lower()
            if choice in ["y", "n", "a", "q"]:
                return choice
            print("잘못된 입력입니다.")
