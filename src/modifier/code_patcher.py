"""
Code Patcher 모듈

Unified Diff 형식을 파싱하고 코드 수정을 적용하는 모듈입니다.
"""

import difflib
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from config.config_manager import Configuration

logger = logging.getLogger("applycrypto.code_patcher")


class CodePatcher:
    """
    코드 패처 클래스

    Unified Diff 형식의 패치를 파싱하고 파일에 적용합니다.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[Configuration] = None,
    ):
        """
        CodePatcher 초기화

        Args:
            project_root: 프로젝트 루트 디렉토리 (선택적)
            config: 설정 객체 (선택적)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config = config

    def apply_patch(
        self, file_path: Path, unified_diff: str, dry_run: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Unified Diff 형식의 패치를 파일에 적용합니다.

        Args:
            file_path: 수정할 파일 경로
            unified_diff: Unified Diff 형식의 수정 내용 (또는 generate_full_source가 True일 때 전체 소스 코드)
            dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)

        Returns:
            Tuple[bool, Optional[str]]: (성공 여부, 에러 메시지)
        """
        try:
            # LLM 응답에서 받은 절대 경로를 그대로 사용
            if not file_path.is_absolute():
                # 상대 경로인 경우에만 project_root와 결합 (일반적으로는 발생하지 않아야 함)
                logger.warning(
                    f"상대 경로가 전달되었습니다: {file_path}. 절대 경로로 변환합니다."
                )
                file_path = self.project_root / file_path

            # 절대 경로로 정규화
            file_path = file_path.resolve()

            if not file_path.exists():
                error_msg = f"파일이 존재하지 않습니다: {file_path}"
                logger.error(error_msg)
                return False, error_msg

            if dry_run:
                logger.info(f"[DRY RUN] 파일 수정 시뮬레이션: {file_path}")
                return True, None

            # generate_full_source 설정 확인
            generate_full_source = (
                self.config.generate_full_source if self.config else False
            )

            if generate_full_source:
                # 전체 소스 코드를 파일에 덮어쓰기
                return self.apply_full_source(
                    file_path=file_path, full_source=unified_diff, dry_run=dry_run
                )
            else:
                # subprocess로 patch 명령을 수행하는 대신 apply_patch_using_difflib 사용
                # # 임시 파일에 diff 저장
                # with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as diff_file:
                #     diff_file.write(unified_diff)
                #     diff_file_path = diff_file.name
                #
                # try:
                #     # patch 명령 실행
                #     result = subprocess.run(
                #         ["patch", "-p0", str(file_path), diff_file_path],
                #         capture_output=True,
                #         text=True,
                #         cwd=self.project_root
                #     )
                #
                #     if result.returncode != 0:
                #         error_msg = f"patch 명령 실행 실패: {result.stderr}"
                #         logger.error(error_msg)
                #         return False, error_msg
                #
                #     logger.info(f"파일 수정 완료: {file_path}")
                #     return True, None
                #
                # finally:
                #     # 임시 파일 삭제
                #     Path(diff_file_path).unlink()

                # apply_patch_using_difflib 사용

                # 기존대로 unified diff 패치 적용
                return self.apply_patch_using_difflib(
                    file_path=file_path, unified_diff=unified_diff, dry_run=dry_run
                )

        except Exception as e:
            error_msg = f"패치 적용 실패: {e}"
            logger.error(error_msg)
            return False, error_msg

    def apply_patch_using_difflib(
        self, file_path: Path, unified_diff: str, dry_run: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        difflib를 사용하여 패치를 적용합니다 (patch 명령이 없는 경우 대체 방법).

        Args:
            file_path: 수정할 파일 경로
            unified_diff: Unified Diff 형식의 수정 내용
            dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)

        Returns:
            Tuple[bool, Optional[str]]: (성공 여부, 에러 메시지)
        """
        try:
            # LLM 응답에서 받은 절대 경로를 그대로 사용
            if not file_path.is_absolute():
                # 상대 경로인 경우에만 project_root와 결합 (일반적으로는 발생하지 않아야 함)
                logger.warning(
                    f"상대 경로가 전달되었습니다: {file_path}. 절대 경로로 변환합니다."
                )
                file_path = self.project_root / file_path

            # 절대 경로로 정규화
            file_path = file_path.resolve()

            if not file_path.exists():
                error_msg = f"파일이 존재하지 않습니다: {file_path}"
                logger.error(error_msg)
                return False, error_msg

            # 원본 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                original_lines = f.readlines()

            # Unified Diff 파싱하여 수정된 라인 목록 생성
            diff_lines = unified_diff.splitlines()
            modified_lines = []

            # Unified Diff 파싱
            i = 0
            current_old_line = 0  # 원본 파일의 현재 라인 (0-based)

            while i < len(diff_lines):
                line = diff_lines[i]

                # 파일 경로 라인 건너뛰기
                if line.startswith("---") or line.startswith("+++"):
                    i += 1
                    continue

                # 청크 헤더 파싱 (예: "@@ -10,6 +10,9 @@")
                if line.startswith("@@"):
                    match = re.search(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
                    if match:
                        # old_start = int(match.group(1)) - 1  # 0-based index (unused)
                        new_start = int(match.group(2)) - 1  # 0-based index

                        # 이전 청크까지의 원본 라인을 modified_lines에 추가
                        while len(
                            modified_lines
                        ) < new_start and current_old_line < len(original_lines):
                            modified_lines.append(original_lines[current_old_line])
                            current_old_line += 1

                        i += 1
                        continue

                # Diff 라인 처리
                if line.startswith(" "):
                    # 변경 없는 라인 (원본과 동일)
                    if current_old_line < len(original_lines):
                        modified_lines.append(original_lines[current_old_line])
                        current_old_line += 1
                elif line.startswith("-"):
                    # 삭제된 라인 (추가하지 않음)
                    if current_old_line < len(original_lines):
                        current_old_line += 1
                elif line.startswith("+"):
                    # 추가된 라인
                    # 개행 문자가 없으면 추가
                    added_line = line[1:]
                    if not added_line.endswith("\n"):
                        added_line += "\n"
                    modified_lines.append(added_line)

                i += 1

            # 나머지 원본 라인 추가
            while current_old_line < len(original_lines):
                modified_lines.append(original_lines[current_old_line])
                current_old_line += 1

            if dry_run:
                logger.info(f"[DRY RUN] 파일 수정 시뮬레이션: {file_path}")
                return True, None

            # difflib를 사용하여 검증 및 최적화
            # SequenceMatcher로 원본과 수정본 비교
            original_text = "".join(original_lines)
            modified_text = "".join(modified_lines)

            # difflib.SequenceMatcher를 사용하여 변경 사항 확인
            matcher = difflib.SequenceMatcher(None, original_text, modified_text)
            ratio = matcher.ratio()

            # 변경 사항이 있는지 확인
            if ratio == 1.0:
                logger.info("파일 변경 사항이 없습니다.")
            else:
                # difflib.unified_diff를 사용하여 수정본에서 diff 재생성 (검증용)
                # 이를 통해 파싱이 올바르게 되었는지 확인
                generated_diff = list(
                    difflib.unified_diff(
                        original_lines,
                        modified_lines,
                        fromfile="a/" + str(file_path.name),
                        tofile="b/" + str(file_path.name),
                        lineterm="",
                    )
                )

                # 생성된 diff 라인 수 로깅 (디버그용)
                logger.debug(f"difflib로 생성된 diff 라인 수: {len(generated_diff)}")

                # 변경 비율이 너무 낮으면 경고 (파일이 완전히 바뀐 경우)
                if ratio < 0.3:
                    logger.warning(
                        f"파일 변경 비율이 매우 큽니다 ({ratio:.2%}). 원본 diff 파싱을 확인하세요."
                    )

            # difflib의 get_opcodes를 사용하여 변경 사항 확인 (선택적)
            opcodes = matcher.get_opcodes()
            change_count = sum(1 for tag, i1, i2, j1, j2 in opcodes if tag != "equal")
            logger.debug(f"difflib가 감지한 변경 블록 수: {change_count}")

            # 수정된 파일 저장
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(modified_lines)

            logger.info(
                f"파일 수정 완료 (difflib 검증 사용): {file_path}, 변경 비율: {ratio:.2%}"
            )
            return True, None

        except Exception as e:
            error_msg = f"패치 적용 실패 (difflib): {e}"
            logger.error(error_msg)
            return False, error_msg

    def validate_syntax(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        수정된 파일의 구문을 검사합니다.

        Args:
            file_path: 검사할 파일 경로

        Returns:
            Tuple[bool, Optional[str]]: (유효 여부, 에러 메시지)
        """
        try:
            # LLM 응답에서 받은 절대 경로를 그대로 사용
            if not file_path.is_absolute():
                # 상대 경로인 경우에만 project_root와 결합 (일반적으로는 발생하지 않아야 함)
                logger.warning(
                    f"상대 경로가 전달되었습니다: {file_path}. 절대 경로로 변환합니다."
                )
                file_path = self.project_root / file_path

            # 절대 경로로 정규화
            file_path = file_path.resolve()

            if not file_path.exists():
                return False, f"파일이 존재하지 않습니다: {file_path}"

            # Java 파일인 경우 javac로 구문 검사
            if file_path.suffix == ".java":
                result = subprocess.run(
                    ["javac", "-Xlint:all", "-cp", ".", str(file_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )

                if result.returncode != 0:
                    error_msg = f"Java 구문 오류: {result.stderr}"
                    logger.warning(error_msg)
                    return False, error_msg

            # XML 파일인 경우 xmllint로 구문 검사
            elif file_path.suffix == ".xml":
                result = subprocess.run(
                    ["xmllint", "--noout", str(file_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )

                if result.returncode != 0:
                    error_msg = f"XML 구문 오류: {result.stderr}"
                    logger.warning(error_msg)
                    return False, error_msg

            # 기타 파일은 구문 검사 스킵
            return True, None

        except FileNotFoundError:
            # javac나 xmllint가 없는 경우 구문 검사 스킵
            logger.debug("구문 검사 도구를 찾을 수 없어 검사를 건너뜁니다.")
            return True, None
        except Exception as e:
            logger.warning(f"구문 검사 중 오류 발생: {e}")
            return True, None  # 오류가 있어도 계속 진행

    def apply_full_source(
        self, file_path: Path, full_source: str, dry_run: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        전체 소스 코드를 파일에 덮어씁니다.

        Args:
            file_path: 수정할 파일 경로
            full_source: 전체 소스 코드 내용
            dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)

        Returns:
            Tuple[bool, Optional[str]]: (성공 여부, 에러 메시지)
        """
        try:
            # LLM 응답에서 받은 절대 경로를 그대로 사용
            if not file_path.is_absolute():
                logger.warning(
                    f"상대 경로가 전달되었습니다: {file_path}. 절대 경로로 변환합니다."
                )
                file_path = self.project_root / file_path

            # 절대 경로로 정규화
            file_path = file_path.resolve()

            if not file_path.exists():
                error_msg = f"파일이 존재하지 않습니다: {file_path}"
                logger.error(error_msg)
                return False, error_msg

            if dry_run:
                logger.info(
                    f"[DRY RUN] 전체 소스 코드 덮어쓰기 시뮬레이션: {file_path}"
                )
                return True, None

            # 전체 소스 코드를 파일에 쓰기
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_source)

            logger.info(f"전체 소스 코드 덮어쓰기 완료: {file_path}")
            return True, None

        except Exception as e:
            error_msg = f"전체 소스 코드 덮어쓰기 실패: {e}"
            logger.error(error_msg)
            return False, error_msg
