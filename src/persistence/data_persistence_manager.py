"""
Data Persistence Manager 모듈

분석 결과를 JSON 파일로 직렬화하여 저장하고 로드하는 데이터 영속화 계층입니다.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from persistence.cache_manager import CacheManager
from persistence.json_decoder import CustomJSONDecoder
from persistence.json_encoder import CustomJSONEncoder

T = TypeVar("T")


class PersistenceError(Exception):
    """데이터 영속화 관련 에러를 나타내는 사용자 정의 예외 클래스"""

    pass


class DataPersistenceManager:
    """
    데이터 영속화 관리자 클래스

    주요 기능:
    1. JSON 직렬화/역직렬화
    2. 프로젝트별 결과 디렉터리 관리
    3. 버전 관리 (타임스탬프 기반)
    4. 데이터 검증 (JSON Schema)
    5. 캐싱
    """

    def __init__(
        self,
        target_project: Path,
        output_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        DataPersistenceManager 초기화

        Args:
            project_path: 프로젝트 루트 경로
            output_dir: 결과 저장 디렉터리 (None이면 현재 작업 디렉터리/.applycrypto/results 사용)
            enable_cache: 캐싱 활성화 여부
        """
        self.target_project = Path(target_project)
        if output_dir is None:
            self.output_dir = self.target_project / ".applycrypto" / "results"
        else:
            self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

        # 결과 디렉터리 생성
        self._ensure_output_directory()

        # 캐시 관리자 초기화
        if enable_cache:
            cache_dir = self.output_dir.parent / "cache"
            self.cache_manager = CacheManager(cache_dir)
        else:
            self.cache_manager = None

    def _ensure_output_directory(self) -> None:
        """결과 디렉터리가 존재하는지 확인하고 없으면 생성"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise PersistenceError(
                f"결과 디렉터리를 생성할 수 없습니다: {self.output_dir} - {e}"
            )

    def serialize_to_json(self, data: Any, indent: int = 2) -> str:
        """
        데이터 모델 객체를 JSON 문자열로 변환

        Args:
            data: 직렬화할 데이터 (리스트, 딕셔너리, 데이터 모델 객체)
            indent: JSON 들여쓰기 (기본값: 2)

        Returns:
            str: JSON 문자열

        Raises:
            PersistenceError: 직렬화 실패 시
        """
        try:
            return json.dumps(
                data, cls=CustomJSONEncoder, indent=indent, ensure_ascii=False
            )
        except (TypeError, ValueError) as e:
            raise PersistenceError(f"JSON 직렬화 실패: {e}")

    def deserialize_from_json(
        self, json_str: str, model_class: Optional[Type[T]] = None
    ) -> Any:
        """
        JSON 문자열을 데이터 모델 객체로 복원

        Args:
            json_str: JSON 문자열
            model_class: 복원할 모델 클래스 (선택적)

        Returns:
            복원된 데이터 객체

        Raises:
            PersistenceError: 역직렬화 실패 시
        """
        try:
            data = json.loads(json_str)

            # 커스텀 디코더로 특수 타입 복원
            decoded_data = CustomJSONDecoder.decode_value(data)

            # 모델 클래스가 지정된 경우 from_dict로 변환
            if model_class and hasattr(model_class, "from_dict"):
                if isinstance(decoded_data, list):
                    return [model_class.from_dict(item) for item in decoded_data]
                elif isinstance(decoded_data, dict):
                    return model_class.from_dict(decoded_data)

            return decoded_data

        except json.JSONDecodeError as e:
            raise PersistenceError(f"JSON 역직렬화 실패: 잘못된 JSON 형식 - {e}")
        except (TypeError, ValueError, KeyError) as e:
            raise PersistenceError(f"JSON 역직렬화 실패: 데이터 변환 오류 - {e}")

    def save_to_file(
        self, data: Any, filename: str, subdirectory: Optional[str] = None
    ) -> Path:
        """
        데이터를 JSON 파일로 저장

        Args:
            data: 저장할 데이터
            filename: 파일명 (확장자 포함)
            subdirectory: 하위 디렉터리 (선택적)

        Returns:
            Path: 저장된 파일 경로

        Raises:
            PersistenceError: 파일 저장 실패 시
        """
        try:
            # 저장 디렉터리 결정
            save_dir = self.output_dir
            if subdirectory:
                save_dir = self.output_dir / subdirectory
                save_dir.mkdir(parents=True, exist_ok=True)

            # 파일 경로
            file_path = save_dir / filename

            # JSON 직렬화
            json_str = self.serialize_to_json(data)

            # 파일 저장
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)

            self.logger.info(f"데이터 저장 완료: {file_path}")
            return file_path

        except (OSError, PermissionError) as e:
            raise PersistenceError(f"파일 저장 실패: {file_path} - {e}")
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"파일 저장 중 예상치 못한 오류: {e}")

    def save_text_file(
        self, content: str, filename: str, subdirectory: Optional[str] = None
    ) -> Path:
        """
        문자열 데이터를 텍스트 파일로 저장

        Args:
            content: 저장할 문자열
            filename: 파일명 (확장자 포함)
            subdirectory: 하위 디렉터리 (선택적)

        Returns:
            Path: 저장된 파일 경로

        Raises:
            PersistenceError: 파일 저장 실패 시
        """
        try:
            # 저장 디렉터리 결정
            save_dir = self.output_dir
            if subdirectory:
                save_dir = self.output_dir / subdirectory
                save_dir.mkdir(parents=True, exist_ok=True)

            # 파일 경로
            file_path = save_dir / filename

            # 파일 저장
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"텍스트 파일 저장 완료: {file_path}")
            return file_path

        except (OSError, PermissionError) as e:
            raise PersistenceError(f"파일 저장 실패: {file_path} - {e}")
        except Exception as e:
            raise PersistenceError(f"파일 저장 중 예상치 못한 오류: {e}")

    def load_from_file(
        self,
        filename: str,
        model_class: Optional[Type[T]] = None,
        subdirectory: Optional[str] = None,
    ) -> Any:
        """
        JSON 파일에서 데이터 로드

        Args:
            filename: 파일명 (확장자 포함)
            model_class: 복원할 모델 클래스 (선택적)
            subdirectory: 하위 디렉터리 (선택적)

        Returns:
            로드된 데이터 객체

        Raises:
            PersistenceError: 파일 로드 실패 시
        """
        try:
            # 파일 경로 결정
            load_dir = self.output_dir
            if subdirectory:
                load_dir = self.output_dir / subdirectory

            file_path = load_dir / filename

            # 파일 존재 확인
            if not file_path.exists():
                raise PersistenceError(f"파일을 찾을 수 없습니다: {file_path}")

            # 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                json_str = f.read()

            # JSON 역직렬화
            data = self.deserialize_from_json(json_str, model_class)

            self.logger.info(f"데이터 로드 완료: {file_path}")
            return data

        except FileNotFoundError:
            raise PersistenceError(f"파일을 찾을 수 없습니다: {file_path}")
        except (OSError, PermissionError) as e:
            raise PersistenceError(f"파일 읽기 실패: {file_path} - {e}")
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"파일 로드 중 예상치 못한 오류: {e}")

    def add_timestamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터에 타임스탬프 추가

        Args:
            data: 타임스탬프를 추가할 데이터 딕셔너리

        Returns:
            타임스탬프가 추가된 데이터 딕셔너리
        """
        now = datetime.now()

        # 기존 타임스탬프가 없으면 생성 시간 추가
        if "created_time" not in data:
            data["created_time"] = now.isoformat()

        # 수정 시간 업데이트
        data["modified_time"] = now.isoformat()

        return data

    def get_version_info(
        self, filename: str, subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        파일의 버전 정보 조회

        Args:
            filename: 파일명
            subdirectory: 하위 디렉터리 (선택적)

        Returns:
            버전 정보 딕셔너리 (created_time, modified_time 포함)
        """
        try:
            data = self.load_from_file(filename, subdirectory=subdirectory)

            if isinstance(data, dict):
                return {
                    "created_time": data.get("created_time"),
                    "modified_time": data.get("modified_time"),
                    "file_size": self._get_file_size(filename, subdirectory),
                }
            else:
                return {
                    "created_time": None,
                    "modified_time": None,
                    "file_size": self._get_file_size(filename, subdirectory),
                }
        except PersistenceError:
            return {"created_time": None, "modified_time": None, "file_size": 0}

    def _get_file_size(self, filename: str, subdirectory: Optional[str] = None) -> int:
        """파일 크기 조회"""
        load_dir = self.output_dir
        if subdirectory:
            load_dir = self.output_dir / subdirectory

        file_path = load_dir / filename
        if file_path.exists():
            return file_path.stat().st_size
        return 0

    def handle_corrupted_file(self, file_path: Path) -> bool:
        """
        손상된 JSON 파일 처리

        Args:
            file_path: 손상된 파일 경로

        Returns:
            bool: 복구 성공 여부
        """
        try:
            # 백업 파일 확인
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            if backup_path.exists():
                # 백업에서 복원
                import shutil

                shutil.copy2(backup_path, file_path)
                self.logger.info(f"백업에서 파일 복원: {file_path}")
                return True

            # 백업이 없으면 False 반환
            self.logger.warning(f"손상된 파일 복구 불가: {file_path} (백업 파일 없음)")
            return False

        except Exception as e:
            self.logger.error(f"파일 복구 중 오류: {e}")
            return False

    def handle_permission_error(self, file_path: Path) -> None:
        """
        파일 권한 문제 처리

        Args:
            file_path: 권한 문제가 있는 파일 경로

        Raises:
            PersistenceError: 권한 문제 설명과 함께 예외 발생
        """
        raise PersistenceError(
            f"파일 접근 권한이 없습니다: {file_path}\n"
            f"파일 권한을 확인하거나 관리자 권한으로 실행해주세요."
        )

    def create_backup(self, file_path: Path) -> Path:
        """
        파일 백업 생성

        Args:
            file_path: 백업할 파일 경로

        Returns:
            Path: 백업 파일 경로
        """
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        try:
            import shutil

            if file_path.exists():
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"백업 파일 생성: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"백업 파일 생성 실패: {e}")
            return backup_path

    def get_cached_result(self, file_path: Path) -> Optional[Any]:
        """
        캐시된 결과 조회

        Args:
            file_path: 원본 파일 경로

        Returns:
            캐시된 결과 (없으면 None)
        """
        if self.cache_manager:
            return self.cache_manager.get_cached_result(file_path)
        return None

    def set_cached_result(self, file_path: Path, data: Any) -> None:
        """
        결과를 캐시에 저장

        Args:
            file_path: 원본 파일 경로
            data: 캐시할 데이터
        """
        if self.cache_manager:
            self.cache_manager.set_cached_result(file_path, data)

    def clear_all(self, use_backup: bool = False) -> None:
        """
        저장된 모든 데이터 삭제

        Args:
            use_backup: 삭제 전 백업 생성 여부
        """
        import shutil

        if not self.output_dir.exists():
            return

        if use_backup:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # output_dir와 동일한 레벨에 백업 생성
                backup_dir = self.output_dir.with_name(
                    f"{self.output_dir.name}_backup_{timestamp}"
                )
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

                shutil.copytree(self.output_dir, backup_dir)
                self.logger.info(f"데이터 백업 완료: {backup_dir}")
            except Exception as e:
                raise PersistenceError(f"백업 생성 실패: {e}")

        try:
            shutil.rmtree(self.output_dir)
            self._ensure_output_directory()
            self.logger.info("모든 데이터 삭제 완료")
        except Exception as e:
            raise PersistenceError(f"데이터 삭제 실패: {e}")

    def remove_all_backups(self) -> int:
        """
        타겟 프로젝트 내의 모든 .backup 또는 .backup.[num] 파일을 삭제합니다.

        Returns:
            int: 삭제된 파일 수
        """
        self.logger.info("백업 파일 삭제 시작...")

        # .backup 또는 .backup.[num] 패턴 (예: file.java.backup, file.java.backup.1)
        backup_pattern = re.compile(r".*\.backup(\.\d+)?$")
        count = 0

        for file_path in self.target_project.rglob("*"):
            if file_path.is_file() and backup_pattern.match(file_path.name):
                try:
                    file_path.unlink()
                    self.logger.debug(f"백업 파일 삭제: {file_path}")
                    count += 1
                except Exception as e:
                    self.logger.warning(f"백업 파일 삭제 실패: {file_path} - {e}")

        self.logger.info(f"총 {count}개의 백업 파일이 삭제되었습니다.")
        return count

