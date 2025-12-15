"""
Source File Collector 모듈

Java Spring Boot 프로젝트의 모든 소스 파일을 재귀적으로 탐색하고 메타데이터와 함께 수집합니다.
"""

from pathlib import Path
from typing import List, Iterator, Set, Optional
from datetime import datetime
import os
import fnmatch

from config.config_manager import ConfigurationManager
from models.source_file import SourceFile


class SourceFileCollector:
    """
    소스 파일을 수집하는 클래스
    
    주요 기능:
    1. 재귀적 파일 탐색
    2. 설정 기반 파일 필터링
    3. 메타데이터 추출
    4. 중복 제거
    5. 크로스 플랫폼 호환성
    6. 제너레이터 패턴으로 메모리 효율성 확보
    """
    
    # 제외할 디렉터리 목록 (빌드 디렉터리 및 버전 관리 디렉터리)
    EXCLUDED_DIRS = {
        ".git", ".svn", ".hg",  # 버전 관리
        "target", "build", "out", "bin",  # 빌드 결과물
        ".idea", ".vscode", ".settings",  # IDE 설정
        "node_modules",  # Node.js 의존성
        "__pycache__", ".pytest_cache",  # Python 캐시
        ".gradle", ".mvn",  # 빌드 도구
    }
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        SourceFileCollector 초기화
        
        Args:
            config_manager: ConfigurationManager 인스턴스
        """
        self._config_manager = config_manager
        self._project_path = config_manager.project_path
        self._source_file_types = config_manager.source_file_types
        self._seen_files: Set[Path] = set[Path]()  # 중복 제거를 위한 Set
        
        # 제외할 디렉터리: 기본값과 config에서 가져온 값 병합
        self._excluded_dirs = self.EXCLUDED_DIRS.copy()
        config_exclude_dirs = config_manager.exclude_dirs
        if config_exclude_dirs:
            self._excluded_dirs.update(config_exclude_dirs)
        
        # 제외할 파일 패턴: config에서 가져온 값
        self._exclude_file_patterns = config_manager.exclude_files
    
    def collect(self) -> Iterator[SourceFile]:
        """
        소스 파일을 수집하는 제너레이터
        
        Yields:
            SourceFile: 수집된 소스 파일 메타데이터
            
        Note:
            제너레이터 패턴을 사용하여 메모리 효율성을 확보합니다.
            대규모 프로젝트에서도 전체 파일 목록을 메모리에 로드하지 않습니다.
        """
        # 프로젝트 경로 존재 여부 확인
        if not self._project_path.exists():
            raise ValueError(f"프로젝트 경로가 존재하지 않습니다: {self._project_path}")
        
        if not self._project_path.is_dir():
            raise ValueError(f"프로젝트 경로가 디렉터리가 아닙니다: {self._project_path}")
        
        # 재귀적 파일 탐색 및 수집
        for file_path in self._walk_directory(self._project_path):
            # 파일 필터링
            if not self._should_collect(file_path):
                continue
            
            # 중복 제거
            normalized_path = self._normalize_path(file_path)
            if normalized_path in self._seen_files:
                continue
            self._seen_files.add(normalized_path)
            
            # 메타데이터 추출 및 SourceFile 객체 생성
            try:
                source_file = self._extract_metadata(file_path)
                yield source_file
            except (OSError, PermissionError) as e:
                # 파일 접근 권한 문제 등은 로깅하고 건너뜀
                continue
    
    def collect_all(self) -> List[SourceFile]:
        """
        모든 소스 파일을 수집하여 리스트로 반환
        
        Returns:
            List[SourceFile]: 수집된 모든 소스 파일 목록
            
        Note:
            메모리에 모든 파일을 로드하므로 대규모 프로젝트에서는 collect() 제너레이터 사용을 권장합니다.
        """
        return list(self.collect())
    
    def _walk_directory(self, root_path: Path) -> Iterator[Path]:
        """
        디렉터리를 재귀적으로 탐색하는 제너레이터
        
        Args:
            root_path: 탐색할 루트 디렉터리 경로
            
        Yields:
            Path: 발견된 파일 경로
            
        Note:
            숨김 파일 및 빌드 디렉터리는 제외합니다.
        """
        try:
            # rglob을 사용하여 재귀적 탐색
            for file_path in root_path.rglob("*"):
                # 디렉터리는 건너뜀
                if file_path.is_dir():
                    continue
                
                # 숨김 파일 제외 (파일명이 .으로 시작)
                if file_path.name.startswith("."):
                    continue
                
                # 제외할 디렉터리 내부 파일 제외
                if self._is_excluded_directory(file_path.parent):
                    continue
                
                yield file_path
        except PermissionError:
            # 권한이 없는 디렉터리는 건너뜀
            pass
    
    def _is_excluded_directory(self, dir_path: Path) -> bool:
        """
        디렉터리가 제외 대상인지 확인
        
        Args:
            dir_path: 확인할 디렉터리 경로
            
        Returns:
            bool: 제외 대상이면 True
        """
        # 경로의 각 부분을 확인
        for part in dir_path.parts:
            if part in self._excluded_dirs:
                return True
            # 숨김 디렉터리 제외 (단, 프로젝트 루트 자체는 제외하지 않음)
            if part.startswith(".") and part != ".":
                return True
        return False
    
    def _should_collect(self, file_path: Path) -> bool:
        """
        파일이 수집 대상인지 확인 (설정 기반 필터링)
        
        Args:
            file_path: 확인할 파일 경로
            
        Returns:
            bool: 수집 대상이면 True
        """
        # 제외할 파일 패턴 확인
        if self._exclude_file_patterns:
            file_name = file_path.name
            relative_path_str = str(file_path.relative_to(self._project_path)) if self._project_path in file_path.parents else str(file_path)
            
            for pattern in self._exclude_file_patterns:
                # 파일명 패턴 매칭
                if fnmatch.fnmatch(file_name, pattern):
                    return False
                # 상대 경로 패턴 매칭 (예: "test/**/*.java")
                if fnmatch.fnmatch(relative_path_str, pattern):
                    return False
        
        # 확장자 추출 (소문자로 변환하여 대소문자 구분 없이 비교)
        file_extension = file_path.suffix.lower()
        
        # 설정에서 지정한 확장자 목록과 비교
        for allowed_extension in self._source_file_types:
            # 대소문자 구분 없이 비교
            if file_extension.lower() == allowed_extension.lower():
                return True
        
        return False
    
    def _normalize_path(self, file_path: Path) -> Path:
        """
        경로를 정규화하여 크로스 플랫폼 호환성 보장
        
        Args:
            file_path: 정규화할 파일 경로
            
        Returns:
            Path: 정규화된 절대 경로
        """
        try:
            # resolve()로 심볼릭 링크를 따라가고 절대 경로로 변환
            resolved_path = file_path.resolve()
            return resolved_path
        except (OSError, RuntimeError):
            # resolve() 실패 시 절대 경로만 반환
            return file_path.absolute()
    
    def _extract_metadata(self, file_path: Path) -> SourceFile:
        """
        파일의 메타데이터를 추출하여 SourceFile 객체 생성
        
        Args:
            file_path: 메타데이터를 추출할 파일 경로
            
        Returns:
            SourceFile: 추출된 메타데이터를 담은 SourceFile 객체
        """
        # 파일 통계 정보 가져오기
        try:
            stat_info = file_path.stat()
        except (OSError, PermissionError) as e:
            raise ValueError(f"파일 정보를 가져올 수 없습니다: {file_path} - {e}")
        
        # 절대 경로
        absolute_path = file_path.resolve()
        
        # 상대 경로 (프로젝트 루트 기준)
        try:
            relative_path = absolute_path.relative_to(self._project_path.resolve())
        except ValueError:
            # 프로젝트 루트 밖에 있는 파일은 절대 경로를 상대 경로로 사용
            relative_path = absolute_path
        
        # 파일명과 확장자
        filename = file_path.name
        extension = file_path.suffix
        
        # 파일 크기
        size = stat_info.st_size
        
        # 수정 시간
        modified_time = datetime.fromtimestamp(stat_info.st_mtime)
        
        # SourceFile 객체 생성
        return SourceFile(
            path=absolute_path,
            relative_path=relative_path,
            filename=filename,
            extension=extension,
            size=size,
            modified_time=modified_time,
            tags=[]  # 초기에는 태그 없음 (나중에 DB Access Analyzer에서 추가)
        )
    
    def get_collected_count(self) -> int:
        """
        현재까지 수집된 파일 개수 반환
        
        Returns:
            int: 수집된 파일 개수
        """
        return len(self._seen_files)
    
    def reset(self) -> None:
        """
        수집 상태 초기화 (중복 제거 Set 초기화)
        """
        self._seen_files.clear()

