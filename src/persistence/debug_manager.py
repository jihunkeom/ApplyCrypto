"""
Debug Manager 모듈

디버깅 관련 파일들을 관리하고 저장합니다.
"""

import logging
import difflib
import shutil
from pathlib import Path
from typing import Optional

from models.modification_plan import ModificationPlan

class DebugManager:
    """
    디버그 관련 데이터 관리자 클래스
    
    주요 기능:
    1. Diff 파일 저장
    2. 디버그 로그 관리
    """
    
    def __init__(self, target_project: Path):
        """
        DebugManager 초기화
        
        Args:
            target_project: 대상 프로젝트 경로
        """
        self.target_project = Path(target_project)
        self.debug_dir = self.target_project / ".applycrypto" / "debug"
        self.diff_dir = self.debug_dir / "diffs"
        self.logger = logging.getLogger(__name__)
        
        self._initialize_debug_directory()
        
    def _initialize_debug_directory(self) -> None:
        """디버그 디렉터리를 초기화(삭제 후 생성)"""
        try:
            if self.debug_dir.exists():
                shutil.rmtree(self.debug_dir)
            self.diff_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.logger.error(f"디버그 디렉터리를 생성할 수 없습니다: {self.diff_dir} - {e}")

    def log_diff(self, plan: ModificationPlan) -> None:
        """
        ModificationPlan의 변경 내용을 Diff 파일로 저장
        
        Args:
            plan: 수정 계획 객체
        """
        if not plan.modified_code:
            return

        try:
            # 파일명 추출
            file_path = Path(plan.file_path)
            filename = file_path.name
            diff_filename = f"{filename}.diff"
            save_path = self.diff_dir / diff_filename
            
            # 중복 파일명 처리
            counter = 1
            while save_path.exists():
                save_path = self.diff_dir / f"{filename}_{counter}.diff"
                counter += 1
            
            # 원본 파일 읽기
            original_path = self.target_project / plan.file_path
            # 만약 plan.file_path가 절대경로라면
            if file_path.is_absolute():
                original_path = file_path
                
            original_content = ""
            if original_path.exists():
                try:
                    with open(original_path, "r", encoding="utf-8") as f:
                        original_content = f.read()
                except Exception as e:
                    self.logger.warning(f"원본 파일을 읽을 수 없습니다: {e}")
            
            # Diff 생성
            original_lines = original_content.splitlines(keepends=True)
            modified_lines = plan.modified_code.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=str(filename),
                tofile=str(filename),
                lineterm=""
            )
            
            diff_content = "".join(diff)
            
            # 만약 diff가 비어있고 modified_code가 비어있지 않으며,
            # modified_code가 diff 형식 같으면 그대로 사용
            if not diff_content.strip() and plan.modified_code:
                 if plan.modified_code.startswith("---") or "diff --git" in plan.modified_code:
                     diff_content = plan.modified_code
            
            # 내용이 있을 때만 저장
            if diff_content.strip():
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(diff_content)
                self.logger.debug(f"Diff 파일 저장 완료: {save_path}")
                
        except Exception as e:
            self.logger.error(f"Diff 파일 저장 중 오류 발생: {e}")
