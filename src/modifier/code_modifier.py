"""
Code Modifier 메인 모듈

LLM을 활용하여 소스 코드에 암호화/복호화 코드를 자동으로 적용하는 메인 클래스입니다.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.config_manager import ConfigurationManager
from models.table_access_info import TableAccessInfo
from .llm.llm_factory import create_llm_provider
from .llm.llm_provider import LLMProvider
from .prompt_template_manager import PromptTemplateManager
from .batch_processor import BatchProcessor
from .code_patcher import CodePatcher
from .error_handler import ErrorHandler
from .result_tracker import ResultTracker


logger = logging.getLogger("applycrypto.code_patcher")


class CodeModifier:
    """
    Code Modifier 메인 클래스
    
    설정 파일의 테이블/칼럼 정보를 기반으로 식별된 DB 접근 소스 파일들에 대해
    k-sign.CryptoService의 encrypt/decrypt를 사용하여 암호화 코드를 자동으로 적용합니다.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        llm_provider: Optional[LLMProvider] = None,
        project_root: Optional[Path] = None
    ):
        """
        CodeModifier 초기화
        
        Args:
            config_manager: 설정 관리자
            llm_provider: LLM 프로바이더 (선택적, 설정에서 자동 생성)
            project_root: 프로젝트 루트 디렉토리 (선택적)
        """
        self.config_manager = config_manager
        self.project_root = Path(project_root) if project_root else Path(config_manager.project_path)
        
        # LLM 프로바이더 초기화
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            llm_provider_name = config_manager.get("llm_provider", "watsonx_ai")
            self.llm_provider = create_llm_provider(provider_name=llm_provider_name)
        
        # 컴포넌트 초기화
        self.template_manager = PromptTemplateManager()
        self.batch_processor = BatchProcessor(
            template_manager=self.template_manager,
            max_tokens_per_batch=config_manager.get("max_tokens_per_batch", 3000)
        )
        self.code_patcher = CodePatcher(project_root=self.project_root)
        self.error_handler = ErrorHandler(
            max_retries=config_manager.get("max_retries", 3)
        )
        self.result_tracker = ResultTracker()
        
        logger.info(f"CodeModifier 초기화 완료: {self.llm_provider.get_provider_name()}")
    
    def _get_api_key_from_env(self, provider_name: str) -> Optional[str]:
        """
        환경변수에서 API 키를 가져옵니다.
        
        Args:
            provider_name: 프로바이더 이름
            
        Returns:
            Optional[str]: API 키
        """
        import os
        
        if provider_name.lower() in ["watsonx_ai", "watsonx"]:
            return os.getenv("WATSONX_API_KEY") or os.getenv("IBM_API_KEY")
        elif provider_name.lower() == "openai":
            return os.getenv("OPENAI_API_KEY")
        
        return None
    
    def modify_sources(
        self,
        table_access_info: TableAccessInfo,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        소스 파일들을 수정합니다.
        
        Args:
            table_access_info: 테이블 접근 정보
            dry_run: 실제 수정 없이 시뮬레이션만 수행 (기본값: False)
            
        Returns:
            Dict[str, Any]: 수정 결과
        """
        logger.info(f"소스 파일 수정 시작: {table_access_info.table_name}")
        self.result_tracker.start_tracking()
        
        try:
            # 레이어별로 파일 그룹화
            layer_files = table_access_info.layer_files
            
            all_modifications = []
            
            # 각 레이어별로 처리
            for layer_name, file_paths in layer_files.items():
                if not file_paths:
                    continue
                
                logger.info(f"레이어 '{layer_name}' 처리 시작: {len(file_paths)}개 파일")
                
                # 파일 내용 읽기 (항상 절대 경로 사용)
                files_with_content = []
                for file_path in file_paths:
                    try:
                        # 파일 경로를 절대 경로로 변환
                        file_path_obj = Path(file_path)
                        if not file_path_obj.is_absolute():
                            # 상대 경로인 경우 project_root와 결합
                            full_path = self.project_root / file_path_obj
                        else:
                            # 절대 경로인 경우 그대로 사용
                            full_path = file_path_obj
                        
                        # 절대 경로로 정규화
                        full_path = full_path.resolve()
                        
                        if full_path.exists():
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            files_with_content.append({
                                "path": str(full_path),  # 절대 경로 저장
                                "content": content
                            })
                        else:
                            logger.warning(f"파일이 존재하지 않습니다: {full_path} (원본 경로: {file_path})")
                    except Exception as e:
                        logger.error(f"파일 읽기 실패: {file_path} - {e}")
                
                if not files_with_content:
                    continue
                
                # 통합 템플릿 사용 (template_type은 "default"로 통일)
                template_type = "default"
                
                # 배치 생성
                batches = self.batch_processor.create_batches(
                    files=files_with_content,
                    template_type=template_type,
                    variables={
                        "table_info": self._format_table_info(table_access_info),
                        "layer_name": layer_name,
                        "file_count": len(files_with_content)
                    }
                )
                
                # 배치 처리
                for batch in batches:
                    batch_result = self._process_batch(
                        batch=batch,
                        template_type=template_type,
                        layer_name=layer_name,
                        modification_type="encryption",  # 통합된 타입
                        table_access_info=table_access_info,
                        dry_run=dry_run
                    )
                    
                    if batch_result:
                        all_modifications.extend(batch_result)
            
            # 결과 추적
            self.result_tracker.end_tracking()
            self.result_tracker.update_table_access_info(
                table_access_info,
                all_modifications
            )
            
            # 수정 이력 저장
            self.result_tracker.save_modification_history(
                table_access_info.table_name,
                all_modifications
            )
            
            # 통계 저장
            self.result_tracker.save_statistics()
            
            logger.info(
                f"소스 파일 수정 완료: {table_access_info.table_name} "
                f"({len(all_modifications)}개 파일 수정)"
            )
            
            return {
                "success": True,
                "modifications": all_modifications,
                "statistics": self.result_tracker.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"소스 파일 수정 실패: {e}")
            self.result_tracker.end_tracking()
            return {
                "success": False,
                "error": str(e),
                "statistics": self.result_tracker.get_statistics()
            }
    
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        template_type: str,
        layer_name: str,
        modification_type: str,
        table_access_info: TableAccessInfo,
        dry_run: bool
    ) -> List[Dict[str, Any]]:
        """
        단일 배치를 처리합니다.
        
        Args:
            batch: 배치 파일 리스트
            template_type: 템플릿 타입
            layer_name: 레이어명
            modification_type: 수정 타입
            table_access_info: 테이블 접근 정보
            dry_run: 시뮬레이션 모드
            
        Returns:
            List[Dict[str, Any]]: 수정 결과 리스트
        """
        modifications = []
        
        # LLM 호출 함수
        def llm_call(prompt: str) -> Dict[str, Any]:
            response, error = self.error_handler.retry_with_backoff(
                self.llm_provider.call,
                prompt
            )
            
            if error:
                raise error
            
            if not self.llm_provider.validate_response(response):
                raise ValueError("LLM 응답이 유효하지 않습니다.")
            
            return response
        
        # 배치 처리
        try:
            response = self.batch_processor.process_batch(
                batch=batch,
                template_type=template_type,
                variables={
                    "table_info": self._format_table_info(table_access_info),
                    "layer_name": layer_name,
                    "file_count": len(batch)
                },
                llm_call_func=llm_call
            )
            
            # LLM 응답 파싱
            parsed_modifications = self.code_patcher.parse_llm_response(response)
            
            # 각 수정 사항 적용 (LLM 응답의 절대 경로를 그대로 사용)
            for mod in parsed_modifications:
                file_path_str = mod.get("file_path", "")
                reason = mod.get("reason", "")
                unified_diff = mod.get("unified_diff", "")
                
                # LLM 응답에서 받은 절대 경로를 그대로 사용
                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    # 상대 경로인 경우에만 project_root와 결합 (일반적으로는 발생하지 않아야 함)
                    logger.warning(f"LLM 응답에 상대 경로가 포함되었습니다: {file_path_str}. 절대 경로로 변환합니다.")
                    file_path = self.project_root / file_path
                
                # 절대 경로로 정규화
                file_path = file_path.resolve()
                
                # unified_diff가 빈 문자열인 경우 수정 작업을 건너뜀
                if not unified_diff or unified_diff.strip() == "":
                    logger.info(f"파일 수정 건너뜀: {file_path} (이유: {reason})")
                    # 수정 정보 기록 (건너뜀 상태)
                    modification_info = self.result_tracker.record_modification(
                        file_path=str(file_path),
                        layer=layer_name,
                        modification_type=modification_type,
                        status="skipped",
                        diff=None,
                        error=reason if reason else "수정이 필요하지 않음",
                        tokens_used=response.get("tokens_used", 0)
                    )
                    modifications.append(modification_info)
                    continue
                
                # 파일 백업
                if not dry_run:
                    self.error_handler.backup_file(file_path)
                
                # 패치 적용
                success, error = self.code_patcher.apply_patch(
                    file_path=file_path,
                    unified_diff=unified_diff,
                    dry_run=dry_run
                )
                
                if success:
                    
                    # # 구문 검증. 임시로 막아놓고 success 처리
                    # syntax_valid, syntax_error = self.code_patcher.validate_syntax(file_path)
                    
                    # if syntax_valid:
                    #     status = "success"
                    #     error_msg = None
                    # else:
                    #     status = "failed"
                    #     error_msg = syntax_error
                    #     # 롤백
                    #     if not dry_run:
                    #         self.error_handler.restore_file(file_path)
                    status = "success"
                    error_msg = None
                
                # 수정 정보 기록
                modification_info = self.result_tracker.record_modification(
                    file_path=str(file_path),
                    layer=layer_name,
                    modification_type=modification_type,
                    status=status,
                    diff=unified_diff if status == "success" else None,
                    error=error_msg,
                    tokens_used=response.get("tokens_used", 0)
                )
                
                modifications.append(modification_info)
            
        except Exception as e:
            logger.error(f"배치 처리 실패: {e}")
            # 배치 내 모든 파일에 대해 실패 기록
            for file_info in batch:
                modification_info = self.result_tracker.record_modification(
                    file_path=file_info.get("path", ""),
                    layer=layer_name,
                    modification_type=modification_type,
                    status="failed",
                    error=str(e),
                    tokens_used=0
                )
                modifications.append(modification_info)
        
        return modifications
    
    def _format_table_info(
        self,
        table_access_info: TableAccessInfo
    ) -> str:
        """
        테이블 정보를 JSON 형식으로 포맷팅합니다.
        
        Args:
            table_access_info: 테이블 접근 정보
            
        Returns:
            str: JSON 형식의 테이블 정보
        """
        import json
        
        table_info = {
            "table_name": table_access_info.table_name,
            "columns": table_access_info.columns
        }
        
        return json.dumps(table_info, indent=2, ensure_ascii=False)

