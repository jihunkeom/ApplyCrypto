"""
Call Chain Processor 모듈

호출 체인(Controller → Service → Repository) 단위로 LLM을 호출하여
가장 적절한 레이어에 암복호화 코드를 삽입하는 모듈입니다.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

from config.config_manager import Configuration
from models.diff_generator import DiffGeneratorInput

from models.table_access_info import TableAccessInfo

# from .diff_generator.call_chain import CallChainDiffGenerator  # TODO: call_chain은 향후 code_generator로 이동 예정
from .error_handler import ErrorHandler
from .llm.llm_factory import create_llm_provider
from .llm.llm_provider import LLMProvider
from .result_tracker import ResultTracker

logger = logging.getLogger("applycrypto.call_chain_processor")


# Bean, VO, DTO 등 제외할 클래스 패턴
EXCLUDED_CLASS_PATTERNS = [
    "Bean",
    "Beans",
    "VO",
    "Vo",
    "DTO",
    "Dto",
    "Entity",
    "Model",
    "Domain",
    "POJO",
    "Pojo",
]


class CallChainProcessor:
    """
    Call Chain Processor 클래스

    호출 체인(Call Chain) 단위로 암복호화 코드를 삽입합니다.
    레이어별 배치 처리 대신, 하나의 호출 체인에 포함된 모든 파일을
    한 번의 LLM 호출로 처리하여 가장 적절한 레이어에 암복호화를 적용합니다.
    """

    def __init__(
        self,
        config: Configuration,
        llm_provider: Optional[LLMProvider] = None,
        project_root: Optional[Path] = None,
    ):
        """
        CallChainProcessor 초기화

        Args:
            config: 설정 객체
            llm_provider: LLM 프로바이더 (선택적, 설정에서 자동 생성)
            project_root: 프로젝트 루트 디렉토리 (선택적)
        """
        self.config = config
        self.project_root = (
            Path(project_root) if project_root else Path(config.target_project)
        )

        # LLM 프로바이더 초기화
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            llm_provider_name = config.llm_provider
            self.llm_provider = create_llm_provider(provider_name=llm_provider_name)

        # DiffGenerator 초기화 (새로운 패턴)
        self.diff_generator = CallChainDiffGenerator(llm_provider=self.llm_provider)

        # 컴포넌트 초기화
        self.error_handler = ErrorHandler(max_retries=config.max_retries)
        self.result_tracker = ResultTracker(self.project_root)

        # 처리 완료된 파일 조합 추적 (단일 실행 내에서만)
        self.processed_combinations: Set[FrozenSet[str]] = set()

        # 이미 수정된 파일 추적 (중복 수정 방지)
        self.modified_files: Set[str] = set()

        logger.info(
            f"CallChainProcessor 초기화 완료: {self.llm_provider.get_provider_name()}"
        )

    def process_all(
        self,
        table_access_info_list: List[TableAccessInfo],
        call_graph_data: Dict[str, Any],
        dry_run: bool = False,
        apply_all: bool = False,
    ) -> Dict[str, Any]:
        """
        모든 테이블의 호출 체인을 처리합니다.

        Args:
            table_access_info_list: 테이블 접근 정보 목록
            call_graph_data: Call Graph 데이터
            dry_run: 시뮬레이션 모드
            apply_all: 사용자 확인 없이 모든 변경사항 자동 적용

        Returns:
            Dict[str, Any]: 처리 결과
        """
        logger.info("Call Chain 처리 시작...")
        self.result_tracker.start_tracking()
        self.processed_combinations.clear()

        # apply_all 플래그 저장 (다른 메서드에서 사용)
        self._apply_all = apply_all

        try:
            # 1. 모든 테이블에서 관련 파일 수집 (Bean, VO, DTO 제외)
            all_target_files = self._collect_all_target_files(table_access_info_list)
            logger.info(f"대상 파일 수집 완료: {len(all_target_files)}개")

            # 2. Call Graph에서 호출 체인 추출
            call_chains = self._extract_call_chains_from_graph(
                call_graph_data, all_target_files
            )
            logger.info(f"호출 체인 추출 완료: {len(call_chains)}개")

            # 3. 중복 제거 및 정렬 (긴 체인 우선)
            unique_chains = self._deduplicate_and_sort_chains(call_chains)
            logger.info(f"고유 체인 수: {len(unique_chains)}개")

            # 4. 테이블/칼럼 정보 취합
            table_column_info = self._aggregate_table_column_info(
                table_access_info_list
            )

            all_modifications = []

            # 5. 각 체인에 대해 처리
            for i, chain_files in enumerate(unique_chains):
                # 체인 구성 파일 분류 및 로깅
                chain_info = self._classify_chain_files(chain_files)
                chain_description = self._format_chain_description(chain_info)

                # 이미 처리한 상위집합이 있으면 스킵
                if self._should_skip(chain_files):
                    logger.info(
                        f"체인 {i + 1}/{len(unique_chains)} 스킵 (상위집합 이미 처리됨): {chain_description}"
                    )
                    continue

                logger.info(
                    f"체인 {i + 1}/{len(unique_chains)} 처리 중: {chain_description}"
                )

                # 단일 체인 처리
                try:
                    plans = self._process_single_chain(
                        chain_files, table_column_info, dry_run
                    )
                    all_modifications.extend(plans)
                except KeyboardInterrupt:
                    logger.info("사용자가 처리를 중단했습니다.")
                    break

                # 처리 완료 기록
                self.processed_combinations.add(chain_files)

            # 결과 추적
            self.result_tracker.end_tracking()

            # 통계 계산
            success_count = sum(
                1 for m in all_modifications if m.get("status") == "success"
            )
            failed_count = sum(
                1 for m in all_modifications if m.get("status") == "failed"
            )
            skipped_count = sum(
                1 for m in all_modifications if m.get("status") == "skipped"
            )

            logger.info(
                f"Call Chain 처리 완료: 성공 {success_count}, 실패 {failed_count}, 스킵 {skipped_count}"
            )

            return {
                "success": True,
                "modifications": all_modifications,
                "statistics": {
                    "total_chains": len(unique_chains),
                    "processed_chains": len(self.processed_combinations),
                    "success": success_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                },
            }

        except Exception as e:
            import traceback

            logger.error(f"Call Chain 처리 실패: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.result_tracker.end_tracking()
            return {
                "success": False,
                "error": str(e),
                "statistics": self.result_tracker.get_statistics(),
            }

    def _normalize_path(self, file_path: str) -> str:
        """
        파일 경로를 정규화합니다.

        Args:
            file_path: 원본 파일 경로

        Returns:
            str: 정규화된 파일 경로
        """
        # Path 객체로 변환 후 절대 경로로 정규화
        return str(Path(file_path).resolve())

    def _classify_chain_files(
        self, chain_files: FrozenSet[str]
    ) -> Dict[str, List[str]]:
        """
        체인 파일들을 레이어별로 분류합니다.

        Args:
            chain_files: 체인에 포함된 파일 경로 집합

        Returns:
            Dict[str, List[str]]: 레이어별 파일 목록
        """
        result = {
            "controller": [],
            "service": [],
            "mapper": [],
            "other": [],
        }

        for file_path in chain_files:
            file_name = Path(file_path).name.lower()

            if "controller" in file_name:
                result["controller"].append(Path(file_path).name)
            elif "service" in file_name:
                result["service"].append(Path(file_path).name)
            elif (
                "mapper" in file_name or "dao" in file_name or "repository" in file_name
            ):
                result["mapper"].append(Path(file_path).name)
            else:
                result["other"].append(Path(file_path).name)

        return result

    def _format_chain_description(self, chain_info: Dict[str, List[str]]) -> str:
        """
        체인 정보를 읽기 쉬운 문자열로 포맷팅합니다.

        Args:
            chain_info: 레이어별 파일 목록

        Returns:
            str: 포맷팅된 체인 설명
        """
        parts = []

        if chain_info["controller"]:
            parts.append(f"Controller: {', '.join(chain_info['controller'])}")
        if chain_info["service"]:
            parts.append(f"Service: {', '.join(chain_info['service'])}")
        if chain_info["mapper"]:
            parts.append(f"Mapper: {', '.join(chain_info['mapper'])}")
        if chain_info["other"]:
            parts.append(f"Other: {', '.join(chain_info['other'])}")

        return " → ".join(parts) if parts else "Unknown chain"

    def _collect_all_target_files(
        self, table_access_info_list: List[TableAccessInfo]
    ) -> Set[str]:
        """
        모든 테이블에서 대상 파일을 수집합니다 (Bean, VO, DTO 제외).

        Args:
            table_access_info_list: 테이블 접근 정보 목록

        Returns:
            Set[str]: 대상 파일 경로 집합 (정규화됨)
        """
        all_files = set()

        for table_info in table_access_info_list:
            # layer_files에서 파일 수집
            for layer_name, files in table_info.layer_files.items():
                for file_path in files:
                    if not self._is_excluded_file(file_path):
                        all_files.add(self._normalize_path(file_path))

            # access_files에서도 수집 (중복 제거됨)
            for file_path in table_info.access_files:
                if not self._is_excluded_file(file_path):
                    all_files.add(self._normalize_path(file_path))

        return all_files

    def _is_excluded_file(self, file_path: str) -> bool:
        """
        Bean, VO, DTO 등 제외 대상 파일인지 확인합니다.

        Args:
            file_path: 파일 경로

        Returns:
            bool: 제외 대상이면 True
        """
        path = Path(file_path)
        filename = path.stem  # 확장자 제외 파일명

        # 제외 패턴 확인
        for pattern in EXCLUDED_CLASS_PATTERNS:
            # 파일명이 패턴으로 끝나거나 패턴을 포함하는 경우
            if filename.endswith(pattern) or pattern.lower() in filename.lower():
                # 단, Controller, Service, Mapper는 제외하지 않음
                if not any(
                    x in filename
                    for x in ["Controller", "Service", "Mapper", "Dao", "Repository"]
                ):
                    logger.debug(f"제외 파일: {file_path} (패턴: {pattern})")
                    return True

        # beans, vo, dto, entity, model, domain 디렉토리 내 파일 제외
        path_lower = file_path.lower()
        excluded_dirs = [
            "/beans/",
            "/vo/",
            "/dto/",
            "/entity/",
            "/model/",
            "/domain/",
            "/pojo/",
            "\\beans\\",
            "\\vo\\",
            "\\dto\\",
            "\\entity\\",
            "\\model\\",
            "\\domain\\",
            "\\pojo\\",
        ]
        for excluded_dir in excluded_dirs:
            if excluded_dir in path_lower:
                logger.debug(f"제외 파일: {file_path} (디렉토리: {excluded_dir})")
                return True

        return False

    def _extract_call_chains_from_graph(
        self, call_graph_data: Dict[str, Any], target_files: Set[str]
    ) -> List[FrozenSet[str]]:
        """
        Call Graph에서 대상 파일과 관련된 호출 체인을 추출합니다.

        Args:
            call_graph_data: Call Graph 데이터
            target_files: 대상 파일 집합

        Returns:
            List[FrozenSet[str]]: 호출 체인 목록 (각 체인은 파일 경로의 frozenset)
        """
        chains = []
        call_trees = call_graph_data.get("call_trees", [])

        # endpoints에서 method_signature → file_path 매핑 생성
        signature_to_file: Dict[str, str] = {}
        for endpoint in call_graph_data.get("endpoints", []):
            sig = endpoint.get("method_signature", "")
            file_path = endpoint.get("file_path", "")
            if sig and file_path:
                signature_to_file[sig] = file_path

        # method_metadata에서도 매핑 추가 (endpoints에 없는 경우 대비)
        for sig, metadata in call_graph_data.get("method_metadata", {}).items():
            if sig not in signature_to_file:
                file_path = metadata.get("file_path", "")
                if file_path:
                    signature_to_file[sig] = file_path

        logger.debug(f"method_signature → file_path 매핑: {len(signature_to_file)}개")

        # 디버깅: target_files와 call_trees 정보 출력
        logger.info(f"call_trees 개수: {len(call_trees)}")
        logger.info(f"target_files (정규화됨): {target_files}")

        # Employee 관련 트리 찾아서 출력
        # emp_trees = [t for t in call_trees if "Emp" in t.get("method_signature", "") or "Employee" in t.get("method_signature", "")]
        # logger.info(f"Employee 관련 트리 개수: {len(emp_trees)}")

        # if emp_trees:
        # first_emp_tree = emp_trees[0]
        # raw_path = first_emp_tree.get('file_path', 'N/A')
        # normalized_path = self._normalize_path(raw_path) if raw_path != 'N/A' else 'N/A'
        # logger.info(f"첫 Employee 트리 file_path (원본): {raw_path}")
        # logger.info(f"첫 Employee 트리 file_path (정규화): {normalized_path}")
        # logger.info(f"정규화된 경로가 target_files에 있는지: {normalized_path in target_files}")

        for tree in call_trees:
            # 각 트리에서 체인 추출
            tree_chains = self._extract_chains_from_tree(
                tree, target_files, signature_to_file
            )
            chains.extend(tree_chains)

        return chains

    def _extract_chains_from_tree(
        self,
        node: Dict[str, Any],
        target_files: Set[str],
        signature_to_file: Dict[str, str],
        current_chain: Optional[Set[str]] = None,
    ) -> List[FrozenSet[str]]:
        """
        트리 노드에서 재귀적으로 호출 체인을 추출합니다.

        Args:
            node: 트리 노드
            target_files: 대상 파일 집합
            signature_to_file: method_signature → file_path 매핑
            current_chain: 현재까지의 체인

        Returns:
            List[FrozenSet[str]]: 추출된 체인 목록
        """
        if current_chain is None:
            current_chain = set()

        chains = []

        # 노드에서 직접 file_path 가져오기 (없으면 signature_to_file 매핑 사용)
        file_path = node.get("file_path", "")
        if not file_path:
            method_signature = node.get("method_signature", "")
            file_path = signature_to_file.get(method_signature, "")

        # 경로 정규화
        if file_path:
            file_path = self._normalize_path(file_path)

        # 현재 노드의 파일이 대상 파일에 포함되고 제외 대상이 아니면 체인에 추가
        if (
            file_path
            and file_path in target_files
            and not self._is_excluded_file(file_path)
        ):
            current_chain = current_chain | {file_path}

        children = node.get("children", [])

        if not children:
            # 리프 노드: 현재 체인이 유효하면 반환
            if len(current_chain) > 0:
                chains.append(frozenset(current_chain))
        else:
            # 자식 노드 탐색
            for child in children:
                child_chains = self._extract_chains_from_tree(
                    child, target_files, signature_to_file, current_chain.copy()
                )
                chains.extend(child_chains)

        return chains

    def _deduplicate_and_sort_chains(
        self, chains: List[FrozenSet[str]]
    ) -> List[FrozenSet[str]]:
        """
        중복 체인을 제거하고 크기 기준 내림차순으로 정렬합니다.

        Args:
            chains: 원본 체인 목록

        Returns:
            List[FrozenSet[str]]: 중복 제거 및 정렬된 체인 목록
        """
        # 중복 제거
        unique_chains = list(set(chains))

        # 크기 기준 내림차순 정렬 (긴 체인 먼저)
        unique_chains.sort(key=lambda x: len(x), reverse=True)

        return unique_chains

    def _should_skip(self, chain_files: FrozenSet[str]) -> bool:
        """
        이 체인이 이미 처리한 체인의 부분집합인지 확인합니다.

        Args:
            chain_files: 확인할 체인

        Returns:
            bool: 스킵해야 하면 True
        """
        for processed in self.processed_combinations:
            if chain_files.issubset(processed):
                return True
        return False

    def _aggregate_table_column_info(
        self, table_access_info_list: List[TableAccessInfo]
    ) -> Dict[str, Any]:
        """
        모든 테이블의 칼럼 정보를 취합합니다.
        config에서 encryption_code를 가져와 병합합니다.

        Args:
            table_access_info_list: 테이블 접근 정보 목록

        Returns:
            Dict[str, Any]: 취합된 테이블/칼럼 정보 (encryption_code 포함)
        """
        tables = []
        for table_info in table_access_info_list:
            # config에서 해당 테이블의 encryption_code 정보 가져오기
            config_columns = {}
            access_tables = self.config.access_tables
            for config_table in access_tables:
                if config_table.table_name.lower() == table_info.table_name.lower():
                    for col in config_table.columns:
                        # Pydantic ColumnDetail 모델 또는 문자열 처리
                        if hasattr(col, "name"):
                            col_name = col.name.lower()
                            encryption_code = getattr(col, "encryption_code", "") or ""
                            if col_name and encryption_code:
                                config_columns[col_name] = encryption_code
                    break

            # columns에 encryption_code 병합
            columns_with_code = []
            for col in table_info.columns:
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    # config에서 encryption_code 가져오기
                    encryption_code = config_columns.get(col_name.lower(), "")
                    columns_with_code.append(
                        {
                            "name": col_name,
                            "encryption_code": encryption_code,
                        }
                    )
                elif isinstance(col, str):
                    encryption_code = config_columns.get(col.lower(), "")
                    columns_with_code.append(
                        {
                            "name": col,
                            "encryption_code": encryption_code,
                        }
                    )

            tables.append(
                {
                    "table_name": table_info.table_name,
                    "columns": columns_with_code,
                }
            )

        return {"tables": tables}

    def _process_single_chain(
        self,
        chain_files: FrozenSet[str],
        table_column_info: Dict[str, Any],
        dry_run: bool,
    ) -> List[Dict[str, Any]]:
        """
        단일 호출 체인을 처리합니다.

        Args:
            chain_files: 체인에 포함된 파일 경로 집합
            table_column_info: 테이블/칼럼 정보
            dry_run: 시뮬레이션 모드

        Returns:
            List[Dict[str, Any]]: 수정 결과 목록
        """
        results = []

        # 파일 내용 읽기 → file_paths 수집
        file_paths: List[str] = []
        files_with_content = []  # 호환성을 위해 유지
        for file_path in sorted(chain_files):
            try:
                path = Path(file_path)
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    file_paths.append(str(path))
                    files_with_content.append(
                        {
                            "path": str(path),
                            "content": content,
                        }
                    )
                else:
                    logger.warning(f"파일이 존재하지 않습니다: {file_path}")
            except Exception as e:
                logger.error(f"파일 읽기 실패: {file_path} - {e}")

        if not file_paths:
            logger.warning("처리할 파일이 없습니다.")
            return results

        # 파일 목록 문자열 생성
        file_list = ", ".join([Path(p).name for p in file_paths])

        # 테이블/칼럼 정보를 텍스트로 포맷팅
        table_info_text = self._format_table_column_info(table_column_info)

        # DiffGeneratorInput 생성
        diff_input = DiffGeneratorInput(
            file_paths=file_paths,
            table_info=table_info_text,
            layer_name="CallChain",
            extra_variables={"file_list": file_list},
        )

        # DiffGenerator로 LLM 호출
        try:
            diff_out = self.diff_generator.generate(diff_input)

            # LLM 응답 파싱 (자체 파싱 메서드 사용)
            parsed_modifications = self._parse_llm_response(diff_out.content)

            # 원본 파일 내용 맵 (path -> content)
            original_content_map = {f["path"]: f["content"] for f in files_with_content}

            # 각 수정 사항 적용
            for mod in parsed_modifications:
                file_path_str = mod.get("file_path", "")
                reason = mod.get("reason", "")
                is_modified = mod.get("modified", False)
                modified_code = mod.get("modified_code", "")

                # 이미 다른 체인에서 수정된 파일인지 확인
                normalized_path = self._normalize_path(file_path_str)
                if normalized_path in self.modified_files:
                    logger.info(
                        f"파일 스킵: {file_path_str} (이미 다른 체인에서 수정됨)"
                    )
                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "skipped",
                            "reason": "Already modified in previous chain",
                        }
                    )
                    continue

                # 수정 필요 없음
                if not is_modified or not modified_code or modified_code.strip() == "":
                    logger.info(f"파일 스킵: {file_path_str} (이유: {reason})")
                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "skipped",
                            "reason": reason,
                        }
                    )
                    continue

                file_path = Path(file_path_str)
                original_content = original_content_map.get(file_path_str, "")

                # 변경사항 없음 확인
                if original_content.strip() == modified_code.strip():
                    logger.info(f"파일 스킵: {file_path_str} (변경사항 없음)")
                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "skipped",
                            "reason": "No actual changes",
                        }
                    )
                    continue

                # 사용자 확인 (dry_run이 아니고 apply_all이 아닌 경우)
                if not dry_run and not getattr(self, "_apply_all", False):
                    self._print_file_diff(file_path, original_content, modified_code)
                    choice = self._get_user_confirmation()

                    if choice == "n":
                        logger.info(f"사용자가 건너뛰기를 선택함: {file_path_str}")
                        results.append(
                            {
                                "file_path": file_path_str,
                                "status": "skipped",
                                "reason": "User skipped",
                            }
                        )
                        continue
                    elif choice == "a":
                        self._apply_all = True
                    elif choice == "q":
                        raise KeyboardInterrupt("사용자가 중단을 선택함")

                # dry_run 모드에서는 diff만 출력
                if dry_run:
                    print(f"\n[미리보기] {file_path.name}")
                    self._print_file_diff(file_path, original_content, modified_code)
                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "preview",
                            "reason": reason,
                        }
                    )
                    continue

                # 파일 수정 적용
                try:
                    # 백업 생성
                    self.error_handler.backup_file(file_path)

                    # 파일 직접 교체
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified_code)

                    logger.info(f"파일 수정 완료: {file_path_str}")

                    # 수정된 파일 추적 (중복 수정 방지)
                    self.modified_files.add(normalized_path)

                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "success",
                            "reason": reason,
                        }
                    )
                except Exception as write_error:
                    logger.error(f"파일 쓰기 실패: {file_path_str} - {write_error}")
                    self.error_handler.restore_file(file_path)
                    results.append(
                        {
                            "file_path": file_path_str,
                            "status": "failed",
                            "error": str(write_error),
                        }
                    )

        except Exception as e:
            logger.error(f"체인 처리 중 오류: {e}")
            for file_info in files_with_content:
                results.append(
                    {
                        "file_path": file_info["path"],
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return results

    def _parse_llm_response(self, content: str) -> List[Dict[str, Any]]:
        """
        LLM 응답을 파싱하여 수정 정보를 추출합니다.
        제어 문자 처리를 포함한 자체 파싱 로직입니다.

        Args:
            response: LLM 응답 딕셔너리

        Returns:
            List[Dict[str, Any]]: 수정 정보 리스트

        Raises:
            Exception: 파싱 실패 시
        """

        # 디버깅: LLM 원본 응답 로깅
        logger.debug(f"LLM content 길이: {len(content) if content else 0}")
        if content:
            # 처음 500자만 로깅 (너무 길면 잘림)
            preview = content[:500] + "..." if len(content) > 500 else content
            logger.debug(f"LLM content 미리보기:\n{preview}")
        else:
            logger.warning(f"LLM 응답 content가 비어있음. 전체 응답: {content}")

        if not content:
            raise Exception("LLM 응답에 content가 없습니다.")

        # JSON 코드 블록 제거
        original_content = content  # 디버깅용 원본 보관
        content = content.strip()

        if content.startswith("```json"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            logger.debug("```json 코드 블록 제거됨")
        elif content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            logger.debug("``` 코드 블록 제거됨")

        # 코드 블록 제거 후 확인
        if not content.strip():
            logger.error(
                f"코드 블록 제거 후 content가 비어있음. 원본:\n{original_content[:1000]}"
            )

        # 제어 문자 및 잘못된 이스케이프 시퀀스 처리 함수
        def escape_control_chars_in_strings(text: str) -> str:
            """JSON 문자열 값 내의 제어 문자와 잘못된 이스케이프 시퀀스를 처리합니다."""
            # 유효한 JSON 이스케이프 문자
            valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

            result = []
            in_string = False
            i = 0

            while i < len(text):
                char = text[i]

                if char == '"' and (i == 0 or text[i - 1] != "\\"):
                    in_string = not in_string
                    result.append(char)
                    i += 1
                    continue

                if in_string and char == "\\":
                    # 백슬래시 처리
                    if i + 1 < len(text):
                        next_char = text[i + 1]

                        if next_char in valid_escapes:
                            # 유효한 이스케이프 시퀀스
                            if next_char == "u":
                                # \uXXXX 형식 확인
                                if i + 5 < len(text) and all(
                                    c in "0123456789abcdefABCDEF"
                                    for c in text[i + 2 : i + 6]
                                ):
                                    result.append(text[i : i + 6])
                                    i += 6
                                    continue
                                else:
                                    # 잘못된 \u 시퀀스 → 이중 이스케이프
                                    result.append("\\\\")
                                    i += 1
                                    continue
                            else:
                                # 유효한 단일 문자 이스케이프
                                result.append(char)
                                result.append(next_char)
                                i += 2
                                continue
                        else:
                            # 잘못된 이스케이프 시퀀스 → 백슬래시 이중 이스케이프
                            result.append("\\\\")
                            i += 1
                            continue
                    else:
                        # 문자열 끝의 백슬래시
                        result.append("\\\\")
                        i += 1
                        continue

                if in_string:
                    # 문자열 내 제어 문자 이스케이프
                    if char == "\n":
                        result.append("\\n")
                    elif char == "\r":
                        result.append("\\r")
                    elif char == "\t":
                        result.append("\\t")
                    elif ord(char) < 32:
                        result.append(f"\\u{ord(char):04x}")
                    else:
                        result.append(char)
                else:
                    result.append(char)

                i += 1

            return "".join(result)

        # 제어 문자 이스케이프 적용
        content = escape_control_chars_in_strings(content)

        # JSON 파싱 시도
        try:
            data = json.loads(content)
            logger.debug("JSON 파싱 성공 (첫 시도)")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 첫 시도 실패: {e}")
            logger.debug(f"파싱 실패한 content 길이: {len(content)}")
            logger.debug(f"파싱 실패한 content 처음 200자:\n{content[:200]}")

            # modifications 부분만 추출 시도
            if "modifications" in content:
                start_idx = content.find('"modifications"')
                if start_idx != -1:
                    brace_start = content.rfind("{", 0, start_idx)
                    if brace_start != -1:
                        brace_count = 0
                        for i in range(brace_start, len(content)):
                            if content[i] == "{":
                                brace_count += 1
                            elif content[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content[brace_start : i + 1]
                                    try:
                                        data = json.loads(json_str)
                                        break
                                    except json.JSONDecodeError:
                                        # 한 번 더 제어 문자 이스케이프 시도
                                        json_str = escape_control_chars_in_strings(
                                            json_str
                                        )
                                        data = json.loads(json_str)
                                        break
                        else:
                            raise Exception(
                                "JSON 파싱 실패: 올바른 JSON 형식이 아닙니다."
                            )
                    else:
                        raise Exception(
                            "JSON 파싱 실패: modifications를 찾을 수 없습니다."
                        )
                else:
                    raise Exception(
                        "JSON 파싱 실패: modifications 키를 찾을 수 없습니다."
                    )
            else:
                raise Exception(f"JSON 파싱 실패: {e}")

        # modifications 추출
        modifications = data.get("modifications", [])
        if not modifications:
            logger.warning("LLM 응답에 modifications가 없거나 비어있습니다.")
            return []

        # 검증 및 정리
        valid_modifications = []
        for mod in modifications:
            if "file_path" not in mod:
                logger.warning("수정 정보에 file_path가 없습니다. 건너뜁니다.")
                continue
            if "reason" not in mod:
                mod["reason"] = "No reason provided"
            if "modified" not in mod:
                # 하위 호환성: unified_diff가 있으면 modified=True로 간주
                mod["modified"] = bool(
                    mod.get("unified_diff", "") or mod.get("modified_code", "")
                )
            if "modified_code" not in mod:
                mod["modified_code"] = ""
            valid_modifications.append(mod)

        logger.info(f"{len(valid_modifications)}개 파일 수정 정보를 파싱했습니다.")
        return valid_modifications

    def _format_table_column_info(self, table_column_info: Dict[str, Any]) -> str:
        """
        테이블/칼럼 정보를 LLM이 이해하기 쉬운 텍스트 형식으로 변환합니다.

        Args:
            table_column_info: 테이블/칼럼 정보 딕셔너리

        Returns:
            str: 포맷팅된 텍스트
        """
        lines = []

        for table in table_column_info.get("tables", []):
            table_name = table.get("table_name", "Unknown")
            lines.append(f"### Table: {table_name}")
            lines.append("")
            lines.append("| Column Name | encryption_code (USE THIS EXACT CODE) |")
            lines.append("|-------------|---------------------------------------|")

            for col in table.get("columns", []):
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    encryption_code = col.get("encryption_code", "")
                    if encryption_code:
                        lines.append(f"| {col_name} | **{encryption_code}** |")
                    else:
                        lines.append(f"| {col_name} | (no encryption_code specified) |")

            lines.append("")

        return "\n".join(lines)

    def _print_file_diff(self, file_path: Path, original: str, modified: str) -> None:
        """
        파일 변경사항을 diff 형식으로 출력합니다.

        Args:
            file_path: 파일 경로
            original: 원본 내용
            modified: 수정된 내용
        """
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
        )

        print(f"\n{'=' * 80}")
        print(f"[Diff] {file_path}")
        print("-" * 80)

        has_diff = False
        for line in diff:
            has_diff = True
            if line.startswith("+") and not line.startswith("+++"):
                print(f"\033[92m{line}\033[0m", end="")  # Green
            elif line.startswith("-") and not line.startswith("---"):
                print(f"\033[91m{line}\033[0m", end="")  # Red
            elif line.startswith("@@"):
                print(f"\033[96m{line}\033[0m", end="")  # Cyan
            else:
                print(line, end="")

        if not has_diff:
            print("(변경사항 없음)")

        print("-" * 80)

    def _get_user_confirmation(self) -> str:
        """
        사용자 확인 입력을 받습니다.

        Returns:
            str: 사용자 선택 ('y', 'n', 'a', 'q')
        """
        while True:
            choice = input(
                "\n이 변경사항을 적용하시겠습니까? [y/n/a/q] "
                "(y:적용, n:건너뛰기, a:모두적용, q:중단): "
            ).lower()
            if choice in ["y", "n", "a", "q"]:
                return choice
            print("잘못된 입력입니다. y, n, a, q 중 하나를 입력하세요.")
