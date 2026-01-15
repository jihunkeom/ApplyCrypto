import logging
import os
from typing import List, Dict, Optional
from pathlib import Path

from modifier.context_generator.base_context_generator import BaseContextGenerator
from models.modification_context import ModificationContext
from parser.java_ast_parser import JavaASTParser

logger = logging.getLogger("applycrypto.context_generator")


class MybatisContextGenerator(BaseContextGenerator):
    """
    Mybatis Context Generator

    Groups files based on import relationships between Controller and Service layers.
    """

    def _match_import_to_file_path(
        self, import_statement: str, target_files: List[str]
    ) -> Optional[str]:
        """
        import 문과 일치하는 파일 경로를 찾습니다.

        Args:
            import_statement: import 문 (예: "com.example.service.UserService")
            target_files: 대상 파일 목록

        Returns:
            Optional[str]: 일치하는 파일 경로, 없으면 None
        """
        # import 문에서 클래스명 추출 (마지막 부분)
        class_name = import_statement.split(".")[-1]

        # 대상 파일 중에서 클래스명과 일치하는 파일 찾기
        for file_path in target_files:
            file_name = os.path.basename(file_path)
            # 확장자 제거
            file_stem = os.path.splitext(file_name)[0]
            if file_stem == class_name:
                return file_path

        return None

    def generate(
        self,
        layer_files: Dict[str, List[str]],
        table_name: str,
        columns: List[Dict],
    ) -> List[ModificationContext]:
        """
        Generates modification contexts based on import relationships between Controller, Service, and Repository layers.

        Args:
            layer_files: Dictionary of layer names and file paths.
            table_name: Table name.
            columns: List of columns.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []

        # Controller Layer와 Service Layer, Repository Layer 파일 목록 가져오기
        controller_files = layer_files.get("controller", [])
        service_files = layer_files.get("service", [])
        repository_files = layer_files.get("repository", [])
        repository_files = [
            x for x in repository_files if x.lower().endswith("vo.java")
        ]

        if not controller_files:
            logger.info("Controller Layer 파일이 없습니다.")
            return all_batches

        # JavaASTParser 인스턴스 생성
        java_parser = JavaASTParser()

        # Service import들을 사전에 수집하여, Controller가 직접 import 하지 않은 Service에서만 import된 VO도 포함되도록 한다.
        # (와일드카드/무import 케이스는 이번 변경 범위에서 제외)
        all_service_imports = set()
        for service_file in service_files:
            try:
                service_path = Path(service_file)
                service_tree, service_error = java_parser.parse_file(service_path)
                if service_error:
                    logger.debug(
                        f"Service 파일 파싱 실패(사전 수집): {service_file} - {service_error}"
                    )
                    continue

                service_classes = java_parser.extract_class_info(
                    service_tree, service_path
                )
                if not service_classes:
                    continue

                service_class = None
                for cls in service_classes:
                    if cls.access_modifier == "public":
                        service_class = cls
                        break
                if service_class is None:
                    service_class = service_classes[0]

                all_service_imports.update(service_class.imports)
                print(service_class.imports)
            except Exception as e:
                logger.warning(
                    f"Service 파일 처리 실패(사전 수집): {service_file} - {e}"
                )
                continue

        # Controller별 파일 묶음을 저장할 딕셔너리
        # key: controller 파일 이름(stem), value: 관련 파일 경로 리스트
        file_groups: Dict[str, List[str]] = {}
        # Controller별 VO 파일 (context용, 수정 대상 아님)
        context_file_groups: Dict[str, List[str]] = {}

        # 1단계: Controller 파일별로 파일 묶음 생성
        for controller_file in controller_files:
            file_path = Path(controller_file)
            controller_key = file_path.stem
            file_group_paths: List[str] = [
                controller_file
            ]  # 최소한 controller 파일은 포함
            vo_group_paths: List[str] = []  # VO 파일은 별도로 관리 (context용)

            # 2. JavaASTParser를 사용하여 class 정보를 가져오고,
            # 그 중에서 public class의 class 이름을 가져온다
            try:
                # 파일 파싱
                tree, error = java_parser.parse_file(file_path)
                if error:
                    logger.debug(
                        f"Controller 파일 파싱 실패: {controller_file} - {error}"
                    )
                    # 파싱 실패해도 controller 파일만 포함된 그룹으로 저장
                    file_groups[controller_key] = file_group_paths
                    continue

                # 클래스 정보 추출
                classes = java_parser.extract_class_info(tree, file_path)

                if not classes:
                    logger.debug(
                        f"Controller 파일에서 클래스 정보를 찾을 수 없습니다: {controller_file}"
                    )
                    # 클래스 정보가 없어도 controller 파일만 포함된 그룹으로 저장
                    file_groups[controller_key] = file_group_paths
                    continue

                # public class 찾기
                controller_class = None
                for cls in classes:
                    if cls.access_modifier == "public":
                        controller_class = cls
                        break

                # public class가 없으면 첫 번째 클래스 사용
                if controller_class is None:
                    controller_class = classes[0]
                    logger.debug(
                        f"Controller 파일에서 public class를 찾을 수 없어 첫 번째 클래스를 사용합니다: {controller_file}"
                    )

                # 3. class 정보에서 import 목록을 가져온다
                controller_imports = set(controller_class.imports)

                # 4. Service Layer에 수집되어 있는 파일 목록 중에서
                # Controller의 import문에 존재하는 파일만 선택해서 파일 그룹에 추가한다
                matched_service_files: List[str] = []
                for import_stmt in controller_imports:
                    matched_file = self._match_import_to_file_path(
                        import_stmt, service_files
                    )
                    if matched_file:
                        matched_service_files.append(matched_file)
                        if matched_file not in file_group_paths:
                            file_group_paths.append(matched_file)

                # 5. 3번에서 포함된 Service Layer의 파일 각각에 대해 JavaASTParser를 사용하여
                # class 정보를 가져오고 그 중에서 public class를 선택하여 import 문을 가져와서 import 목록을 보강한다
                # 기존: matched_service_files(=Controller가 import한 Service)만 보강
                # 변경: service_files 전체에서 수집한 import를 합쳐 service-only import VO도 포함
                enhanced_imports = set(controller_imports)
                enhanced_imports.update(all_service_imports)

                # 6. Repository Layer에 수집되어 있는 파일 목록 중에서
                # import 목록에 포함되어 있는 파일들만 선택해서 VO 그룹에 추가한다 (context용)
                for import_stmt in enhanced_imports:
                    matched_file = self._match_import_to_file_path(
                        import_stmt, repository_files
                    )
                    if matched_file and matched_file not in vo_group_paths:
                        vo_group_paths.append(matched_file)

            except Exception as e:
                logger.warning(f"Controller 파일 처리 실패: {controller_file} - {e}")
                # 에러 발생해도 controller 파일만 포함된 그룹으로 저장
                file_groups[controller_key] = file_group_paths
                continue

            # 파일 그룹 저장
            file_groups[controller_key] = file_group_paths  # sorted(file_group_paths)
            context_file_groups[controller_key] = vo_group_paths

        # 2단계: 각 파일 묶음에 대해 create_batches 실행
        for controller_key, file_group_paths in file_groups.items():
            if len(file_group_paths) == 0:
                continue

            # VO 파일은 context_files로 전달
            vo_files = context_file_groups.get(controller_key, [])

            logger.info(
                f"Controller '{controller_key}'에 대한 파일 그룹 생성: {len(file_group_paths)}개 파일, {len(vo_files)}개 VO 파일 (context)"
            )

            batches = self.create_batches(
                file_paths=file_group_paths,
                table_name=table_name,
                columns=columns,
                layer="",  # 빈 문자열 기본값
                context_files=vo_files,  # VO 파일은 context로 전달
            )
            all_batches.extend(batches)

        return all_batches
