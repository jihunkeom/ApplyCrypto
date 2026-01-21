"""
MybatisCCS Batch Context Generator

CCS 배치 프로그램용 Context Generator입니다.
BAT.java 파일을 수정 대상으로, BATVO.java 파일을 context로 포함합니다.

특징:
    - 수정 대상 (file_paths): BAT.java
    - 참조용 context (context_files): BATVO.java
    - 레이어: bat, batvo
"""

import logging
from typing import Dict, List

from models.modification_context import ModificationContext

from .base_context_generator import BaseContextGenerator

logger = logging.getLogger("applycrypto.context_generator")


class MybatisCCSBatchContextGenerator(BaseContextGenerator):
    """
    CCS 배치 프로그램용 Context Generator

    BAT.java 파일을 수정 대상으로 하고,
    BATVO.java 파일을 참조용 context로 포함합니다.
    """

    # CCS 배치 레이어명 매핑 (소문자로 통일)
    LAYER_NAME_MAPPING = {
        "bat": "bat",
        "batvo": "batvo",
    }

    def _normalize_layer_files(
        self, layer_files: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        레이어명을 소문자로 정규화합니다.

        Args:
            layer_files: 레이어별 파일 경로 딕셔너리

        Returns:
            Dict[str, List[str]]: 정규화된 레이어 파일 딕셔너리
        """
        normalized: Dict[str, List[str]] = {}

        for layer_name, files in layer_files.items():
            normalized_name = self.LAYER_NAME_MAPPING.get(
                layer_name.lower(), layer_name.lower()
            )
            if normalized_name not in normalized:
                normalized[normalized_name] = []
            for f in files:
                if f not in normalized[normalized_name]:
                    normalized[normalized_name].append(f)

        logger.debug(
            f"레이어 정규화: {list(layer_files.keys())} -> {list(normalized.keys())}"
        )
        return normalized

    def generate(
        self,
        layer_files: Dict[str, List[str]],
        table_name: str,
        columns: List[Dict],
    ) -> List[ModificationContext]:
        """
        CCS 배치용 context 생성

        수정 대상 (file_paths): BAT.java
        참조용 context (context_files): BATVO.java

        Args:
            layer_files: 레이어별 파일 경로 딕셔너리 (bat, batvo)
            table_name: 테이블명
            columns: 컬럼 목록

        Returns:
            List[ModificationContext]: 생성된 context 목록
        """
        normalized = self._normalize_layer_files(layer_files)

        logger.info(
            f"MybatisCCSBatchContextGenerator: 레이어 정규화 완료 "
            f"({list(layer_files.keys())} -> {list(normalized.keys())})"
        )

        all_batches: List[ModificationContext] = []

        # BAT 파일 (수정 대상)
        bat_files = normalized.get("bat", [])

        # BATVO 파일 (context용)
        batvo_files = normalized.get("batvo", [])

        if not bat_files:
            logger.info("BAT Layer 파일이 없습니다.")
            return all_batches

        logger.info(
            f"CCS 배치 context 생성: "
            f"{len(bat_files)}개 BAT 파일, "
            f"{len(batvo_files)}개 BATVO context"
        )

        # 각 BAT 파일에 대해 batch 생성
        for bat_file in bat_files:
            # BAT.java를 수정 대상으로, BATVO.java를 context로
            batches = self.create_batches(
                file_paths=[bat_file],
                table_name=table_name,
                columns=columns,
                layer="bat",
                context_files=batvo_files,
            )
            all_batches.extend(batches)

            logger.debug(
                f"BAT 파일 '{bat_file}': "
                f"context로 {len(batvo_files)}개 BATVO 파일 포함"
            )

        return all_batches
