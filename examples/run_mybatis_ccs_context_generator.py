"""
MybatisCCS / MybatisCCSBatch Context Generator 검증 스크립트

실제 타겟 프로젝트의 table_access_info.json을 읽어서
Context Generator가 어떤 파일들을 context로 묶어서 LLM에게 전달하는지 검증합니다.

지원하는 sql_wrapping_type:
    - mybatis_ccs: CCS 온라인용 (CTL, SVCImpl, DQM 레이어)
    - mybatis_ccs_batch: CCS 배치용 (BAT, BATVO 레이어)

사용법:
    python examples/run_mybatis_ccs_context_generator.py --config /path/to/config.json

옵션:
    --config: config.json 파일 경로 (필수)
    --table: 특정 테이블만 검증 (선택, 기본값: 전체)
    --output: 결과를 저장할 파일 경로 (선택)
    --verbose: 상세 로그 출력 (선택)
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

# Add src to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from config.config_manager import Configuration, load_config
from models.table_access_info import TableAccessInfo
from models.modification_context import ModificationContext
from modifier.context_generator.mybatis_ccs_context_generator import (
    MybatisCCSContextGenerator,
)
from modifier.context_generator.mybatis_ccs_batch_context_generator import (
    MybatisCCSBatchContextGenerator,
)
from modifier.context_generator.base_context_generator import BaseContextGenerator


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("run_mybatis_ccs_context_generator")


def load_table_access_info(config: Configuration) -> List[TableAccessInfo]:
    """
    타겟 프로젝트의 table_access_info.json 파일 로드

    Args:
        config: Configuration 객체

    Returns:
        List[TableAccessInfo]: 테이블 접근 정보 목록
    """
    target_project = Path(config.target_project)
    json_path = target_project / ".applycrypto" / "results" / "table_access_info.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"table_access_info.json을 찾을 수 없습니다: {json_path}\n"
            "'analyze' 명령어를 먼저 실행하세요."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON 데이터를 TableAccessInfo 객체로 변환
    table_access_info_list = []
    for item in data:
        if isinstance(item, dict):
            table_access_info_list.append(TableAccessInfo.from_dict(item))

    return table_access_info_list


def create_mock_code_generator(config: Configuration) -> MagicMock:
    """
    Mock CodeGenerator 생성 (실제 LLM 호출 없이 테스트)

    Args:
        config: Configuration 객체

    Returns:
        MagicMock: Mock CodeGenerator
    """
    mock_code_generator = MagicMock()
    # 토큰 계산은 작은 값을 반환하여 배치 분할이 쉽게 일어나지 않도록 설정
    mock_code_generator.calculate_token_size.return_value = 100
    mock_code_generator.create_prompt.return_value = "Mock Prompt"
    return mock_code_generator


def create_context_generator(
    config: Configuration,
    mock_code_generator: MagicMock,
) -> BaseContextGenerator:
    """
    sql_wrapping_type에 따라 적절한 Context Generator 생성

    Args:
        config: Configuration 객체
        mock_code_generator: Mock CodeGenerator

    Returns:
        BaseContextGenerator: Context Generator 인스턴스
    """
    if config.sql_wrapping_type == "mybatis_ccs_batch":
        return MybatisCCSBatchContextGenerator(config, mock_code_generator)
    elif config.sql_wrapping_type == "mybatis_ccs":
        return MybatisCCSContextGenerator(config, mock_code_generator)
    else:
        raise ValueError(
            f"지원하지 않는 sql_wrapping_type: {config.sql_wrapping_type}. "
            f"가능한 값: mybatis_ccs, mybatis_ccs_batch"
        )


def run_context_generator(
    config: Configuration,
    table_access_info: TableAccessInfo,
    logger: logging.Logger,
) -> List[ModificationContext]:
    """
    Context Generator 실행

    Args:
        config: Configuration 객체
        table_access_info: 테이블 접근 정보
        logger: 로거

    Returns:
        List[ModificationContext]: 생성된 context 목록
    """
    # Mock CodeGenerator 생성
    mock_code_generator = create_mock_code_generator(config)

    # sql_wrapping_type에 따라 적절한 Context Generator 생성
    generator = create_context_generator(config, mock_code_generator)
    logger.info(f"사용 중인 Context Generator: {generator.__class__.__name__}")

    # Context 생성
    contexts = generator.generate(
        layer_files=table_access_info.layer_files,
        table_name=table_access_info.table_name,
        columns=table_access_info.columns,
    )

    return contexts


def print_context_summary(
    contexts: List[ModificationContext],
    table_name: str,
    logger: logging.Logger,
):
    """
    생성된 context 요약 출력

    Args:
        contexts: ModificationContext 목록
        table_name: 테이블명
        logger: 로거
    """
    print("\n" + "=" * 70)
    print(f"테이블: {table_name}")
    print(f"생성된 Context 수: {len(contexts)}")
    print("=" * 70)

    for i, ctx in enumerate(contexts, 1):
        print(f"\n[Context {i}] - Layer/Group: {ctx.layer or '(없음)'}")
        print(f"  파일 수: {ctx.file_count}")
        print(f"  Context Files (VO 등): {len(ctx.context_files) if ctx.context_files else 0}개")
        print("  포함된 파일들:")
        for file_path in ctx.file_paths:
            # 파일명만 추출하여 출력 (경로가 길 수 있으므로)
            filename = Path(file_path).name
            # 경로에서 레이어 정보 추출
            layer_hint = extract_layer_from_path(file_path)
            print(f"    - [{layer_hint}] {filename}")

        if ctx.context_files:
            print("  Context Files (수정 대상 아님):")
            for file_path in ctx.context_files:
                filename = Path(file_path).name
                print(f"    - [VO] {filename}")

        print("-" * 70)


def extract_layer_from_path(file_path: str) -> str:
    """
    파일 경로에서 레이어 힌트 추출

    Args:
        file_path: 파일 경로

    Returns:
        str: 레이어 힌트 (ctl, biz, svc, dqm, bat, batvo 등)
    """
    path_lower = file_path.lower()
    filename_lower = Path(file_path).name.lower()

    # CCS 배치 레이어 (BAT, BATVO)
    if "/bat/" in path_lower or "\\bat\\" in path_lower:
        if "/batvo/" in path_lower or "\\batvo\\" in path_lower:
            return "BATVO"
        if filename_lower.endswith("batvo.java"):
            return "BATVO"
        if filename_lower.endswith("bat.java"):
            return "BAT"
        return "BAT"

    # 파일명 기반 배치 레이어 감지
    if filename_lower.endswith("batvo.java"):
        return "BATVO"
    if filename_lower.endswith("bat.java") and not filename_lower.endswith("_sql.xml"):
        return "BAT"

    # CCS 온라인 레이어 (CTL, SVC, BIZ, DQM)
    if "/ctl/" in path_lower or "\\ctl\\" in path_lower:
        return "CTL"
    elif "/biz/" in path_lower or "\\biz\\" in path_lower:
        if "/bvo/" in path_lower or "\\bvo\\" in path_lower:
            return "BVO"
        return "BIZ"
    elif "/svc/" in path_lower or "\\svc\\" in path_lower:
        if "/impl/" in path_lower or "\\impl\\" in path_lower:
            return "SVCImpl"
        elif "/svo/" in path_lower or "\\svo\\" in path_lower:
            return "SVO"
        return "SVC"
    elif "/dqm/" in path_lower or "\\dqm\\" in path_lower:
        if "/dvo/" in path_lower or "\\dvo\\" in path_lower:
            return "DVO"
        return "DQM"
    else:
        return "OTHER"


def save_results(
    results: List[dict],
    output_path: Path,
    logger: logging.Logger,
):
    """
    결과를 JSON 파일로 저장

    Args:
        results: 결과 데이터
        output_path: 출력 파일 경로
        logger: 로거
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"결과가 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MybatisCCS Context Generator 검증 스크립트"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config.json 파일 경로",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="특정 테이블만 검증 (선택)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    # 로깅 설정
    logger = setup_logging(args.verbose)

    # Config 로드
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logger.error(f"Config 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    try:
        config = load_config(str(config_path))
        logger.info(f"Config 로드 완료: {config_path}")
        logger.info(f"  - framework_type: {config.framework_type}")
        logger.info(f"  - sql_wrapping_type: {config.sql_wrapping_type}")
    except Exception as e:
        logger.error(f"Config 로드 실패: {e}")
        sys.exit(1)

    # sql_wrapping_type 검증
    supported_types = ["mybatis_ccs", "mybatis_ccs_batch"]
    if config.sql_wrapping_type not in supported_types:
        logger.error(
            f"이 스크립트는 {supported_types}만 지원합니다. "
            f"현재 sql_wrapping_type: {config.sql_wrapping_type}"
        )
        sys.exit(1)

    # table_access_info.json 로드
    try:
        table_access_info_list = load_table_access_info(config)
        logger.info(f"table_access_info.json 로드 완료: {len(table_access_info_list)}개 테이블")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"table_access_info.json 로드 실패: {e}")
        sys.exit(1)

    # 특정 테이블만 필터링
    if args.table:
        table_access_info_list = [
            t for t in table_access_info_list
            if t.table_name.lower() == args.table.lower()
        ]
        if not table_access_info_list:
            logger.error(f"테이블 '{args.table}'을 찾을 수 없습니다.")
            sys.exit(1)

    # 결과 수집
    all_results = []

    # 각 테이블에 대해 Context Generator 실행
    for table_info in table_access_info_list:
        logger.info(f"테이블 '{table_info.table_name}' 처리 중...")

        # layer_files가 비어있으면 스킵
        if not table_info.layer_files:
            logger.warning(f"테이블 '{table_info.table_name}'에 layer_files가 없습니다. 스킵합니다.")
            continue

        # layer_files 정보 출력
        print(f"\n{'=' * 70}")
        print(f"테이블: {table_info.table_name}")
        print(f"입력 layer_files:")
        for layer, files in table_info.layer_files.items():
            print(f"  {layer}: {len(files)}개 파일")
            if args.verbose:
                for f in files[:5]:  # 최대 5개만 출력
                    print(f"    - {Path(f).name}")
                if len(files) > 5:
                    print(f"    ... 외 {len(files) - 5}개")

        # Context 생성
        try:
            contexts = run_context_generator(config, table_info, logger)
        except Exception as e:
            logger.error(f"Context 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 결과 출력
        print_context_summary(contexts, table_info.table_name, logger)

        # 결과 수집
        table_result = {
            "table_name": table_info.table_name,
            "input_layer_files": {
                layer: len(files) for layer, files in table_info.layer_files.items()
            },
            "contexts": [
                {
                    "layer": ctx.layer,
                    "file_count": ctx.file_count,
                    "file_paths": ctx.file_paths,
                    "context_files": ctx.context_files or [],
                }
                for ctx in contexts
            ],
        }
        all_results.append(table_result)

    # 전체 요약 출력
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    print(f"처리된 테이블 수: {len(all_results)}")
    total_contexts = sum(len(r["contexts"]) for r in all_results)
    print(f"생성된 총 Context 수: {total_contexts}")

    # 결과 저장
    if args.output:
        output_path = Path(args.output).resolve()
        save_results(all_results, output_path, logger)

    print("\n검증 완료!")


if __name__ == "__main__":
    main()
