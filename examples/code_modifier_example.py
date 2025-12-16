"""
Code Modifier 예제

Code Modifier의 수정 계획 생성 기능을 테스트하는 예제입니다.
이 예제는 실제 LLM을 호출하지 않고 Mock Provider를 사용하여 동작을 검증합니다.
"""

import json
import logging
from pathlib import Path

from config.config_manager import ConfigurationManager
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo
from modifier.code_modifier import CodeModifier
from modifier.llm.mock_llm_provider import MockLLMProvider

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("code_modifier_example")


def main():
    # 현재 스크립트의 위치를 기준으로 프로젝트 루트 설정 (examples/.. -> project_root)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 임시 디렉터리 생성 (테스트용 소스 파일만 여기 생성)
    temp_dir = Path("temp_code_modifier_example")
    temp_dir.mkdir(exist_ok=True)

    try:
        print("=" * 60)
        print("Code Modifier 예제 - 수정 계획 생성")
        print("=" * 60)

        # 1. 테스트 환경 설정

        # 설정 파일 경로 (프로젝트 루트의 config.json 사용)
        config_file = project_root / "config.json"

        if not config_file.exists():
            print(f"오류: 설정 파일을 찾을 수 없습니다: {config_file}")
            return

        # ConfigManager 초기화
        config_manager = ConfigurationManager(str(config_file))

        # 테스트용 소스 파일 생성
        target_file = temp_dir / "UserDAO.java"
        target_file_content = """
package com.example.dao;

public class UserDAO {
    public void save(User user) {
        // DB 저장 로직
    }
}
"""
        target_file.write_text(target_file_content, encoding="utf-8")

        # 절대 경로 얻기
        target_file_absolute = target_file.resolve()

        # 2. Mock LLM 설정

        # 예상되는 LLM 응답 (CodePatcher가 파싱할 수 있는 형식)
        mock_modifications = {
            "modifications": [
                {
                    "file_path": str(target_file_absolute),
                    "unified_diff": """--- a/UserDAO.java
+++ b/UserDAO.java
@@ -5,5 +5,6 @@
 public class UserDAO {
     public void save(User user) {
+        // 암호화 적용
         // DB 저장 로직
     }
 }""",
                    "reason": "암호화 적용을 위해 코드 수정",
                }
            ],
            # MockLLMProvider의 응답 구조에 맞게 토큰 사용량 등은 provider 내부에서 처리하거나
            # response content string에 포함될 필요는 없음 (MockLLMProvider는 content 문자열만 받음)
        }

        # MockLLMProvider는 문자열 형태의 응답을 받습니다.
        mock_response_str = json.dumps(mock_modifications)

        mock_llm = MockLLMProvider(mock_response=mock_response_str)

        # 3. CodeModifier 초기화
        modifier = CodeModifier(
            config_manager=config_manager, llm_provider=mock_llm, project_root=temp_dir
        )

        # 4. 입력 데이터(TableAccessInfo) 생성
        table_info = TableAccessInfo(
            table_name="USERS",
            columns=[{"name": "PASSWORD", "new_column": False}],
            access_files=[str(target_file_absolute)],
            query_type="INSERT",
            layer="DAO",
            layer_files={"DAO": [str(target_file_absolute)]},
        )

        # 5. 수정 계획 생성 실행
        print(f"\n[{table_info.table_name}] 테이블에 대한 수정 계획 생성 중...")
        plans = modifier.generate_modification_plans(table_info)

        # 6. 결과 검증 및 출력
        print("\n" + "=" * 60)
        print(f"생성된 계획 수: {len(plans)}")
        print("=" * 60)

        for i, plan in enumerate(plans, 1):
            print(f"\n[계획 #{i}]")
            print(f"  파일 경로: {plan.file_path}")
            print(f"  레이어: {plan.layer_name}")
            print(f"  상태: {plan.status}")
            print(f"  토큰 사용량: {plan.tokens_used}")
            print(f"  이유: {plan.reason}")

            if isinstance(plan, ModificationPlan):
                print("  검증: ModificationPlan 객체임")
            else:
                print("  검증 실패: ModificationPlan 객체가 아님")

            if plan.unified_diff:
                print("\n  [Diff 내용]")
                for line in plan.unified_diff.splitlines():
                    print(f"    {line}")

    finally:
        # 정리: 임시 디렉터리 삭제
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\n임시 디렉터리 삭제됨")


if __name__ == "__main__":
    main()
