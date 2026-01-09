"""
ServiceImpl/Biz Code Generator

ServiceImpl/Biz 레이어를 대상으로 코드 수정을 수행하는 CodeGenerator입니다.
TODO: 구현 예정
"""

from typing import List

from models.code_generator import CodeGeneratorInput, CodeGeneratorOutput
from models.modification_plan import ModificationPlan
from models.table_access_info import TableAccessInfo
from models.modification_context import ModificationContext

from ..base_code_generator import BaseCodeGenerator


class ServiceImplOrBizCodeGenerator(BaseCodeGenerator):
    """ServiceImpl/Biz Code 생성기"""

    def generate(self, input_data: CodeGeneratorInput) -> CodeGeneratorOutput:
        """
        입력 데이터를 바탕으로 Code를 생성합니다.

        Args:
            input_data: Code 생성 입력

        Returns:
            CodeGeneratorOutput: LLM 응답 (Code 포함)
        """
        # TODO: ServiceImplOrBiz 특화 로직 구현 예정
        # 현재는 ControllerOrServiceCodeGenerator와 동일한 로직 사용 가능
        raise NotImplementedError(
            "ServiceImplOrBizCodeGenerator.generate()는 아직 구현되지 않았습니다."
        )

    def generate_modification_plans(
        self, table_access_info: TableAccessInfo
    ) -> List[ModificationPlan]:
        """
        수정 계획을 생성합니다.

        Args:
            table_access_info: 테이블 접근 정보

        Returns:
            List[ModificationPlan]: 수정 계획 리스트
        """
        # TODO: ServiceImplOrBiz 특화 로직 구현 예정
        # 현재는 ControllerOrServiceCodeGenerator와 동일한 로직 사용 가능
        raise NotImplementedError(
            "ServiceImplOrBizCodeGenerator.generate_modification_plans()는 아직 구현되지 않았습니다."
        )

