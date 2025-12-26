from typing import Optional

from config.config_manager import Configuration
from modifier.llm.llm_provider import LLMProvider

from .base_diff_generator import BaseDiffGenerator


class DiffGeneratorFactory:
    """DiffGenerator 생성을 위한 팩토리 클래스"""

    @staticmethod
    def create(
        config: Configuration, llm_provider: Optional[LLMProvider]
    ) -> BaseDiffGenerator:
        """
        설정에 따라 적절한 DiffGenerator 인스턴스를 생성합니다.

        Args:
            config: 설정 객체
            llm_provider: LLM 프로바이더

        Returns:
            BaseDiffGenerator: 생성된 DiffGenerator 인스턴스

        Raises:
            ValueError: 지원하지 않는 diff_gen_type인 경우
        """
        diff_type = config.diff_gen_type

        if diff_type == "mybatis_service":
            from .mybatis_service.mybatis_service_diff_generator import (
                MyBatisServiceDiffGenerator,
            )

            return MyBatisServiceDiffGenerator(llm_provider=llm_provider, config=config)

        elif diff_type == "mybatis_dao":
            from .mybatis_dao.mybatis_dao_diff_generator import MyBatisDaoDiffGenerator

            return MyBatisDaoDiffGenerator(llm_provider=llm_provider, config=config)

        elif diff_type == "mybatis_typehandler":
            from .mybatis_typehandler.mybatis_typehandler_diff_generator import (
                MyBatisTypeHandlerDiffGenerator,
            )

            return MyBatisTypeHandlerDiffGenerator(
                llm_provider=llm_provider, config=config
            )

        elif diff_type == "call_chain":
            from .call_chain.call_chain_diff_generator import CallChainDiffGenerator

            return CallChainDiffGenerator(llm_provider=llm_provider, config=config)

        else:
            raise ValueError(f"Unknown diff_generator type: {diff_type}")
