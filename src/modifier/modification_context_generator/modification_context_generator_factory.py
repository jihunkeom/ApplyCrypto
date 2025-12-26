from config.config_manager import Configuration
from modifier.diff_generator.base_diff_generator import BaseDiffGenerator

from .base_modification_context_generator import BaseModificationContextGenerator
from .parallel_modification_context_generator import (
    ParallelModificationContextGenerator,
)


class ModificationContextGeneratorFactory:
    """
    Factory class for creating ModificationContextGenerator instances.
    """

    @staticmethod
    def create(
        config: Configuration,
        diff_generator: BaseDiffGenerator,
        generator_type: str = "parallel",
    ) -> BaseModificationContextGenerator:
        """
        Creates a ModificationContextGenerator instance based on the specified type.

        Args:
            config: Configuration object.
            diff_generator: BaseDiffGenerator instance.
            generator_type: Type of generator to create (default: "parallel").

        Returns:
            BaseModificationContextGenerator: An instance of a modification context generator.

        Raises:
            ValueError: If the generator_type is unknown.
        """
        if generator_type == "parallel":
            return ParallelModificationContextGenerator(config, diff_generator)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
