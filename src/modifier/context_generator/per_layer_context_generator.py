import logging
from pathlib import Path
from typing import List

from models.modification_context import ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.context_generator.base_context_generator import BaseContextGenerator

logger = logging.getLogger("applycrypto.per_layer_context_generator")


class PerLayerContextGenerator(BaseContextGenerator):
    """
    Per Layer Context Generator

    Generates batches of ModificationContext from TableAccessInfo,
    grouping files by their layer definition.
    """

    def generate(
        self,
        table_access_info: TableAccessInfo,
    ) -> List[ModificationContext]:
        """
        Generates modification contexts. Failed files are logged.

        Args:
            table_access_info: Table access information containing file paths.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []
        project_root = Path(self._config.target_project)

        # Layer-wise file grouping
        layer_files = table_access_info.layer_files

        for layer_name, file_paths in layer_files.items():
            if not file_paths:
                continue

            logger.info(
                f"Preparing plan generation for layer '{layer_name}': {len(file_paths)} files"
            )

            code_snippets = self._read_files(file_paths, project_root)

            # Create batches considering token limits
            modification_context_batches = self.create_batches(
                code_snippets=code_snippets,
                table_access_info=table_access_info,
                layer=layer_name,
            )

            all_batches.extend(modification_context_batches)

        return all_batches
