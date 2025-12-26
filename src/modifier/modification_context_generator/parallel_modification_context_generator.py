import logging
from pathlib import Path
from typing import List

from models.modification_context import CodeSnippet, ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.modification_context_generator.base_modification_context_generator import (
    BaseModificationContextGenerator,
)

logger = logging.getLogger("applycrypto.modification_context_generator")


class ParallelModificationContextGenerator(BaseModificationContextGenerator):
    """
    Parallel Modification Context Generator

    Generates batches of ModificationContext from TableAccessInfo,
    handling file reading and token limits.
    """

    def generate(self, table_access_info: TableAccessInfo) -> List[ModificationContext]:
        """
        Generates modification contexts. Failed files are logged.

        Args:
            table_access_info: Table access information containing file paths.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []

        # Layer-wise file grouping
        layer_files = table_access_info.layer_files

        for layer_name, file_paths in layer_files.items():
            if not file_paths:
                continue

            logger.info(
                f"Preparing plan generation for layer '{layer_name}': {len(file_paths)} files"
            )

            code_snippets: List[CodeSnippet] = []
            for file_path in file_paths:
                try:
                    # Convert file path to absolute
                    file_path_obj = Path(file_path)
                    if not file_path_obj.is_absolute():
                        full_path = self.project_root / file_path_obj
                    else:
                        full_path = file_path_obj

                    # Resolve absolute path
                    full_path = full_path.resolve()

                    if full_path.exists():
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        code_snippets.append(
                            CodeSnippet(
                                path=str(full_path),
                                content=content,
                            )
                        )
                    else:
                        logger.warning(
                            f"File not found: {full_path} (Original path: {file_path})"
                        )

                except Exception as e:
                    logger.error(f"Failed to read file: {file_path} - {e}")

            # Create batches considering token limits
            modification_context_batches = self.create_batches(
                code_snippets=code_snippets,
                table_access_info=table_access_info,
                layer=layer_name,
            )

            all_batches.extend(modification_context_batches)

        return all_batches
