import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from config.config_manager import Configuration
from models.code_generator import CodeGeneratorInput
from models.modification_context import CodeSnippet, ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.code_generator.base_code_generator import BaseCodeGenerator

logger = logging.getLogger("applycrypto.context_generator")


class BaseContextGenerator(ABC):
    """
    Base Context Generator
    """

    def __init__(self, config: Configuration, code_generator: BaseCodeGenerator):
        self._config = config
        self._code_generator = code_generator

    @abstractmethod
    def generate(
        self,
        table_access_info: TableAccessInfo,
    ) -> List[ModificationContext]:
        """
        Generates modification contexts.

        Args:
            table_access_info: Table access information containing file paths.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        raise NotImplementedError

    def create_batches(
        self,
        code_snippets: List[CodeSnippet],
        table_access_info: TableAccessInfo,
        layer: str,
    ) -> List[ModificationContext]:
        """
        Splits file list into batches based on token size.

        Args:
            code_snippets: List of code snippets.
            table_access_info: Table access information.
            layer: Layer name.

        Returns:
            List[ModificationContext]: List of split modification contexts.
        """
        if not code_snippets:
            return []

        batches: List[ModificationContext] = []
        current_snippets: List[CodeSnippet] = []

        # Prepare basic info
        table_info = {
            "table_name": table_access_info.table_name,
            "columns": table_access_info.columns,
        }
        formatted_table_info = json.dumps(table_info, indent=2, ensure_ascii=False)
        max_tokens = self._config.max_tokens_per_batch

        input_empty_data = CodeGeneratorInput(
            code_snippets=[], table_info=formatted_table_info, layer_name=layer
        )

        # Create empty prompt and calculate tokens
        empty_prompt = self._code_generator.create_prompt(input_empty_data)
        empty_num_tokens = self._code_generator.calculate_token_size(empty_prompt)

        # Calculate separator tokens ("\n\n")
        separator_tokens = self._code_generator.calculate_token_size("\n\n")

        current_batch_tokens = empty_num_tokens

        for snippet in code_snippets:
            # Format snippet and calculate tokens
            snippet_formatted = (
                f"=== File Path (Absolute): {snippet.path} ===\n{snippet.content}"
            )
            snippet_tokens = self._code_generator.calculate_token_size(snippet_formatted)

            # Add separator tokens if not the first snippet
            tokens_to_add = snippet_tokens
            if current_snippets:
                tokens_to_add += separator_tokens

            if current_snippets and (current_batch_tokens + tokens_to_add) > max_tokens:
                # Save current batch and start new one
                batches.append(
                    ModificationContext(
                        code_snippets=current_snippets,
                        table_access_info=table_access_info,
                        file_count=len(current_snippets),
                        layer=layer,
                    )
                )
                current_snippets = [snippet]
                current_batch_tokens = empty_num_tokens + snippet_tokens
            else:
                # Add to current batch
                current_snippets.append(snippet)
                current_batch_tokens += tokens_to_add

        # Add last batch
        if current_snippets:
            batches.append(
                ModificationContext(
                    code_snippets=current_snippets,
                    table_access_info=table_access_info,
                    file_count=len(current_snippets),
                    layer=layer,
                )
            )

        logger.info(
            f"Split {len(code_snippets)} files into {len(batches)} batches (ModificationContext)."
        )
        return batches

    def _read_files(self, file_paths: List[str], project_root: Path) -> List[CodeSnippet]:
        """
        Reads files and returns a list of CodeSnippets.

        Args:
            file_paths: List of file paths to read.
            project_root: The root directory of the project for resolving relative paths.

        Returns:
            List[CodeSnippet]: List of read code snippets.
        """
        code_snippets: List[CodeSnippet] = []
        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.is_absolute():
                    full_path = project_root / file_path_obj
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

        return code_snippets
