import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from config.config_manager import Configuration
from models.diff_generator import DiffGeneratorInput
from models.modification_context import CodeSnippet, ModificationContext
from models.table_access_info import TableAccessInfo
from modifier.diff_generator.base_diff_generator import BaseDiffGenerator

# Don't remove the following comment.. this is for example format of TableAccessInfo
# "layer_files": {
#   "Repository": [
#     "C:\\Users\\hypot\\Works\\book-ssm\\src\\com\\mybatis\\beans\\Employee.java",
#     "C:\\Users\\hypot\\Works\\book-ssm\\src\\com\\mybatis\\dao\\EmployeeMapper.java"
#   ],
#   "service": [
#     "C:\\Users\\hypot\\Works\\book-ssm\\src\\com\\mybatis\\service\\EmployeeService.java"
#   ],
#   "controller": [
#     "C:\\Users\\hypot\\Works\\book-ssm\\src\\com\\mybatis\\controller\\EmpController.java",
#     "C:\\Users\\hypot\\Works\\book-ssm\\src\\com\\mybatis\\controller\\EmployeeController.java"
#   ]
# },

logger = logging.getLogger("applycrypto.modification_context_generator")


class BaseModificationContextGenerator(ABC):
    """
    Base Modification Context Generator

    Generates batches of ModificationContext from TableAccessInfo,
    handling file reading and token limits.
    """

    def __init__(
        self,
        config: Configuration,
        diff_generator: BaseDiffGenerator,
    ):
        self.config = config
        self.diff_generator = diff_generator
        self.project_root = Path(config.target_project)

    @abstractmethod
    def generate(self, table_access_info: TableAccessInfo) -> List[ModificationContext]:
        """
        Generates modification contexts. Failed files are logged.

        Args:
            table_access_info: Table access information containing file paths.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        pass

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
        formatted_table_info = self._format_table_info(table_access_info)
        max_tokens = self.config.max_tokens_per_batch

        input_empty_data = DiffGeneratorInput(
            code_snippets=[], table_info=formatted_table_info, layer_name=layer
        )

        # Create empty prompt and calculate tokens
        empty_prompt = self.diff_generator.create_prompt(input_empty_data)
        empty_num_tokens = self.diff_generator.calculate_token_size(empty_prompt)

        # Calculate separator tokens ("\n\n")
        separator_tokens = self.diff_generator.calculate_token_size("\n\n")

        current_batch_tokens = empty_num_tokens

        for snippet in code_snippets:
            # Format snippet and calculate tokens
            snippet_formatted = (
                f"=== File Path (Absolute): {snippet.path} ===\n{snippet.content}"
            )
            snippet_tokens = self.diff_generator.calculate_token_size(snippet_formatted)

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

    def _format_table_info(self, table_access_info: TableAccessInfo) -> str:
        """
        Formats table info as JSON string.
        """
        table_info = {
            "table_name": table_access_info.table_name,
            "columns": table_access_info.columns,
        }

        return json.dumps(table_info, indent=2, ensure_ascii=False)
