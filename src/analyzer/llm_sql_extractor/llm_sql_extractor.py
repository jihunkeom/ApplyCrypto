"""
LLM SQL Extractor module

Extracts SQL using LLM based on strategy.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from models.source_file import SourceFile
from models.sql_extraction_output import ExtractedSQLQuery, SQLExtractionOutput
from modifier.batch_processor import BatchProcessor
from modifier.llm.llm_factory import create_llm_provider


class LLMSQLExtractor:
    """
    LLM SQL Extractor class

    Extracts SQL using LLM strategy.
    """

    def __init__(
        self,
        sql_wrapping_type: str,
        llm_provider_name: str = "watsonx_ai",
    ):
        """
        Initialize LLMSQLExtractor

        Args:
            sql_wrapping_type: SQL Wrapping Type (mybatis, jpa, jdbc)
            llm_provider_name: LLM Provider Name (default: watsonx_ai)
        """
        self.sql_wrapping_type = sql_wrapping_type
        self.logger = logging.getLogger(__name__)
        self.llm_provider = create_llm_provider(llm_provider_name)

        # Load template
        self.template_path = Path(__file__).parent / "template.md"
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                self.template_content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to load template from {self.template_path}: {e}")
            self.template_content = ""

    def extract_from_files(
        self, source_files: List[SourceFile]
    ) -> List[SQLExtractionOutput]:
        """
        Extract SQL queries from source files using LLM.

        Args:
            source_files: List of source files to analyze

        Returns:
            List[SQLExtractionOutput]: List of extracted SQL query information
        """
        if not self.template_content:
            self.logger.error("Template content is empty. Skipping extraction.")
            return []

        # Filter files based on sql_wrapping_type
        target_files = self._filter_target_files(source_files)

        # Use BatchProcessor for parallel processing
        processor = BatchProcessor()
        results = processor.process_items_parallel(
            target_files, self._process_single_file, desc="LLM SQL Extraction"
        )

        # Filter out None results (failed processing)
        return [r for r in results if r is not None]

    def _filter_target_files(self, source_files: List[SourceFile]) -> List[SourceFile]:
        """
        Filter source files based on sql_wrapping_type.
        """
        target_files = []
        for f in source_files:
            name_lower = f.filename.lower()

            if self.sql_wrapping_type == "mybatis":
                # Filter for *mapper.xml
                if f.extension == ".xml" and name_lower.endswith("mapper.xml"):
                    target_files.append(f)
            else:
                # Filter for *dao.java, *service.java, or *repository.java
                if f.extension == ".java" and (
                    name_lower.endswith("dao.java")
                    or name_lower.endswith("service.java")
                    or name_lower.endswith("repository.java")
                ):
                    target_files.append(f)

        return target_files

    def _process_single_file(
        self, java_file: SourceFile
    ) -> Optional[SQLExtractionOutput]:
        """
        Process a single file to extract SQL.
        """
        try:
            # Read source code
            with open(java_file.path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Prepare prompt
            prompt = self.template_content.replace("{{ source_code }}", source_code)

            # Call LLM
            response_dict = self.llm_provider.call(prompt)
            response = response_dict.get("content", "")

            # Parse JSON response
            extracted_data = self._parse_llm_response(response)

            if extracted_data:
                sql_queries = []
                for item in extracted_data:
                    try:
                        sql_query = ExtractedSQLQuery(
                            id=item.get("id", ""),
                            query_type=item.get("query_type", "SELECT"),
                            sql=item.get("sql", ""),
                            strategy_specific=item.get("strategy_specific", {}),
                        )
                        sql_queries.append(sql_query)
                    except Exception as e:
                        self.logger.warning(
                            f"Error creating ExtractedSQLQuery item: {e}"
                        )

                if sql_queries:
                    return SQLExtractionOutput(file=java_file, sql_queries=sql_queries)

        except Exception as e:
            self.logger.error(f"Error processing file {java_file.path}: {e}")

        return None

    def _parse_llm_response(self, response: str) -> List[dict]:
        """
        Parse LLM response to JSON list.
        Handles markdown code blocks and potential noise.
        """
        try:
            # Try to find JSON block if wrapped in markdown
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find array brackets if simply returned as text
                array_match = re.search(r"\[.*\]", response, re.DOTALL)
                if array_match:
                    json_str = array_match.group(0)
                else:
                    json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse LLM response as JSON: {e}\nResponse: {response}"
            )
            return []
