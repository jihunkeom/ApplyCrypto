import logging
import os
import re
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

from modifier.context_generator.base_context_generator import BaseContextGenerator
from models.modification_context import ModificationContext
from models.table_access_info import TableAccessInfo

logger = logging.getLogger("applycrypto.context_generator")

class MybatisContextGenerator(BaseContextGenerator):
    """
    Mybatis Context Generator

    Groups files based on naming conventions for Mybatis architecture.
    Files related to the same entity (Service, Build, Controller, Repository, Dao)
    are grouped into a single context.
    """

    def generate(
        self,
        table_access_info: TableAccessInfo,
    ) -> List[ModificationContext]:
        """
        Generates modification contexts based on Mybatis naming conventions.

        Args:
            table_access_info: Table access information.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []
        project_root = Path(self._config.target_project)

        # 1. Flatten all files from all layers
        all_file_paths = set()
        if table_access_info.layer_files:
            for paths in table_access_info.layer_files.values():
                if paths:
                    all_file_paths.update(paths)

        # 2. Group by Name
        files_by_name: Dict[str, List[str]] = defaultdict(list)
        
        # Regex to capture the Entity Name
        # Convention: [name]Service.java, [name]ServiceImpl.java, 
        # [name]Controller.java, [name]Repository.java, [name]Dao.java
        # pattern = re.compile(r"^(?P<name>.*?)(?:Service|ServiceImpl|Controller|Repository|Dao)\.java$")
        pattern = re.compile(r"^(?P<name>.*?)(?:Service|ServiceImpl|Controller)\.java$")

        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            match = pattern.match(filename)
            if match:
                name = match.group("name")
                files_by_name[name].append(file_path)
            else:
                # If it doesn't match the convention, treat it as its own group
                # using the filename as the key.
                files_by_name["others"].append(file_path)

        logger.info(f"Grouped files into {len(files_by_name)} entity contexts.")

        # 3. Create Contexts
        # Sort by name for deterministic order
        sorted_names = sorted(files_by_name.keys())
        
        for name in sorted_names:
            paths = files_by_name[name]
            if name == "others":
                continue
            
            # Deduplicate paths just in case
            paths = sorted(list(set(paths)))
            
            if not paths:
                continue

            code_snippets = self._read_files(paths, project_root)
            
            if not code_snippets:
                continue

            # Create batches using the entity name as the layer name
            batches = self.create_batches(
                code_snippets=code_snippets,
                table_access_info=table_access_info,
                layer=name 
            )
            all_batches.extend(batches)

        return all_batches
