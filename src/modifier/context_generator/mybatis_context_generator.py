import logging
import os
import re
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

from modifier.context_generator.base_context_generator import BaseContextGenerator
from models.modification_context import ModificationContext


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
        layer_files: Dict[str, List[str]],
        table_name: str,
        columns: List[Dict],
    ) -> List[ModificationContext]:
        """
        Generates modification contexts based on Mybatis naming conventions.

        Args:
            layer_files: Dictionary of layer names and file paths.
            table_name: Table name.
            columns: List of columns.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []
        project_root = Path(self._config.target_project)

        # 1. Flatten all files from all layers
        all_file_paths = set()
        if layer_files:
            for paths in layer_files.values():
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

            # Create batches using the entity name as the layer name
            batches = self.create_batches(
                file_paths=paths,
                table_name=table_name,
                columns=columns,
                layer=name 
            )
            all_batches.extend(batches)

        return all_batches
