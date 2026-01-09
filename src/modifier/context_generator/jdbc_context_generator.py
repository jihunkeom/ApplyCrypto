import logging
from pathlib import Path
from typing import Dict, List, Set

from models.modification_context import CodeSnippet, ModificationContext
from models.table_access_info import TableAccessInfo

from modifier.context_generator.base_context_generator import BaseContextGenerator

logger = logging.getLogger("applycrypto.jdbc_context_generator")


class JdbcContextGenerator(BaseContextGenerator):
    """
    JDBC specific Context Generator.
    Groups 'biz' and 'svc' layer files by their parent 'keyword' directory.
    """

    def generate(
        self,
        table_access_info: TableAccessInfo,
    ) -> List[ModificationContext]:
        """
        Generates modification contexts with JDBC-specific grouping logic.

        Args:
            table_access_info: Table access information.

        Returns:
            List[ModificationContext]: The generated batches of contexts.
        """
        all_batches: List[ModificationContext] = []
        project_root = Path(self._config.target_project)

        layer_files = table_access_info.layer_files
        
        # 1. Identify files for 'biz' and 'svc' layers
        target_layers = {"biz", "svc"}
        keyword_groups: Dict[str, Set[str]] = {}
        other_files: Dict[str, List[str]] = {}

        for layer_name, file_paths in layer_files.items():
            if not file_paths:
                continue
                
            if layer_name in target_layers:
                for file_path in file_paths:
                    path_obj = Path(file_path)
                    
                    # Logic to find 'keyword' directory
                    # The user says: "directory name before this layer directory is a keyword directory"
                    # .../<keyword>/<layer>/...
                    
                    # We need to find the part of the path that corresponds to the layer name
                    # and take the parent of that. 
                    # Note: The file path provided in example is absolute or relative. 
                    # We should handle it robustly.
                    
                    parts = path_obj.parts
                    try:
                        # Find the index of the layer name in the path parts (searching from the end)
                        # We search from the end to handle cases where 'biz' or 'svc' might appear earlier in the path
                        idx = -1
                        for i in range(len(parts) - 1, -1, -1):
                            if parts[i] == layer_name:
                                idx = i
                                break
                        
                        if idx > 0:
                            keyword = parts[idx - 1]
                            if keyword not in keyword_groups:
                                keyword_groups[keyword] = set()
                            keyword_groups[keyword].add(file_path)
                        else:
                            # If layer directory not found as expected, fallback to normal handling or log warning
                            # For now, let's treat it as a 'misc' group or separate? 
                            # If we can't group it, maybe just add to a default group or handle separately.
                            # Let's log and add to a generic group? Or maybe just treat as 'other_files'?
                            # But wait, 'biz' and 'svc' are specifically requested to be grouped.
                            # If we can't find the pattern, let's log a warning and add to a standalone group.
                            logger.warning(f"Could not find keyword directory for {file_path} with layer {layer_name}")
                            # Fallback: treat as individual file or maybe group by layer name as default?
                            if layer_name not in other_files:
                                other_files[layer_name] = []
                            other_files[layer_name].append(file_path)

                    except ValueError:
                         logger.warning(f"Layer {layer_name} not found in path {file_path}")
                         if layer_name not in other_files:
                                other_files[layer_name] = []
                         other_files[layer_name].append(file_path)
            else:
                other_files[layer_name] = file_paths

        # 2. Process Keyword Groups (Merged biz and svc)
        for keyword, file_paths_set in keyword_groups.items():
            file_paths = list(file_paths_set)
            logger.info(f"Preparing plan generation for keyword group '{keyword}': {len(file_paths)} files")
            
            code_snippets = self._read_files(file_paths, project_root)
            
            # Create batches for this keyword group
            # We use the keyword as the 'layer' name in the context so it's identifiable
            batches = self.create_batches(
                code_snippets=code_snippets,
                table_access_info=table_access_info,
                layer=keyword, 
            )
            all_batches.extend(batches)

        # 3. Process Other Layers
        for layer_name, file_paths in other_files.items():
             logger.info(f"Preparing plan generation for layer '{layer_name}': {len(file_paths)} files")
             code_snippets = self._read_files(file_paths, project_root)
             batches = self.create_batches(
                code_snippets=code_snippets,
                table_access_info=table_access_info,
                layer=layer_name,
             )
             all_batches.extend(batches)

        return all_batches


