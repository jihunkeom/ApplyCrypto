import os
import shutil
import sys
from pathlib import Path

from config.config_manager import load_config
from persistence.data_persistence_manager import DataPersistenceManager

# Add src to sys.path to import project modules
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))


def main():
    # 1. Load Configuration
    config_file_path = project_root / "config.json"
    if not config_file_path.exists():
        print(f"Error: Config file not found at {config_file_path}")
        return

    try:
        config = load_config(str(config_file_path))
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    target_project = Path(config.target_project)
    if not target_project.exists():
        print(f"Error: Target project path does not exist: {target_project}")
        return

    # 2. Setup Output Directory using DataPersistenceManager
    # The user requested to use the persistent manager
    output_path = target_project / ".applycrypto" / "extracted_modified"

    # Initialize DataPersistenceManager with the custom output directory
    # This will ensure the directory exists
    persistence_manager = DataPersistenceManager(
        target_project=target_project, output_dir=output_path, enable_cache=False
    )

    print(f"Extraction directory prepared at: {persistence_manager.output_dir}")

    # 3. Find and Copy Files
    count = 0
    for root, dirs, files in os.walk(target_project):
        for file in files:
            if file.endswith(".backup"):
                backup_file_path = Path(root) / file
                # Assuming original file has the same name without .backup
                original_file_path = backup_file_path.with_suffix("")

                if original_file_path.exists():
                    # Calculate relative path to maintain structure
                    rel_path = original_file_path.relative_to(target_project)

                    # Create destination directory structure
                    dest_dir = persistence_manager.output_dir / rel_path.parent
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Copy files
                    shutil.copy2(original_file_path, dest_dir / original_file_path.name)
                    shutil.copy2(backup_file_path, dest_dir / backup_file_path.name)

                    print(f"Extracted: {rel_path} and {rel_path}.backup")
                    count += 1

    print(f"\nExtraction complete. Total sets of files extracted: {count}")


if __name__ == "__main__":
    main()
