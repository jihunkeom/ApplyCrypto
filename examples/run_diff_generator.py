import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from config.config_manager import Configuration, load_config
from models.diff_generator import DiffGeneratorInput
from models.modification_context import CodeSnippet
from modifier.code_modifier import CodeModifier
from persistence.data_persistence_manager import DataPersistenceManager

# Add src to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

# Load .env
load_dotenv(project_root / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_diff_generator")


def main():
    parser = argparse.ArgumentParser(
        description="Test DiffGenerator with a single file snippet."
    )
    parser.add_argument("file_path", type=str, help="Path to the source file to test")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Force use of mock LLM provider"
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    file_path = Path(args.file_path).resolve()

    if not args.file_path:  # In case arg is empty string
        parser.print_help()
        return

    if not config_path.exists():
        # Try finding config in project root if not found
        if (project_root / args.config).exists():
            config_path = project_root / args.config
        else:
            logger.error(f"Config file not found: {config_path}")
            return

    if not file_path.exists():
        logger.error(f"Source file not found: {file_path}")
        return

    # Load Config
    try:
        config: Configuration = load_config(str(config_path))
        if args.mock:
            config.llm_provider = "mock"
            logger.info("Forcing LLM Provider to 'mock'")

        logger.info(f"Loaded config from {config_path}")
        logger.info(f"Diff Generator Type: {config.diff_gen_type}")
        logger.info(f"Generate Full Source: {config.generate_full_source}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return

    # Create CodeSnippet
    snippet = CodeSnippet(path=str(file_path), content=content)

    # Initialize CodeModifier to get the correct DiffGenerator
    try:
        # We pass project_root from config
        modifier = CodeModifier(config=config, project_root=Path(config.target_project))
        diff_generator = modifier.diff_generator
        logger.info(f"Initialized DiffGenerator: {type(diff_generator).__name__}")

        # Prepare dummy table info or relevant info
        # Check if the file matches any table in config?
        # For simplicity, we create a generic context or try to match if possible.
        # But for testing, a fixed dummy context usually suffices unless the prompt heavily relies on it.
        # Let's see if we can get a matching table from config.

        target_table_name = "UNKNOWN_TABLE"
        target_columns = []

        # Simple heuristic: if file name matches a table name (ignoring case)
        file_stem = file_path.stem.lower()
        if config.access_tables:
            for table in config.access_tables:
                if (
                    table.table_name.lower() in file_stem
                    or file_stem in table.table_name.lower()
                ):
                    target_table_name = table.table_name
                    target_columns = table.columns
                    break

            # If still not found, just use the first one from config to have valid schema
            if target_table_name == "UNKNOWN_TABLE" and len(config.access_tables) > 0:
                first_table = config.access_tables[0]
                target_table_name = first_table.table_name
                target_columns = first_table.columns

        # Format columns for JSON
        formatted_columns = []
        for col in target_columns:
            if hasattr(col, "dict"):
                formatted_columns.append(col.dict())
            else:
                formatted_columns.append(col)  # it might be string or dict

        dummy_table_info = {
            "table_name": target_table_name,
            "columns": formatted_columns,
        }
        table_info_str = json.dumps(dummy_table_info, indent=2, default=str)

        logger.info(f"Using Table Info: {target_table_name}")

        extra_vars = {"file_count": 1}

        input_data = DiffGeneratorInput(
            code_snippets=[snippet],
            table_info=table_info_str,
            layer_name="service",  # Default assumption
            extra_variables=extra_vars,
        )

        logger.info("Generating diff...")
        diff_out = diff_generator.generate(input_data)

        # Prepare log content
        log_content = []
        log_content.append("\n" + "=" * 50)
        log_content.append("GENERATED RESPONSE")
        log_content.append("=" * 50)
        log_content.append(f"Tokens Used: {diff_out.tokens_used}")
        log_content.append("-" * 20)

        # Handle \n in text for pretty printing
        if diff_out.parsed_out:
            formatted_lines = []
            for mod in diff_out.parsed_out:
                formatted_lines.append("-" * 30)
                for k, v in mod.items():
                    if k == "unified_diff":
                        formatted_lines.append(f"{k}:")
                        formatted_lines.append(str(v))
                    else:
                        formatted_lines.append(f"{k}: {v}")
                formatted_lines.append("-" * 30)
            content_str = "\n".join(formatted_lines)
        else:
            content_str = diff_out.content
            if content_str:
                content_str = content_str.replace("\\n", "\n")

        log_content.append(content_str)
        log_content.append("=" * 50 + "\n")

        # Save to log file
        try:
            target_project_path = Path(config.target_project)
            persistence = DataPersistenceManager(
                target_project=target_project_path,
                output_dir=target_project_path / ".applycrypto",
            )
            persistence.save_text_file("\n".join(log_content), "diff_generator_log.txt")
            logger.info(
                f"Log file generated at {target_project_path / '.applycrypto/diff_generator_log.txt'}"
            )
        except Exception as e:
            logger.error(f"Failed to generate log file: {e}")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
