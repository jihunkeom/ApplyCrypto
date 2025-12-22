"""
LLM SQL Extractor Example

This example demonstrates how to use LLMSQLExtractor to extract SQL from Java SourceFile.
"""

from pathlib import Path

from analyzer.llm_sql_extractor import LLMSQLExtractor
from models.source_file import SourceFile


def main():
    try:
        # 3. Initialize LLMSQLExtractor
        # Using MyBatis strategy as an example, but LLM extraction is likely strategy-agnostic or adaptable
        llm_provider = "watsonx_ai"
        sql_wrapping_type = "mybatis"

        print(f"Initializing LLMSQLExtractor with provider: {llm_provider}")
        extractor = LLMSQLExtractor(
            sql_wrapping_type=sql_wrapping_type, llm_provider_name=llm_provider
        )

        # Create a dummy file with SQL for testing
        dummy_file_path = Path("src/test/DummyDao.java")
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dummy_file_path, "w", encoding="utf-8") as f:
            f.write("""
package com.example.test;

public class DummyDao {
    public void getUser(int id) {
        String sql = "SELECT * FROM users WHERE id = " + id;
        System.out.println(sql);
    }
    
    public void updateUser(String name, int id) {
        String query = "UPDATE users SET name = '" + name + "' WHERE id = " + id;
        execute(query);
    }
}
""")
        print(f"Created dummy file at {dummy_file_path}")

        from datetime import datetime

        # Add dummy file to processing list
        dummy_source_file = SourceFile(
            path=dummy_file_path.absolute(),
            relative_path=Path("src/test/DummyDao.java"),
            filename="DummyDao.java",
            extension=".java",
            size=0,
            modified_time=datetime.now(),
            tags=[],
        )

        files_to_process = [dummy_source_file]
        print("Extracting SQL from dummy file...")

        results = extractor.extract_from_files(files_to_process)

        # 5. Save results to JSON file
        output_file = "llm_extraction_results.json"
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")

        # 6. Display results
        print(f"\nExtraction Results ({len(results)} files produced output):")
        for result in results:
            print(f"\nFile: {result.file.path}")
            if result.sql_queries:
                for query in result.sql_queries:
                    print(f"  - ID: {query.id}")
                    print(f"    Type: {query.query_type}")
                    print(
                        f"    SQL: {query.sql[:100]}..."
                        if len(query.sql) > 100
                        else f"    SQL: {query.sql}"
                    )
            else:
                print("  No SQL extracted.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
