# SQL Extraction Template

Please extract SQL queries from the following source code.

## Source Code
{{ source_code }}

## Output Format
Return a JSON list of objects with the following structure:
[
  {
    "id": "{query_identifier}",
    "query_type": "{SELECT|INSERT|UPDATE|DELETE}",
    "sql": "{extracted_sql_statement}",
  }
]
