# Data Mapping Extraction (Phase 1) - CCS Version (resultMap 기반)

## Role

You are an expert in analyzing DQM.xml resultMap mappings, Java VO classes, and SQL queries.
Your task is to extract **query-based data mapping information** from the provided SQL queries.

**Important**:

- This is an **extraction and analysis** task only
- You are NOT modifying any code
- Field mappings from `resultMap` tags are already provided with each query - use them directly
- You are providing structured information to help the next phase (Planning) understand how data flows between Java and DB

---

## ★★★ Target Table Information (CRITICAL) ★★★

**IMPORTANT: Focus ONLY on the target table and columns specified below.**

{{ table_info }}

**table_info.columns Structure:**

Each column in `table_info.columns` may contain:
- `name`: Column name (always present) - **this is the DB column name to match in SQL**
- `new_column`: Whether this is a new column (boolean)
- `column_type`: Type of sensitive data - `"name"`, `"dob"`, or `"rrn"` (optional)
- `encryption_code`: Direct policy constant - e.g., `"SliEncryptionConstants.Policy.NAME"` (optional)

**CRITICAL: Every column listed in table_info.columns IS an encryption target.**
Do NOT skip any column. Even if the column name seems unusual (e.g., `gvnm`, `aenam`), it must be included in `crypto_fields`.

---

## SQL Queries with Relevant Field Mappings

Below are the SQL queries to analyze. Each query includes **only the relevant field mappings** for the target columns listed above. Use these mappings to determine the exact Java field names.

{{ sql_queries_with_mappings }}

---

## Output Format (★★★ JSON ONLY ★★★)

**CRITICAL OUTPUT RULES:**

1. Output **ONLY** valid JSON - no explanations, no markdown, no comments
2. Do **NOT** include ```json or ``` markers
3. Do **NOT** add trailing commas
4. Use **double quotes** for all strings and keys
5. Start directly with `{` and end with `}`

**JSON Structure:**

```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.UserMapper.selectUser",
      "command_type": "SELECT",
      "sql_summary": "Brief description of what the query does",
      "input_mapping": {
        "type_category": "PRIMITIVE | MAP | VO | NONE",
        "class_name": "String",
        "crypto_fields": []
      },
      "output_mapping": {
        "type_category": "PRIMITIVE | MAP | VO | NONE",
        "class_name": "HashMap",
        "crypto_fields": [
          {
            "column_name": "jumin_no",
            "java_field": "juminNo"
          }
        ]
      }
    }
  ]
}
```

---

## Field Descriptions

| Field           | Description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| `query_id`      | Unique ID of the query (e.g., `com.example.mapper.UserMapper.selectUser`)            |
| `command_type`  | SQL command type: `SELECT`, `INSERT`, `UPDATE`, `DELETE`                             |
| `sql_summary`   | Brief description of what the query does with target columns                         |
| `type_category` | Type of Java object: `VO`, `MAP`, `PRIMITIVE`, `NONE`                                |
| `class_name`    | Simple class name without package path (e.g., `UserVO`, `HashMap`, `String`)         |
| `column_name`   | Original DB column name                                                              |
| `java_field`    | For VO: field name, For Map: key name (alias if used in SQL, otherwise column name) |
| `getter`        | Getter method name (**ONLY when VO file is provided in context**)                    |
| `setter`        | Setter method name (**ONLY when VO file is provided in context**)                    |

---

## Extraction Guidelines

### 1. Identify Input/Output Types

- **Look for `Parameter Type` and `Result Type`** listed under each query
- **class_name**: Always use simple class name (e.g., `UserVO` not `com.example.UserVO`, `HashMap` not `java.util.HashMap`)

### 2. Handle Map Types

**For Input (parameterType is Map):**
- `#{keyName}` in SQL → `java_field` is `keyName`

**For Output (resultType is Map):**
- With alias: `SELECT col AS alias` → `java_field` is `alias`
- Without alias: `java_field` is `column_name`

**Example:**
```sql
SELECT jumin_no AS ssn, emp_nm FROM TB_EMP WHERE name = #{searchName}
```
- Input: `java_field: "searchName"` (from `#{searchName}`)
- Output: `jumin_no` → `java_field: "ssn"` (alias), `emp_nm` → `java_field: "emp_nm"` (no alias)

### 3. Handle VO Types (★ Use inline field mappings ★)

**Each query includes "Relevant Field Mappings for Target Columns" section directly.**

**When inline field mappings are provided:**
- Look for the "Relevant Field Mappings for Target Columns" section under each query
- Use the exact `java_field` from `Column X → Java field Y` mapping
- **Do NOT include `getter` and `setter`** (resultMap provides the mapping directly)

**Example:** If a query shows:
```
**Relevant Field Mappings for Target Columns (EMP_NM):**
- Column `EMP_NM` → Java field `empNm`
```

Then output:
```json
{
  "column_name": "EMP_NM",
  "java_field": "empNm"
}
```

**When inline field mappings are NOT available (fallback):**
- Infer `java_field` by converting column name to camelCase
- `snake_case` → `camelCase`: `emp_nm` → `empNm`
- `UPPER_SNAKE` → `camelCase`: `EMP_NM` → `empNm`, `BIRTH_DT` → `birthDt`

### 4. Handle SELECT * Queries (★★★ CRITICAL ★★★)

When SQL uses `SELECT *` or `SELECT t.*`:

1. **Include ALL sensitive columns** from `{{ table_info }}` in `output_mapping.crypto_fields`
2. Do NOT leave `crypto_fields` empty just because columns aren't explicit in SQL
3. `SELECT *` returns ALL columns - decryption is required for all sensitive columns

**For java_field:**
- Map result type: use column name as-is
- VO result type with inline mappings: use the `java_field` from "Relevant Field Mappings" section
- VO result type without mappings: infer using camelCase conversion

---

## Complete Examples

### Example 1: SELECT with VO result (inline mapping provided)

**Input (query with inline mappings):**
```
### Query 1: selectEmp (SELECT)
**SQL:**
SELECT jumin_no, emp_nm FROM TB_EMP WHERE emp_id = #{empId}

- **Parameter Type:** `String`
- **Result Type:** `com.example.vo.EmpVO`

**Relevant Field Mappings for Target Columns (JUMIN_NO, EMP_NM):**
- Column `JUMIN_NO` → Java field `juminNo`
- Column `EMP_NM` → Java field `empNm`
```

**Output:**
```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.EmpMapper.selectEmp",
      "command_type": "SELECT",
      "sql_summary": "SELECT jumin_no, emp_nm from TB_EMP",
      "input_mapping": {
        "type_category": "PRIMITIVE",
        "class_name": "String",
        "crypto_fields": []
      },
      "output_mapping": {
        "type_category": "VO",
        "class_name": "EmpVO",
        "crypto_fields": [
          { "column_name": "jumin_no", "java_field": "juminNo" },
          { "column_name": "emp_nm", "java_field": "empNm" }
        ]
      }
    }
  ]
}
```

**★ Note:** No `getter`/`setter` fields - the exact `java_field` is taken from resultMap mapping.

---

### Example 2: SELECT with Map result (aliases)

**Input:**
- Query: `SELECT jumin_no AS ssn, emp_nm AS name FROM TB_EMP WHERE emp_nm = #{searchName}`
- Parameter Type: `java.util.HashMap`
- Result Type: `java.util.HashMap`

**Output:**
```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.EmpMapper.searchByName",
      "command_type": "SELECT",
      "sql_summary": "SELECT by emp_nm - emp_nm in WHERE, jumin_no/emp_nm in result",
      "input_mapping": {
        "type_category": "MAP",
        "class_name": "HashMap",
        "crypto_fields": [
          { "column_name": "emp_nm", "java_field": "searchName" }
        ]
      },
      "output_mapping": {
        "type_category": "MAP",
        "class_name": "HashMap",
        "crypto_fields": [
          { "column_name": "jumin_no", "java_field": "ssn" },
          { "column_name": "emp_nm", "java_field": "name" }
        ]
      }
    }
  ]
}
```

---

### Example 3: UPDATE with sensitive columns in SET and WHERE

**Input:**
- Query: `UPDATE TB_EMP SET birth_dt = #{newBirthDt} WHERE emp_nm = #{searchName}`
- Parameter Type: `java.util.HashMap`

**Output:**
```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.EmpMapper.updateBirthDt",
      "command_type": "UPDATE",
      "sql_summary": "UPDATE birth_dt WHERE emp_nm - both are sensitive",
      "input_mapping": {
        "type_category": "MAP",
        "class_name": "HashMap",
        "crypto_fields": [
          { "column_name": "emp_nm", "java_field": "searchName" },
          { "column_name": "birth_dt", "java_field": "newBirthDt" }
        ]
      },
      "output_mapping": {
        "type_category": "NONE",
        "class_name": null,
        "crypto_fields": []
      }
    }
  ]
}
```

---

### Example 4: SELECT * with Map result

**Input:**
- Query: `SELECT * FROM TB_EMP WHERE emp_id = #{empId}`
- Result Type: `java.util.HashMap`
- Target table has sensitive columns: `jumin_no`, `emp_nm`, `birth_dt`

**Analysis:**
- `SELECT *` returns ALL columns including sensitive ones
- Must include ALL sensitive columns in crypto_fields
- No alias used → java_field = column_name

**Output:**
```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.EmpMapper.selectAll",
      "command_type": "SELECT",
      "sql_summary": "SELECT * - returns ALL columns including jumin_no, emp_nm, birth_dt",
      "input_mapping": {
        "type_category": "PRIMITIVE",
        "class_name": "String",
        "crypto_fields": []
      },
      "output_mapping": {
        "type_category": "MAP",
        "class_name": "HashMap",
        "crypto_fields": [
          { "column_name": "jumin_no", "java_field": "jumin_no" },
          { "column_name": "emp_nm", "java_field": "emp_nm" },
          { "column_name": "birth_dt", "java_field": "birth_dt" }
        ]
      }
    }
  ]
}
```

---

### Example 5: SELECT * with VO result (★ VO file NOT provided ★)

**Input:**
- Query: `SELECT * FROM TARGET_TABLE WHERE USER_UUID = #{userUuid}`
- Result Type: `com.example.vo.UserInfoVO` (★ VO file NOT in context ★)
- Target table has sensitive columns: `USER_NM`, `JUMIN_NO`, `BIRTH_DT`

**Analysis:**
- `SELECT *` returns ALL columns
- VO file is NOT provided → infer java_field using camelCase
- **No getter/setter** because VO structure is unknown

**Output:**
```json
{
  "queries": [
    {
      "query_id": "com.example.mapper.UserMapper.selectByUuid",
      "command_type": "SELECT",
      "sql_summary": "SELECT * - returns ALL columns including USER_NM, JUMIN_NO, BIRTH_DT",
      "input_mapping": {
        "type_category": "PRIMITIVE",
        "class_name": "String",
        "crypto_fields": []
      },
      "output_mapping": {
        "type_category": "VO",
        "class_name": "UserInfoVO",
        "crypto_fields": [
          { "column_name": "USER_NM", "java_field": "userNm" },
          { "column_name": "JUMIN_NO", "java_field": "juminNo" },
          { "column_name": "BIRTH_DT", "java_field": "birthDt" }
        ]
      }
    }
  ]
}
```

**★ Key Point:** No `getter`/`setter` fields because VO file was not provided. Only `column_name` and inferred `java_field` are included.

---

## Start Extraction Now

Analyze the provided SQL queries using the inline field mappings shown with each query.

**Key Points:**
- Each query has a "Relevant Field Mappings for Target Columns" section - use it directly
- For SELECT queries: mappings show `Column X → Java field Y` format
- For INSERT/UPDATE queries: use `#{fieldName}` patterns from SQL
- Do NOT include getter/setter - resultMap provides the mapping directly

**REMINDER: Output ONLY the JSON object. Start directly with `{` and end with `}`.**
