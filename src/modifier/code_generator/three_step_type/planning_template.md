# Encryption/Decryption Modification Planning (Phase 2)

## Role

You are an expert in analyzing **Data Flow** in Java Spring Boot legacy systems.
Based on the information below, output specific modification instructions in JSON format describing **where**, **what**, and **how** to insert encryption/decryption logic.

**Important**: Your role is **analysis and planning**. Actual code writing will be done in the next step.

---

## Encryption Framework Information (KSign)

This project uses the **KSign** encryption framework with `ksignUtil`:

### Encryption/Decryption Methods

- **Encryption**: `ksignUtil.ksignEnc(policyId, inputValue)` - Returns encrypted string
- **Decryption**: `ksignUtil.ksignDec(policyId, inputValue)` - Returns decrypted string

### Policy ID Determination (★★★ CRITICAL ★★★)

**IMPORTANT: Use the following priority order to determine the correct policy_id:**

1. **FIRST**: Check `table_info.columns[].encryption_code` - If provided, use it directly
2. **SECOND**: Check `table_info.columns[].column_type` and map to policy_id:
   - `column_type: "name"` → `"P017"`
   - `column_type: "dob"` → `"P018"`
   - `column_type: "rrn"` → `"P019"`
3. **FALLBACK**: If neither is provided, use column name pattern matching (see table below)

### Policy ID Reference Table

| Field Type                     | column_type | Policy ID | Column Name Patterns (fallback only)                                                     |
| ------------------------------ | ----------- | --------- | ---------------------------------------------------------------------------------------- |
| **Name (이름)**                | `name`      | `"P017"`  | name, userName, user_name, fullName, firstName, lastName, custNm, CUST_NM, empNm, EMP_NM |
| **Date of Birth (생년월일)**   | `dob`       | `"P018"`  | dob, dateOfBirth, birthDate, birthday, dayOfBirth, birthDt, BIRTH_DT                     |
| **Resident Number (주민등록번호)** | `rrn`   | `"P019"`  | jumin, juminNumber, ssn, residentNumber, juminNo, JUMIN_NO, residentNo, rrn              |

### ★★★ CRITICAL: All columns in table_info ARE encryption targets ★★★

**Every column listed in `table_info.columns` has been explicitly configured by the user as an encryption target.**

- **DO NOT skip** any column that appears in `table_info.columns`
- `table_info.columns[].name` is the **DB column name** (e.g., `gvnm`)
- The corresponding **Java field name/alias** (e.g., `aenam`) is identified in Phase 1's `mapping_info`
- Even if the DB column name doesn't match common patterns, it IS an encryption target
- Use `column_type` or `encryption_code` from table_info to determine the correct policy_id

**Important Distinction:**
- `table_info.columns`: Contains **DB column names** configured by user (e.g., `gvnm`)
- `mapping_info.crypto_fields`: Contains **Java field mappings** analyzed in Phase 1 (e.g., `gvnm` → `aenam`)

---

## Analysis Target Information

### ★★★ Target Table/Column Information (CRITICAL) ★★★

**IMPORTANT: Focus ONLY on the target table specified below.**

This is the specific table that requires encryption/decryption modifications. Analyze the data flow ONLY for operations involving this table.

{{ table_info }}

**table_info.columns Structure:**

Each column in `table_info.columns` may contain:
- `name`: Column name (always present)
- `new_column`: Whether this is a new column (boolean)
- `column_type`: Type of sensitive data - `"name"`, `"dob"`, or `"rrn"` (optional but authoritative)
- `encryption_code`: Direct policy_id to use - e.g., `"P017"` (optional but highest priority)

**Example:**
```json
{
  "table_name": "TB_BANNER",
  "columns": [
    { "name": "gvnm", "new_column": false, "column_type": "name", "encryption_code": "P017" }
  ]
}
```

**Instructions:**

1. **ALL columns in table_info.columns ARE encryption targets** - do NOT skip any of them
2. Use `encryption_code` or `column_type` to determine the correct policy_id
3. Analyze queries from `mapping_info.queries[]` (NOT raw SQL - SQL was analyzed in Phase 1)
4. Only analyze methods that are part of call chains in `call_stacks`
5. Generate modification instructions ONLY for files in `source_files`

**IMPORTANT: SQL queries are NOT provided directly in this phase.**
Use `mapping_info` from Phase 1 which contains pre-analyzed query information including:
- `query_id`: Which query is being called
- `command_type`: SELECT/INSERT/UPDATE/DELETE
- `crypto_fields`: Which fields need encryption/decryption with their Java field names

### Data Mapping Summary (★ Pre-analyzed from Phase 1)

The following `mapping_info` was extracted in Phase 1 and contains all SQL query analysis results.

**mapping_info Structure:**

```json
{
  "summary": {
    "total_queries": 3,
    "encryption_needed_count": 2
  },
  "queries": [
    {
      "query_id": "com.example.mapper.UserMapper.selectUser",
      "command_type": "SELECT | INSERT | UPDATE | DELETE",
      "sql_summary": "Brief description of query purpose with target columns",
      "input_mapping": {
        "type_category": "VO | MAP | PRIMITIVE | NONE",
        "class_name": "Simple class name (e.g., UserVO, HashMap)",
        "crypto_fields": [
          {
            "column_name": "DB column name",
            "java_field": "Java field name or Map key",
            "getter": "getXxx (only for VO with provided VO file)",
            "setter": "setXxx (only for VO with provided VO file)"
          }
        ]
      },
      "output_mapping": {
        "type_category": "VO | MAP | PRIMITIVE | NONE",
        "class_name": "Simple class name",
        "crypto_fields": [...]
      }
    }
  ]
}
```

**Key Fields to Use:**

| Field | Description | How to Use |
|-------|-------------|------------|
| `query_id` | Matches DAO/Mapper method name | Match with call chain to identify which query is called |
| `command_type` | SQL command type | `SELECT` → DECRYPT results, `INSERT/UPDATE` → ENCRYPT inputs |
| `sql_summary` | Query purpose description | Understand what the query does |
| `crypto_fields` | Array of fields needing encryption | Contains `column_name`, `java_field`, and optional `getter/setter` |
| `java_field` | Field name (VO) or Map key | Use for code generation (e.g., `vo.getJavaField()` or `map.get("java_field")`) |
| `getter/setter` | Methods for VO types (optional) | Use directly in code_pattern_hint; **only present when VO file was provided in Phase 1** |

**Determining ENCRYPT/DECRYPT action:**

Based on `command_type` and mapping location:
- `input_mapping` fields with `INSERT/UPDATE` command → ENCRYPT before DB operation
- `output_mapping` fields with `SELECT` command → DECRYPT after DB operation
- `input_mapping` fields used in WHERE clause → ENCRYPT (search parameter must match encrypted DB data)

**For Map types with aliases:**

When SQL uses aliases (e.g., `SELECT jumin_no AS ssn`), `java_field` contains the alias:
```json
{
  "column_name": "jumin_no",
  "java_field": "ssn"
}
```
Use the alias for Map operations: `map.get("ssn")` / `map.put("ssn", ...)`

{{ mapping_info }}

### Method Call Chain (Endpoint → SQL)

Call path from controller to SQL:
{{ call_stacks }}

### Source Files to Modify

{{ source_files }}

---

## Analysis Guidelines

### 1. Data Flow Analysis (Per Call Chain)

**For each call chain in call_stacks**, analyze the data flow:

1. Find the matching SQL query in `mapping_info.queries` by matching `query_id` with DAO method.
2. Use `command_type` and mapping location to determine crypto action:
   - `INSERT/UPDATE` command with `input_mapping.crypto_fields` → **ENCRYPT** before DAO call
   - `SELECT` command with `output_mapping.crypto_fields` → **DECRYPT** after DAO returns
   - `SELECT` command with `input_mapping.crypto_fields` (WHERE clause) → **ENCRYPT** search param first
3. Match `crypto_fields[].column_name` with `table_info.columns[].name` to verify encryption target.

### 2. Using Data Mapping (from mapping_info)

**If Input/Output is a VO:**

- **With `getter`/`setter` provided**: Use them directly
  - Example: `vo.setEmpNm(ksignUtil.ksignEnc("P017", vo.getEmpNm()));`
- **Without `getter`/`setter`** (VO file wasn't provided in Phase 1): Infer from `java_field`
  - Assume standard JavaBean conventions: `getXxx()` / `setXxx()`
  - Example: `java_field: "empNm"` → `vo.getEmpNm()` / `vo.setEmpNm()`

**If Input/Output is a Map:**

- Use `java_field` as the Map key (this includes aliases if SQL uses them)
- Example: `map.put("ssn", ksignUtil.ksignDec("P019", (String)map.get("ssn")));`

**If Input/Output is Primitive:**

- It's a single value (e.g., String param). Encrypt/Decrypt the variable directly.
- Example: `jumin = ksignUtil.ksignEnc("P019", jumin);`

### 3. Modification Location Decision (★ CRITICAL: Service Layer Priority)

**IMPORTANT: Always prioritize Service layer for encryption/decryption logic.**

**Why Service Layer First?**
1. **Consistency**: The same file may appear in multiple modification contexts (different tables). Consistent placement in Service layer prevents duplicate encryption/decryption.
2. **Separation of Concerns**: Controller handles HTTP request/response only. Service handles business logic including encryption/decryption.
3. **Maintainability**: All crypto logic in one layer makes it easier to audit and maintain.

**Decision Rules:**
| Call Chain Pattern | Modification Location | Reason |
|-------------------|----------------------|--------|
| Controller → Service → DAO | **Service** (preferred) | Standard pattern, crypto in Service |
| Controller → DAO (no Service) | Controller | No Service layer exists |
| Service → DAO (no Controller) | Service | Batch/scheduled job pattern |

**⚠️ WARNING: Never add crypto logic to BOTH Controller AND Service for the same data flow!**
This causes double encryption/decryption which corrupts data.

```java
// ❌ WRONG: Crypto in both layers
// Controller
vo.setName(ksignUtil.ksignEnc("P017", vo.getName()));  // First encryption
employeeService.save(vo);

// Service
vo.setName(ksignUtil.ksignEnc("P017", vo.getName()));  // Double encryption! DATA CORRUPTED!
employeeDao.insert(vo);

// ✅ CORRECT: Crypto only in Service layer
// Controller - NO crypto logic
employeeService.save(vo);

// Service - crypto logic HERE
vo.setName(ksignUtil.ksignEnc("P017", vo.getName()));
employeeDao.insert(vo);
```

**When to SKIP Controller files:**
- If the call chain includes a Service layer, mark Controller modifications as `action: "SKIP"` with reason: "Encryption/decryption handled in Service layer"

### 4. Minimize Modification Scope

- Maintain existing code structure and logic as much as possible
- Only add encryption/decryption logic
- Do not modify unnecessary files

---

## Output Format (★★★ CRITICAL: JSON ONLY ★★★)

**IMPORTANT OUTPUT RULES:**
1. Output **ONLY** valid JSON - no explanations, no markdown, no comments before or after
2. Do **NOT** include ```json or ``` markers
3. Do **NOT** add trailing commas (e.g., `{"a": 1,}` is INVALID)
4. Do **NOT** include JavaScript-style comments (`//` or `/* */`)
5. Use **double quotes** for all strings and keys
6. Ensure all brackets and braces are properly closed
7. Use `null` instead of `undefined` or empty values

**Expected JSON Structure:**
```json
{
  "data_flow_analysis": {
    "overview": "Overview of the entire data flow (2-3 sentences)",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "User Registration Flow",
        "direction": "INBOUND_TO_DB | DB_TO_OUTBOUND | BIDIRECTIONAL",
        "data_source": {
          "type": "HTTP_REQUEST | SESSION | DB | EXTERNAL_API",
          "description": "Where the data comes from"
        },
        "data_sink": {
          "type": "DB | HTTP_RESPONSE | SESSION | EXTERNAL_API",
          "description": "Where the data goes to"
        },
        "path": "Controller.method() → Service.method() → DAO.method() → DB",
        "sensitive_columns": ["last_name", "jumin_number"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001 (matches data_flow_analysis.flows[].flow_id)",
      "file_name": "File name (e.g., UserService.java)",
      "target_method": "Method name to modify",
      "action": "ENCRYPT | DECRYPT | ENCRYPT_THEN_DECRYPT | SKIP",
      "reason": "Reason for this modification (or reason for SKIP)",
      "target_properties": ["empNm", "birthDt", "juminNo"],
      "insertion_point": "Code insertion location description (e.g., 'right before dao.insert(list) call')",
      "code_pattern_hint": "Code pattern hint to insert"
    }
  ]
}
```

### BIDIRECTIONAL Flow Structure

For `direction: "BIDIRECTIONAL"` (e.g., search with encrypted WHERE + decrypted result), use nested objects for each direction:

```json
{
  "flow_id": "FLOW_001",
  "flow_name": "Customer Search with Encrypted WHERE",
  "direction": "BIDIRECTIONAL",
  "INBOUND_TO_DB": {
    "data_source": {
      "type": "HTTP_REQUEST",
      "description": "Search parameter (name) from user input"
    },
    "data_sink": {
      "type": "DB",
      "description": "Query executes against encrypted DB data"
    }
  },
  "DB_TO_OUTBOUND": {
    "data_source": {
      "type": "DB",
      "description": "Encrypted results from database"
    },
    "data_sink": {
      "type": "HTTP_RESPONSE",
      "description": "Decrypted results returned to client"
    }
  },
  "path": "Controller → Service.search() → DAO → DB → Service → Controller → Client",
  "sensitive_columns": ["cust_nm", "birth_dt"]
}
```

### Field Descriptions

| Field               | Description                                              | Example                                                       |
| ------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| `flow_id`           | Reference to data_flow_analysis.flows[].flow_id          | `FLOW_001`, `FLOW_002`                                        |
| `file_name`         | File name to modify                                      | `UserService.java`, `EmpController.java`                      |
| `target_method`     | Method name to modify                                    | `saveUser`, `getUserList`                                     |
| `action`            | Action to perform                                        | `ENCRYPT`, `DECRYPT`, `ENCRYPT_THEN_DECRYPT`, `SKIP`          |
| `target_properties` | Properties to encrypt/decrypt (array of strings)         | `["empNm", "birthDt", "juminNo"]`                             |
| `insertion_point`   | Insertion location description                           | `right before dao.insert() call`, `right before return list;` |
| `code_pattern_hint` | Code pattern example                                     | `vo.setEmpNm(ksignUtil.ksignEnc("P017", vo.getEmpNm()));`     |

### ⚠️ CRITICAL: Every Flow MUST Have a modification_instruction Entry ⚠️

**For EVERY flow in `data_flow_analysis.flows`, you MUST output a corresponding entry in `modification_instructions`.**

- If the flow requires encryption/decryption → output with `action: "ENCRYPT"` or `action: "DECRYPT"`
- If the flow does NOT require modification → output with `action: "SKIP"` and explain `reason`

**DO NOT skip any flow!** The Execution phase relies on explicit SKIP entries to know which flows were analyzed and intentionally skipped.

**Example:** If `data_flow_analysis.flows` contains FLOW_001, FLOW_002, FLOW_003, then `modification_instructions` MUST contain entries for ALL three flows (even if some are SKIP).

### Important Notes

1. **When action is SKIP**: Specify in `reason` which flow (flow_id) this refers to and why no modification is needed
2. **target_properties**: Array of `java_field` names (strings) from `crypto_fields`. Use the Java field name, not DB column name.
3. **insertion_point**: Describe specifically so code can be inserted in the next step
4. **code_pattern_hint**:
   - For VO with getter/setter: Use them directly (e.g., `vo.setEmpNm(ksignUtil.ksignEnc("P017", vo.getEmpNm()));`)
   - For VO without getter/setter: Infer from java_field (e.g., `java_field: "empNm"` → `vo.setEmpNm(...)`)
   - For Map: Use java_field as key (e.g., `map.put("ssn", ksignUtil.ksignDec("P019", (String)map.get("ssn")));`)
5. **Use mapping_info.crypto_fields**: Reference `java_field`, and `getter`/`setter` (if present) for accurate code patterns

---

## Critical Encryption/Decryption Rules

### Core Principle: Encrypt/Decrypt ONLY when data crosses the DB boundary

| Data Source  | Data Sink     | Action      | Reason                                                            |
| ------------ | ------------- | ----------- | ----------------------------------------------------------------- |
| HTTP_REQUEST | DB            | **ENCRYPT** | Plaintext from client must be encrypted before DB storage         |
| DB           | HTTP_RESPONSE | **DECRYPT** | Encrypted data from DB must be decrypted before sending to client |
| DB           | EXTERNAL_API  | **DECRYPT** | External systems expect plaintext data                            |
| EXTERNAL_API | DB            | **ENCRYPT** | Data from external systems must be encrypted before DB storage    |
| SESSION      | DB            | **ENCRYPT** | Session data is plaintext, must be encrypted for DB               |
| DB           | SESSION       | **DECRYPT** | Encrypted DB data must be decrypted for session storage           |
| SESSION      | HTTP_RESPONSE | **NONE**    | Session data is already plaintext, no decryption needed           |
| HTTP_REQUEST | SESSION       | **NONE**    | No DB involved, no encryption needed                              |

### ⚠️ CRITICAL: Session Data is ALWAYS Plaintext - NEVER Decrypt

**Session data has already been decrypted during login.** When you see code like:

```java
MemberVO member = (MemberVO) session.getAttribute("member");
String userName = member.getUserNm();  // This is ALREADY plaintext!
```

**DO NOT decrypt session data!** The decryption already happened when the user logged in (DB → Session flow).

| Pattern                                        | Action           | Reason                          |
| ---------------------------------------------- | ---------------- | ------------------------------- |
| `session.getAttribute(...)` → use data         | **NO DECRYPT**   | Session stores plaintext        |
| `session.getAttribute(...)` → save to DB       | **ENCRYPT only** | Plaintext → DB needs encryption |
| `session.getAttribute(...)` → return to client | **NO DECRYPT**   | Already plaintext               |

### Special Case: SELECT with WHERE clause on sensitive columns

When `mapping_info` shows a sensitive column in BOTH `input_mapping` AND `output_mapping` for a SELECT:

1. **First**: ENCRYPT the search parameter (to match encrypted data in DB)
2. **Then**: Execute the query
3. **Finally**: DECRYPT the result (to return plaintext to caller)

**How to identify this pattern:**
```json
{
  "command_type": "SELECT",
  "input_mapping": {
    "crypto_fields": [
      {"column_name": "jumin_no", "java_field": "searchJumin"}  // → ENCRYPT (WHERE clause)
    ]
  },
  "output_mapping": {
    "crypto_fields": [
      {"column_name": "jumin_no", "java_field": "juminNo"},     // → DECRYPT (result)
      {"column_name": "emp_nm", "java_field": "empNm"}          // → DECRYPT (result)
    ]
  }
}
```

---

## Example Output

### Example: INSERT and SELECT with VO

**mapping_info (from Phase 1):**
```json
{
  "queries": [
    {
      "query_id": "EmpMapper.insertEmp",
      "command_type": "INSERT",
      "sql_summary": "Employee registration - INSERT emp_nm, birth_dt, jumin_no",
      "input_mapping": {
        "type_category": "VO",
        "class_name": "EmpVO",
        "crypto_fields": [
          {"column_name": "emp_nm", "java_field": "empNm", "getter": "getEmpNm", "setter": "setEmpNm"},
          {"column_name": "birth_dt", "java_field": "birthDt", "getter": "getBirthDt", "setter": "setBirthDt"},
          {"column_name": "jumin_no", "java_field": "juminNo", "getter": "getJuminNo", "setter": "setJuminNo"}
        ]
      },
      "output_mapping": {
        "type_category": "NONE",
        "class_name": null,
        "crypto_fields": []
      }
    },
    {
      "query_id": "EmpMapper.selectEmp",
      "command_type": "SELECT",
      "sql_summary": "Employee retrieval - SELECT emp_nm, birth_dt, jumin_no",
      "input_mapping": {
        "type_category": "PRIMITIVE",
        "class_name": "String",
        "crypto_fields": []
      },
      "output_mapping": {
        "type_category": "VO",
        "class_name": "EmpVO",
        "crypto_fields": [
          {"column_name": "emp_nm", "java_field": "empNm", "getter": "getEmpNm", "setter": "setEmpNm"},
          {"column_name": "birth_dt", "java_field": "birthDt", "getter": "getBirthDt", "setter": "setBirthDt"},
          {"column_name": "jumin_no", "java_field": "juminNo", "getter": "getJuminNo", "setter": "setJuminNo"}
        ]
      }
    }
  ]
}
```

**Output:**

```json
{
  "data_flow_analysis": {
    "overview": "The employee table stores user information. INSERT requires encryption before DB save, SELECT requires decryption after DB retrieval.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Employee Registration (INSERT)",
        "direction": "INBOUND_TO_DB",
        "data_source": {"type": "HTTP_REQUEST", "description": "Client POST request"},
        "data_sink": {"type": "DB", "description": "INSERT into TB_EMP"},
        "path": "Controller → Service.save() → DAO.insert() → DB",
        "sensitive_columns": ["emp_nm", "birth_dt", "jumin_no"]
      },
      {
        "flow_id": "FLOW_002",
        "flow_name": "Employee Retrieval (SELECT)",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {"type": "DB", "description": "SELECT from TB_EMP"},
        "data_sink": {"type": "HTTP_RESPONSE", "description": "JSON response"},
        "path": "DB → DAO.select() → Service.get() → Controller → Client",
        "sensitive_columns": ["emp_nm", "birth_dt", "jumin_no"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "EmployeeService.java",
      "target_method": "saveEmployee",
      "action": "ENCRYPT",
      "reason": "FLOW_001: INSERT command requires encryption before DB save",
      "target_properties": ["empNm", "birthDt", "juminNo"],
      "insertion_point": "Right before employeeDao.insert(vo) call",
      "code_pattern_hint": "vo.setEmpNm(ksignUtil.ksignEnc(\"P017\", vo.getEmpNm()));\nvo.setBirthDt(ksignUtil.ksignEnc(\"P018\", vo.getBirthDt()));\nvo.setJuminNo(ksignUtil.ksignEnc(\"P019\", vo.getJuminNo()));"
    },
    {
      "flow_id": "FLOW_002",
      "file_name": "EmployeeService.java",
      "target_method": "getEmployeeById",
      "action": "DECRYPT",
      "reason": "FLOW_002: SELECT command requires decryption after DB retrieval",
      "target_properties": ["empNm", "birthDt", "juminNo"],
      "insertion_point": "Right after DAO return, before return statement",
      "code_pattern_hint": "result.setEmpNm(ksignUtil.ksignDec(\"P017\", result.getEmpNm()));\nresult.setBirthDt(ksignUtil.ksignDec(\"P018\", result.getBirthDt()));\nresult.setJuminNo(ksignUtil.ksignDec(\"P019\", result.getJuminNo()));"
    }
  ]
}
```

### Example: Map with Alias

**mapping_info shows:**
```json
{
  "output_mapping": {
    "type_category": "MAP",
    "class_name": "HashMap",
    "crypto_fields": [
      {"column_name": "jumin_no", "java_field": "ssn"}
    ]
  }
}
```

**code_pattern_hint for Map:**
```java
resultMap.put("ssn", ksignUtil.ksignDec("P019", (String)resultMap.get("ssn")));
```

---

### Example: Session data to DB (Session → DB) ★ IMPORTANT

**Scenario:**
- SQL: `INSERT INTO audit_log (user_nm, action) VALUES (#{userNm}, #{action})`
- Method chain: `AuditService.logAction()` → `AuditDao.insertLog()`
- Note: `userNm` is retrieved from HTTP Session (already plaintext)

**Key Point:** Session data is ALREADY PLAINTEXT (was decrypted during login). Only ENCRYPT is needed before DB save. **DO NOT decrypt session data!**

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Audit logging saves user action with user name from session. Session data is already plaintext, so only encryption is needed before DB save.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Audit Log Insert (Session → DB)",
        "direction": "INBOUND_TO_DB",
        "data_source": {"type": "SESSION", "description": "User name from HTTP session (already plaintext)"},
        "data_sink": {"type": "DB", "description": "INSERT into audit_log table"},
        "path": "Session → AuditService.logAction() → AuditDao.insertLog() → DB",
        "sensitive_columns": ["user_nm"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "AuditService.java",
      "target_method": "logAction",
      "action": "ENCRYPT",
      "reason": "FLOW_001: Session data is plaintext, must encrypt before DB storage. No decryption needed.",
      "target_properties": ["userNm"],
      "insertion_point": "Right before auditDao.insertLog() call",
      "code_pattern_hint": "logData.setUserNm(ksignUtil.ksignEnc(\"P017\", logData.getUserNm()));"
    }
  ]
}
```

---

### Example: DB to Session storage (Login scenario)

**Scenario:**
- SQL: `SELECT user_nm, birth_dt FROM users WHERE login_id = #{loginId}`
- Method chain: `LoginController.login()` → `AuthService.authenticate()` → `UserDao.selectByLoginId()` → Session storage
- Note: After successful login, user info is stored in session for later use

**Key Point:** Session should store PLAINTEXT for easy access. DB data must be DECRYPTED before session storage.

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "User authentication retrieves encrypted data from DB and stores plaintext in session for easy access during user's session.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "User Login (DB → Session)",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {"type": "DB", "description": "SELECT user info (encrypted in DB)"},
        "data_sink": {"type": "SESSION", "description": "Store user info in HTTP session (as plaintext)"},
        "path": "DB → UserDao.selectByLoginId() → AuthService.authenticate() → Session",
        "sensitive_columns": ["user_nm", "birth_dt"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "AuthService.java",
      "target_method": "authenticate",
      "action": "DECRYPT",
      "reason": "FLOW_001: DB data is encrypted, must decrypt before storing plaintext in session",
      "target_properties": ["userNm", "birthDt"],
      "insertion_point": "Right after userDao.selectByLoginId() return, before session.setAttribute()",
      "code_pattern_hint": "userInfo.setUserNm(ksignUtil.ksignDec(\"P017\", userInfo.getUserNm()));\nuserInfo.setBirthDt(ksignUtil.ksignDec(\"P018\", userInfo.getBirthDt()));"
    }
  ]
}
```

---

### Example: Session to HTTP Response (No crypto needed) ★ IMPORTANT

**Scenario:**
- Method chain: `ProfileController.getMyProfile()` → retrieves data from Session → returns HTTP response
- Note: No DB access, data comes directly from session

**Key Point:** Session data is ALREADY PLAINTEXT. NO encryption/decryption needed. **This should be SKIP.**

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "User profile is retrieved directly from session and returned to client. Session stores plaintext (decrypted during login), so NO crypto operation is needed.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Profile from Session (No DB)",
        "direction": "SESSION_TO_OUTBOUND",
        "data_source": {"type": "SESSION", "description": "User profile stored in session (already plaintext)"},
        "data_sink": {"type": "HTTP_RESPONSE", "description": "Return profile to client"},
        "path": "Session → ProfileController.getMyProfile() → Client",
        "sensitive_columns": ["user_nm", "birth_dt"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "ProfileController.java",
      "target_method": "getMyProfile",
      "action": "SKIP",
      "reason": "FLOW_001: No DB access. Session data is already plaintext (decrypted during login). No encryption/decryption needed.",
      "target_properties": [],
      "insertion_point": "",
      "code_pattern_hint": ""
    }
  ]
}
```

---

### Example: DB to External API

**Scenario:**
- SQL: `SELECT mem_nm, birth_dt, jumin_no FROM member WHERE id = #{memberId}`
- Method chain: `IntegrationController.sendToPartner()` → `MemberService.getMemberForExport()` → `MemberDao.selectById()` → `ExternalApiClient.sendMemberInfo()`
- Note: External partner system expects plaintext data

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Member data is retrieved from DB and sent to external partner API. External systems expect plaintext, so encrypted DB data must be decrypted.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Member Export to External API",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {"type": "DB", "description": "SELECT member data (encrypted in DB)"},
        "data_sink": {"type": "EXTERNAL_API", "description": "Partner system API expects plaintext"},
        "path": "DB → MemberDao.selectById() → MemberService.getMemberForExport() → ExternalApiClient → Partner",
        "sensitive_columns": ["mem_nm", "birth_dt", "jumin_no"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "MemberService.java",
      "target_method": "getMemberForExport",
      "action": "DECRYPT",
      "reason": "FLOW_001: External API expects plaintext. Must decrypt DB data before sending.",
      "target_properties": ["memNm", "birthDt", "juminNo"],
      "insertion_point": "Right after memberDao.selectById() return, before externalApiClient call",
      "code_pattern_hint": "memberData.setMemNm(ksignUtil.ksignDec(\"P017\", memberData.getMemNm()));\nmemberData.setBirthDt(ksignUtil.ksignDec(\"P018\", memberData.getBirthDt()));\nmemberData.setJuminNo(ksignUtil.ksignDec(\"P019\", memberData.getJuminNo()));"
    }
  ]
}
```

---

### Example: External API to DB

**Scenario:**
- SQL: `INSERT INTO external_customer (cust_nm, birth_dt) VALUES (#{custNm}, #{birthDt})`
- Method chain: `WebhookController.receiveCustomer()` → `ExternalCustomerService.saveFromPartner()` → `ExternalCustomerDao.insert()`
- Note: Data received from external partner system (plaintext)

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Customer data received from external partner webhook. External data arrives as plaintext and must be encrypted before storing in DB.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "External Customer Import",
        "direction": "INBOUND_TO_DB",
        "data_source": {"type": "EXTERNAL_API", "description": "Partner system sends data via webhook (plaintext)"},
        "data_sink": {"type": "DB", "description": "INSERT into external_customer table"},
        "path": "Partner → WebhookController → ExternalCustomerService.saveFromPartner() → DAO → DB",
        "sensitive_columns": ["cust_nm", "birth_dt"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "ExternalCustomerService.java",
      "target_method": "saveFromPartner",
      "action": "ENCRYPT",
      "reason": "FLOW_001: External API data is plaintext, must encrypt before DB storage",
      "target_properties": ["custNm", "birthDt"],
      "insertion_point": "Right before externalCustomerDao.insert() call",
      "code_pattern_hint": "customerData.setCustNm(ksignUtil.ksignEnc(\"P017\", customerData.getCustNm()));\ncustomerData.setBirthDt(ksignUtil.ksignEnc(\"P018\", customerData.getBirthDt()));"
    }
  ]
}
```

---

### Example: SELECT with WHERE on sensitive column + DECRYPT result

**Scenario:**
- SQL: `SELECT id, cust_nm, birth_dt FROM customer WHERE cust_nm = #{custNm}`
- Method chain: `SearchController.search()` → `CustomerService.searchByName()` → `CustomerDao.selectByName()`
- Note: WHERE clause uses encrypted column, results need decryption

**Key Point:** Must ENCRYPT search parameter first (to match encrypted DB data), then DECRYPT results.

**mapping_info shows:**
```json
{
  "input_mapping": {
    "type_category": "MAP",
    "class_name": "HashMap",
    "crypto_fields": [
      {"column_name": "cust_nm", "java_field": "custNm"}
    ]
  },
  "output_mapping": {
    "type_category": "VO",
    "class_name": "Customer",
    "crypto_fields": [
      {"column_name": "cust_nm", "java_field": "custNm", "getter": "getCustNm", "setter": "setCustNm"},
      {"column_name": "birth_dt", "java_field": "birthDt", "getter": "getBirthDt", "setter": "setBirthDt"}
    ]
  }
}
```

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Search by name requires encrypting the search parameter to match encrypted DB data, then decrypting results for display.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Customer Search with Encrypted WHERE",
        "direction": "BIDIRECTIONAL",
        "INBOUND_TO_DB": {
          "data_source": {"type": "HTTP_REQUEST", "description": "Search parameter (name) from user input"},
          "data_sink": {"type": "DB", "description": "Query executes against encrypted DB data"}
        },
        "DB_TO_OUTBOUND": {
          "data_source": {"type": "DB", "description": "Encrypted results from database"},
          "data_sink": {"type": "HTTP_RESPONSE", "description": "Decrypted results returned to client"}
        },
        "path": "SearchController → CustomerService.searchByName() → DAO → DB → Service → Controller → Client",
        "sensitive_columns": ["cust_nm", "birth_dt"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "CustomerService.java",
      "target_method": "searchByName",
      "action": "ENCRYPT_THEN_DECRYPT",
      "reason": "FLOW_001: BIDIRECTIONAL - search param needs ENCRYPT, results need DECRYPT",
      "target_properties": ["custNm", "birthDt"],
      "insertion_point": "ENCRYPT: Before customerDao.selectByName() call; DECRYPT: After DAO return",
      "code_pattern_hint": "// Before DAO call: encrypt search parameter\nsearchParam.put(\"custNm\", ksignUtil.ksignEnc(\"P017\", (String)searchParam.get(\"custNm\")));\nList<Customer> resultList = customerDao.selectByName(searchParam);\n// After DAO call: decrypt results\nfor (Customer c : resultList) {\n    c.setCustNm(ksignUtil.ksignDec(\"P017\", c.getCustNm()));\n    c.setBirthDt(ksignUtil.ksignDec(\"P018\", c.getBirthDt()));\n}"
    }
  ]
}
```

---

### Example: UPDATE with WHERE on sensitive column

**Scenario:**
- SQL: `UPDATE customer SET birth_dt = #{newBirthDt} WHERE cust_nm = #{custNm}`
- Method chain: `CustomerController.updateBirthDt()` → `CustomerService.updateBirthDtByName()` → `CustomerDao.updateBirthDtByName()`

**Key Point:** Both WHERE condition (cust_nm) and SET value (birth_dt) must be ENCRYPTED.

**mapping_info shows:**
```json
{
  "input_mapping": {
    "type_category": "MAP",
    "class_name": "HashMap",
    "crypto_fields": [
      {"column_name": "cust_nm", "java_field": "custNm"},
      {"column_name": "birth_dt", "java_field": "newBirthDt"}
    ]
  }
}
```

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Update customer birthdate by name. Both the WHERE condition and SET value are sensitive columns requiring encryption.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Update BirthDt by Name",
        "direction": "INBOUND_TO_DB",
        "data_source": {"type": "HTTP_REQUEST", "description": "Client sends name (search) and newBirthDt (update value)"},
        "data_sink": {"type": "DB", "description": "UPDATE customer table"},
        "path": "CustomerController → CustomerService.updateBirthDtByName() → DAO → DB",
        "sensitive_columns": ["cust_nm", "birth_dt"]
      }
    ]
  },
  "modification_instructions": [
    {
      "flow_id": "FLOW_001",
      "file_name": "CustomerService.java",
      "target_method": "updateBirthDtByName",
      "action": "ENCRYPT",
      "reason": "FLOW_001: UPDATE command - both WHERE (custNm) and SET (newBirthDt) need encryption",
      "target_properties": ["custNm", "newBirthDt"],
      "insertion_point": "Right before customerDao.updateBirthDtByName() call",
      "code_pattern_hint": "// Encrypt both search param and update value\nupdateParams.put(\"custNm\", ksignUtil.ksignEnc(\"P017\", (String)updateParams.get(\"custNm\")));\nupdateParams.put(\"newBirthDt\", ksignUtil.ksignEnc(\"P018\", (String)updateParams.get(\"newBirthDt\")));"
    }
  ]
}
```

---

## Start Analysis Now

Based on the information above:

1. **For each call chain in call_stacks**, find the matching query in mapping_info
2. **Use `command_type`** and mapping location to determine ENCRYPT/DECRYPT action
3. **Use `java_field`, `getter`, `setter`** from crypto_fields to generate accurate code patterns
4. **Output modification instructions** for each flow in JSON format
5. **SKIP** flows that don't involve the target table's encryption columns

**Remember**: Focus on the target table. Only include modification instructions for operations that interact with the target table.

**REMINDER: Output ONLY the JSON object. Start directly with `{` and end with `}`. No other text allowed.**
