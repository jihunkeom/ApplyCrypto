# Encryption/Decryption Modification Planning (Planning Phase)

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
- The corresponding **Java field name** may differ due to aliasing in SQL or VO mapping
- Even if the DB column name doesn't match common patterns, it IS an encryption target
- Use `column_type` or `encryption_code` from table_info to determine the correct policy_id

---

## Analysis Target Information

### ★★★ Target Table/Column Information (CRITICAL) ★★★

**IMPORTANT: Focus ONLY on the target table specified below.**

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
3. Only analyze SQL queries that access the **target table** above
4. Generate modification instructions ONLY for files involved in **target table** operations

### SQL Query Analysis (★ Core Data Flow Information)
The following are actual SQL queries accessing this table. **Query type (SELECT/INSERT/UPDATE/DELETE)** determines encryption/decryption location:
{{ sql_queries }}

### Method Call Chain (Endpoint → SQL)
Call path from controller to SQL:
{{ call_stacks }}

### Source Files to Modify
{{ source_files }}

{% if context_files %}
### Reference Files (VO/DTO Classes - Not to be modified)
Reference files for understanding data structures:
{{ context_files }}
{% endif %}

---

## Analysis Guidelines

### 1. Data Flow Analysis
Analyze SQL queries and call chains to understand how data flows:
- **INSERT/UPDATE queries** → Encryption needed **before** saving to DB
- **SELECT queries** → Decryption needed **after** retrieving from DB

### 2. Modification Location Decision (★ CRITICAL: Service Layer Priority)

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

### 3. Minimize Modification Scope
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

| Field | Description | Example |
|-------|-------------|---------|
| `flow_id` | Reference to data_flow_analysis.flows[].flow_id | `FLOW_001`, `FLOW_002` |
| `file_name` | File name to modify | `UserService.java`, `EmpController.java` |
| `target_method` | Method name to modify | `saveUser`, `getUserList` |
| `action` | Action to perform | `ENCRYPT`, `DECRYPT`, `ENCRYPT_THEN_DECRYPT`, `SKIP` |
| `target_properties` | Properties to encrypt/decrypt (array of strings) | `["empNm", "birthDt", "juminNo"]` |
| `insertion_point` | Insertion location description | `right before dao.insert() call`, `right before return list;` |
| `code_pattern_hint` | Code pattern example | `vo.setEmpNm(ksignUtil.ksignEnc("P017", vo.getEmpNm()));` |

### ⚠️ CRITICAL: Every Flow MUST Have a modification_instruction Entry ⚠️

**For EVERY flow in `data_flow_analysis.flows`, you MUST output a corresponding entry in `modification_instructions`.**

- If the flow requires encryption/decryption → output with `action: "ENCRYPT"` or `action: "DECRYPT"`
- If the flow does NOT require modification → output with `action: "SKIP"` and explain `reason`

**DO NOT skip any flow!** The Execution phase relies on explicit SKIP entries to know which flows were analyzed and intentionally skipped.

**Example:** If `data_flow_analysis.flows` contains FLOW_001, FLOW_002, FLOW_003, then `modification_instructions` MUST contain entries for ALL three flows (even if some are SKIP).

### Important Notes

1. **When action is SKIP**: Specify in `reason` which flow (flow_id) this refers to and why no modification is needed
2. **target_properties**: Array of Java property names (strings) corresponding to sensitive columns. Use the Java field name (e.g., `empNm`), not DB column name (e.g., `emp_nm`).
3. **insertion_point**: Describe specifically so code can be inserted in the next step
4. **code_pattern_hint**:
   - For VO: `vo.setEmpNm(ksignUtil.ksignEnc("P017", vo.getEmpNm()));`
   - For Map: `map.put("key", ksignUtil.ksignEnc("P017", (String)map.get("key")));`
5. **Policy ID determination**: Use `encryption_code` > `column_type` > column name pattern (in priority order)

---

## Critical Encryption/Decryption Rules

### Core Principle: Encrypt/Decrypt ONLY when data crosses the DB boundary

| Data Source | Data Sink | Action | Reason |
|-------------|-----------|--------|--------|
| HTTP_REQUEST | DB | **ENCRYPT** | Plaintext from client must be encrypted before DB storage |
| DB | HTTP_RESPONSE | **DECRYPT** | Encrypted data from DB must be decrypted before sending to client |
| DB | EXTERNAL_API | **DECRYPT** | External systems expect plaintext data |
| EXTERNAL_API | DB | **ENCRYPT** | Data from external systems must be encrypted before DB storage |
| SESSION | DB | **ENCRYPT** | Session data is plaintext, must be encrypted for DB |
| DB | SESSION | **DECRYPT** | Encrypted DB data must be decrypted for session storage |
| SESSION | HTTP_RESPONSE | **NONE** | Session data is already plaintext, no decryption needed |
| HTTP_REQUEST | SESSION | **NONE** | No DB involved, no encryption needed |

### ⚠️ CRITICAL: Session Data is ALWAYS Plaintext - NEVER Decrypt

**Session data has already been decrypted during login.** When you see code like:
```java
MemberVO member = (MemberVO) session.getAttribute("member");
String userName = member.getUserNm();  // This is ALREADY plaintext!
```

**DO NOT decrypt session data!** The decryption already happened when the user logged in (DB → Session flow).

| Pattern | Action | Reason |
|---------|--------|--------|
| `session.getAttribute(...)` → use data | **NO DECRYPT** | Session stores plaintext |
| `session.getAttribute(...)` → save to DB | **ENCRYPT only** | Plaintext → DB needs encryption |
| `session.getAttribute(...)` → return to client | **NO DECRYPT** | Already plaintext |

**Common Mistake to Avoid:**
```java
// ❌ WRONG - DO NOT DO THIS
MemberVO member = (MemberVO) session.getAttribute("member");
member.setUserNm(ksignUtil.ksignDec("P017", member.getUserNm()));  // WRONG!

// ✅ CORRECT - Session data is already plaintext, use as-is
MemberVO member = (MemberVO) session.getAttribute("member");
String userName = member.getUserNm();  // Already plaintext, just use it
```

### Special Case: SELECT with WHERE clause on sensitive columns

When a SELECT query has a WHERE clause that references sensitive columns (name, DOB, or jumin):
1. **First**: ENCRYPT the search parameter (to match encrypted data in DB)
2. **Then**: Execute the query
3. **Finally**: DECRYPT the result (to return plaintext to caller)

```
Example: SELECT empNm, birthDt FROM employee WHERE empNm = #{searchName}
→ Step 1: Encrypt searchName with ksignUtil.ksignEnc("P017", searchName)
→ Step 2: Execute query (matches encrypted name in DB)
→ Step 3: Decrypt result's empNm with ksignUtil.ksignDec("P017", empNm)
         Decrypt result's birthDt with ksignUtil.ksignDec("P018", birthDt)
```

---

## Examples

### Example 1: Basic INSERT and SELECT (HTTP ↔ DB)

**Input:**
- SQL 1: `INSERT INTO employee (emp_nm, birth_dt, jumin_no) VALUES (#{empNm}, #{birthDt}, #{juminNo})`
- SQL 2: `SELECT emp_nm, birth_dt, jumin_no FROM employee WHERE id = #{id}`
- Method chain 1: `EmpController.save()` → `EmployeeService.saveEmployee()` → `EmployeeDao.insert()`
- Method chain 2: `EmpController.getEmployee()` → `EmployeeService.getEmployeeById()` → `EmployeeDao.selectById()`

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "The employee table stores user information, with emp_nm (name), birth_dt (DOB), and jumin_no (resident number) as sensitive data requiring encryption. Data from HTTP requests needs encryption before DB save, and data from DB needs decryption before response.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Employee Registration (INSERT)",
        "direction": "INBOUND_TO_DB",
        "data_source": {
          "type": "HTTP_REQUEST",
          "description": "Client sends employee information via POST request"
        },
        "data_sink": {
          "type": "DB",
          "description": "INSERT into employee table"
        },
        "path": "EmpController.save() → EmployeeService.saveEmployee() → EmployeeDao.insert() → DB",
        "sensitive_columns": ["emp_nm", "birth_dt", "jumin_no"]
      },
      {
        "flow_id": "FLOW_002",
        "flow_name": "Employee Retrieval (SELECT)",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {
          "type": "DB",
          "description": "SELECT from employee table"
        },
        "data_sink": {
          "type": "HTTP_RESPONSE",
          "description": "Return as JSON response to client"
        },
        "path": "DB → EmployeeDao.selectById() → EmployeeService.getEmployeeById() → EmpController.getEmployee() → Client",
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
      "reason": "FLOW_001: Encryption needed before saving HTTP request data to DB",
      "target_properties": ["empNm", "birthDt", "juminNo"],
      "insertion_point": "Right before employeeDao.insert(vo) call",
      "code_pattern_hint": "vo.setEmpNm(ksignUtil.ksignEnc(\"P017\", vo.getEmpNm()));\nvo.setBirthDt(ksignUtil.ksignEnc(\"P018\", vo.getBirthDt()));\nvo.setJuminNo(ksignUtil.ksignEnc(\"P019\", vo.getJuminNo()));"
    },
    {
      "flow_id": "FLOW_002",
      "file_name": "EmployeeService.java",
      "target_method": "getEmployeeById",
      "action": "DECRYPT",
      "reason": "FLOW_002: Decryption needed before returning encrypted data from DB to client",
      "target_properties": ["empNm", "birthDt", "juminNo"],
      "insertion_point": "Right after employeeDao.selectById(id) return, before return statement",
      "code_pattern_hint": "employee.setEmpNm(ksignUtil.ksignDec(\"P017\", employee.getEmpNm()));\nemployee.setBirthDt(ksignUtil.ksignDec(\"P018\", employee.getBirthDt()));\nemployee.setJuminNo(ksignUtil.ksignDec(\"P019\", employee.getJuminNo()));"
    },
    {
      "flow_id": "FLOW_001",
      "file_name": "EmpController.java",
      "target_method": "any",
      "action": "SKIP",
      "reason": "Controller only handles data passing, encryption/decryption is handled in Service layer",
      "target_properties": [],
      "insertion_point": "",
      "code_pattern_hint": ""
    }
  ]
}
```

---

### Example 2: SELECT with WHERE clause on sensitive column (Search scenario)

**Input:**
- SQL: `SELECT id, cust_nm, birth_dt FROM customer WHERE cust_nm = #{custNm}`
- Method chain: `SearchController.search()` → `CustomerService.searchByName()` → `CustomerDao.selectByName()`

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Search functionality queries customer table using name as search criteria. Since cust_nm is encrypted in DB, search parameter must be encrypted first to match, then results must be decrypted for display.",
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
        "path": "SearchController.search() → CustomerService.searchByName() → CustomerDao.selectByName() → DB → CustomerService → SearchController → Client",
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
      "reason": "FLOW_001: Search param must be encrypted to match DB data, results must be decrypted for response",
      "target_properties": ["custNm", "birthDt"],
      "insertion_point": "ENCRYPT: Right before customerDao.selectByName() call; DECRYPT: Right after DAO return",
      "code_pattern_hint": "// Before DAO call: encrypt search parameter\nsearchParam.setCustNm(ksignUtil.ksignEnc(\"P017\", searchParam.getCustNm()));\nList<Customer> resultList = customerDao.selectByName(searchParam);\n// After DAO call: decrypt results\nfor (Customer c : resultList) {\n    c.setCustNm(ksignUtil.ksignDec(\"P017\", c.getCustNm()));\n    c.setBirthDt(ksignUtil.ksignDec(\"P018\", c.getBirthDt()));\n}"
    }
  ]
}
```

---

### Example 3: Session data to DB (Session → DB)

**Input:**
- SQL: `INSERT INTO audit_log (user_nm, action, ip_address) VALUES (#{userNm}, #{action}, #{ipAddress})`
- Method chain: `AuditService.logAction()` → `AuditDao.insertLog()`
- Note: `userNm` is retrieved from HTTP Session (already plaintext)

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Audit logging saves user action with user name from session. Session data is already plaintext (was decrypted when user logged in), so only encryption is needed before DB save. DO NOT decrypt session data.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Audit Log Insert (Session → DB)",
        "direction": "INBOUND_TO_DB",
        "data_source": {
          "type": "SESSION",
          "description": "User name retrieved from HTTP session (already plaintext)"
        },
        "data_sink": {
          "type": "DB",
          "description": "INSERT into audit_log table"
        },
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
      "reason": "FLOW_001: Session data is plaintext, must encrypt before DB storage. No decryption needed for session data.",
      "target_properties": ["userNm"],
      "insertion_point": "Right before auditDao.insertLog() call",
      "code_pattern_hint": "logData.setUserNm(ksignUtil.ksignEnc(\"P017\", logData.getUserNm()));"
    }
  ]
}
```

---

### Example 4: DB to External API (DB → External System)

**Input:**
- SQL: `SELECT mem_nm, birth_dt, jumin_no FROM member WHERE id = #{memberId}`
- Method chain: `IntegrationController.sendToPartner()` → `MemberService.getMemberForExport()` → `MemberDao.selectById()` → `ExternalApiClient.sendMemberInfo()`
- Note: External partner system expects plaintext data

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Member data is retrieved from DB and sent to external partner API. External systems expect plaintext, so encrypted DB data must be decrypted before sending to external API.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Member Export to External API",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {
          "type": "DB",
          "description": "SELECT member data (encrypted in DB)"
        },
        "data_sink": {
          "type": "EXTERNAL_API",
          "description": "Partner system API expects plaintext"
        },
        "path": "DB → MemberDao.selectById() → MemberService.getMemberForExport() → ExternalApiClient.sendMemberInfo() → Partner System",
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
      "reason": "FLOW_001: External API expects plaintext. Must decrypt DB data before sending to partner system.",
      "target_properties": ["memNm", "birthDt", "juminNo"],
      "insertion_point": "Right after memberDao.selectById() return, before externalApiClient.sendMemberInfo() call",
      "code_pattern_hint": "memberData.setMemNm(ksignUtil.ksignDec(\"P017\", memberData.getMemNm()));\nmemberData.setBirthDt(ksignUtil.ksignDec(\"P018\", memberData.getBirthDt()));\nmemberData.setJuminNo(ksignUtil.ksignDec(\"P019\", memberData.getJuminNo()));"
    }
  ]
}
```

---

### Example 5: External API to DB (External System → DB)

**Input:**
- SQL: `INSERT INTO external_customer (cust_nm, birth_dt) VALUES (#{custNm}, #{birthDt})`
- Method chain: `WebhookController.receiveCustomer()` → `ExternalCustomerService.saveFromPartner()` → `ExternalCustomerDao.insert()`
- Note: Data received from external partner system (plaintext)

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Customer data received from external partner webhook. External data arrives as plaintext and must be encrypted before storing in our DB.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "External Customer Import",
        "direction": "INBOUND_TO_DB",
        "data_source": {
          "type": "EXTERNAL_API",
          "description": "Partner system sends customer data via webhook (plaintext)"
        },
        "data_sink": {
          "type": "DB",
          "description": "INSERT into external_customer table"
        },
        "path": "Partner System → WebhookController.receiveCustomer() → ExternalCustomerService.saveFromPartner() → ExternalCustomerDao.insert() → DB",
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

### Example 6: DB to Session storage (Login scenario)

**Input:**
- SQL: `SELECT user_id, user_nm, birth_dt FROM users WHERE login_id = #{loginId}`
- Method chain: `LoginController.login()` → `AuthService.authenticate()` → `UserDao.selectByLoginId()` → Session storage
- Note: After successful login, user info is stored in session for later use

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "User authentication retrieves user data from DB and stores in session. Session should store plaintext for easy access throughout the user's session, so DB data must be decrypted before session storage.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "User Login (DB → Session)",
        "direction": "DB_TO_OUTBOUND",
        "data_source": {
          "type": "DB",
          "description": "SELECT user info (encrypted in DB)"
        },
        "data_sink": {
          "type": "SESSION",
          "description": "Store user info in HTTP session (as plaintext)"
        },
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

### Example 7: Session to HTTP Response (No crypto needed)

**Input:**
- Method chain: `ProfileController.getMyProfile()` → retrieves data from Session → returns HTTP response
- Note: No DB access, data comes directly from session

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "User profile is retrieved directly from session and returned to client. Since session already stores plaintext (decrypted during login), NO encryption or decryption is needed.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Profile from Session (No DB)",
        "direction": "SESSION_TO_OUTBOUND",
        "data_source": {
          "type": "SESSION",
          "description": "User profile stored in session (already plaintext)"
        },
        "data_sink": {
          "type": "HTTP_RESPONSE",
          "description": "Return profile to client"
        },
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

### Example 8: UPDATE with WHERE on sensitive column

**Input:**
- SQL: `UPDATE customer SET birth_dt = #{newBirthDt} WHERE cust_nm = #{custNm}`
- Method chain: `CustomerController.updateBirthDt()` → `CustomerService.updateBirthDtByName()` → `CustomerDao.updateBirthDtByName()`

**Output:**
```json
{
  "data_flow_analysis": {
    "overview": "Update customer birthdate by name. Both the WHERE condition (cust_nm) and the SET value (birth_dt) are sensitive columns. Must encrypt both the search parameter and the new value before executing UPDATE.",
    "flows": [
      {
        "flow_id": "FLOW_001",
        "flow_name": "Update BirthDt by Name",
        "direction": "INBOUND_TO_DB",
        "data_source": {
          "type": "HTTP_REQUEST",
          "description": "Client sends name (search) and newBirthDt (update value)"
        },
        "data_sink": {
          "type": "DB",
          "description": "UPDATE customer table"
        },
        "path": "CustomerController.updateBirthDt() → CustomerService.updateBirthDtByName() → CustomerDao.updateBirthDtByName() → DB",
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
      "reason": "FLOW_001: Both WHERE condition (cust_nm) and SET value (birth_dt) must be encrypted before UPDATE",
      "target_properties": ["custNm", "newBirthDt"],
      "insertion_point": "Right before customerDao.updateBirthDtByName() call",
      "code_pattern_hint": "// Encrypt both search param and update value\nupdateParams.setCustNm(ksignUtil.ksignEnc(\"P017\", updateParams.getCustNm()));\nupdateParams.setNewBirthDt(ksignUtil.ksignEnc(\"P018\", updateParams.getNewBirthDt()));"
    }
  ]
}
```

---

## data_flow_analysis Field Details

| Field | Description | Example |
|-------|-------------|---------|
| `overview` | Overview of the entire data flow | "The employee table stores user information..." |
| `flows[].flow_id` | Flow identifier | "FLOW_001", "FLOW_002" |
| `flows[].flow_name` | Flow name (function description) | "Employee Registration", "Employee Retrieval" |
| `flows[].direction` | Data direction | "INBOUND_TO_DB", "DB_TO_OUTBOUND", "BIDIRECTIONAL" |
| `flows[].data_source.type` | Data source type (non-BIDIRECTIONAL) | "HTTP_REQUEST", "SESSION", "DB", "EXTERNAL_API" |
| `flows[].data_sink.type` | Data destination type (non-BIDIRECTIONAL) | "DB", "HTTP_RESPONSE", "SESSION", "EXTERNAL_API" |
| `flows[].INBOUND_TO_DB` | (BIDIRECTIONAL only) Inbound flow with data_source/data_sink | `{data_source: {...}, data_sink: {...}}` |
| `flows[].DB_TO_OUTBOUND` | (BIDIRECTIONAL only) Outbound flow with data_source/data_sink | `{data_source: {...}, data_sink: {...}}` |
| `flows[].path` | Call path (expressed with arrows) | "Controller → Service → DAO → DB" |
| `flows[].sensitive_columns` | Columns requiring encryption | ["emp_nm", "birth_dt", "jumin_no"] |

---

## Start Analysis Now

Based on the information above, **analyze the Data Flow first**, then output modification instructions for each file in JSON format based on that analysis.

**Important**: `data_flow_analysis` is the analysis result of the overall data flow, and `modification_instructions` are specific code modification instructions based on that analysis. Clearly distinguish the roles of these two sections.

**REMINDER: Output ONLY the JSON object. Start directly with `{` and end with `}`. No other text allowed.**
