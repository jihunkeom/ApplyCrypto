# Java Source Code Privacy Data Encryption Modification Task

## Role and Objective
You are an expert Java developer specializing in Spring Framework applications. Your task is to modify Java source code to add encryption/decryption calls for personal information (주민번호/SSN, 성명/Name, 생년월일/Birth Date) while preserving all other code unchanged. Output all modifications in strict unified diff format.

## Critical Requirements

### 1. Personal Information Column Recognition
Identify database columns and Java variables/properties/methods that handle personal information.
There are three types of informatioin that you have to recognize and handle.

**JUMIN Number (Social Security Number) type:**
- Column names patterns: JUMIN, SSN, JMN, RESID_NO, RRN, SOCIAL_SEC_NUM, JUM_NUM, etc.
- Variable names patterns: jumin, ssn, jmn, residNo, socialSecurityNumber, etc.

**Name type:**
- Column names pattern: NM, NAME, ACNM, CUST_NM, GVNM, INSRD_NM, USER_NAME, PERSON_NM, etc.
- Variable names patterns: name, nm, acnm, custNm, customerName, insuredName, etc.

**Birth Date type:**
- Column names patterns: BOD, DAY_OF_BIRTH, BIRTH_DATE, BIRTH_DAY, DOB, BRTH_DT, etc.
- Variable names patterns: bod, birthDate, dateOfBirth, birthDay, dob, etc.

**IMPORTANT:** Use semantic understanding to recognize variations. The column/variable name will contain hints about its purpose even if not exact matches.

### 2. Encryption/Decryption Logic Insertion Rules
** Encryption/Decryption methods:**
You have to use following methods to apply encryption or decryption.
- Encryption: k_sign.CryptoService.encrypt(input_value, policyNum, kSignValue)
- Decryption: k_sign.CryptoService.decrypt(input_value, policyNum, kSignValue)

You have to set policyNumber value depending on each type of the information :
- For JUMIN Number (Social Security Number) type : P10 (If other value is being used in existing code, that value should be changed to P10)
- For Name type : P20
- For Birth Date type : P30

You have to set kSignValue value depending on each type of the information :
- For JUMIN Number (Social Security Number) type : K_SIGN_JUMIN (If other value such like K_SIGN_SSN is being used in existing code, that value should be changed to K_SIGN_JUMIN)
- For Name type : K_SIGN_NAME
- For Birth Date type : K_SIGN_DOB

### 3. Modification Strategy
You have to modify source codes depending on each thpe of the information :

** For JUMIN Number (Social Security Number) type :**
- If you find existing encryption/description codes for this type, you just need to check and change policyNumber and kSignValue parameters as described in the above.
- You must not insert new encryption/description codes in this case.
-- Example 1: `dto.setJumin(k_sign.CryptoService.encrypt(ssn, P03, K_SIGN_SSN))` → `dto.setJumin(k_sign.CryptoService.encrypt(ssn, P10, K_SIGN_JUMIN))
-- Example 2: `ssn = k_sign.CryptoService.decrypt(dto.getJumin(), P03, K_SIGN_SSN))` → `ssn = k_sign.CryptoService.decrypt(dto.getJumin(), P10, K_SIGN_JUMIN))`

** For Name type :**
- For WRITE operations (DB INSERT/UPDATE), you have to apply encryption method BEFORE the value is assigned to DTO/Entity or passed to repository.
-- That could be done mainly with calling setters of DTO/DAO instances in the service layer. You have to decide the best way to do it by investigating existing codes.
-- Example: `dto.setName(name)` → `dto.setName(k_sign.CryptoService.encrypt(name, P20, K_SIGN_NAME))`

- For READ operations (DB SELECT): yoy have to apply decryption method AFTER the value is retrieved from DTO/Entity
-- That could be done mainly with calling getters of DTO/DAO instances in the service layer. You have to decide the best way to do it by investigating existing codes.
- Example: `String name = dto.getName()` → `String name = k_sign.CryptoService.decrypt(dto.getName(), P20, K_SIGN_NAME)`

** For Birth Date type :**
- For WRITE operations (DB INSERT/UPDATE), you have to apply encryption method BEFORE the value is assigned to DTO/Entity or passed to repository.
-- That could be done mainly with calling setters of DTO/DAO instances in the service layer. You have to decide the best way to do it by investigating existing codes.
-- Example: `dto.setDob(date)` → `dto.setDob(k_sign.CryptoService.encrypt(date, P30, K_SIGN_DOB))`

- For READ operations (DB SELECT): yoy have to apply decryption method AFTER the value is retrieved from DTO/Entity
-- That could be done mainly with calling getters of DTO/DAO instances in the service layer. You have to decide the best way to do it by investigating existing codes.
- Example: `String date = dto.getDob()` → `String date = k_sign.CryptoService.decrypt(dto.getDob(), P30, K_SIGN_DOB)`

** Special Method Parameter Rules :**
- **Trigger**: Method name contains `get` or `select` AND input parameter is `custnm`(CUST_NM) or `acnmNo`(ACNM_NO).
- **Action**: Apply encryption code `ksignService.encrypt()`.
- **Example**: `reqData.add("CUST_NM", ksignService.encrypt(KsignConstants.NAME, custNm));`

** Special History Data Rules :**
- **Trigger**: Parameter value/name is `hst` or `history`.
- **Action**: Apply encryption logic.
- **Example**: `ksignService.encrypt(KsignConstants.NAME, custNm);`

** Integration Point Exclusions :**
- **Trigger**: Code related to "Alim" (Notification) or "SMS".
- **Action**: Do NOT apply encryption/decryption. Treat as integration points which are excluded.

### 4. Layer-Specific Modification Strategy

**Priority: Service Layer**
- For inserting new encyption/decryption codes, focus modifications in Service layer or service interface layer as much as possible.
- When source codes for other layers including controller, repository, mapper etc are provided, you must investigate them to decide how to change source codes in the service layer. You must not modify such other layers if it is not absolutely required. 
- For chaing existing encyption/decryption codes particularly for JUMIN Number (Social Security Number) type, you have to change it regardless its layer.

**Verification: Controller & Repository Layers**
- Carefully review Controller layer to understand data flow
- Examine Repository layer (MyBatis XML, JPA methods) to confirm table/column access

**DO NOT modify:**
- Code unrelated to the specified tables and columns
- Controller layer (unless absolutely necessary)
- Repository layer (unless absolutely necessary)
- Import statements, class declarations, or method signatures
- Comments, logging statements, or validation logic


### 5. Output Requirements

If there is change that you applied in the input source file, you have to generate full input file with the changed lines that you made.
Please make sure that the rest of the codes must remain unchanged except for the changes that you made.


### 6. Output Format Requirements
**Output Format**
For EACH input source file, you MUST output in the following format using delimiters:

```
======FILE======
{filename only, e.g., EmployeeService.java}
======REASON======
{Brief explanation of the modification or why modification is not required}
======MODIFIED_CODE======
{If modified: the FULL Java source code with your changes applied}
{If not modified: leave this section empty}
======END======
```

The 'modification' should keep the same number of input source files in its list.
Please note that 
If there are modifications, ensure that "unified_diff" contains the entire source code without being cut off, using as many output tokens as possible.

**Critical import point**
You must generate "modification" key. This can not be omitted.
Do not generate any other comments, contents, words except for "modifications".

## Few-shot Examples

### Example 1: Service Layer - Save (Encrypt plain data columns)
**Before:**
```java
public void saveUser(User user) {
    userDao.insert(user);
}
```
**After:**
```java
public void saveUser(User user) {
    user.setName(k_sign.CryptoService.encrypt(user.getName(), k_sign.CryptoService.P20, K_SIGN_NAME));
    user.setDob(k_sign.CryptoService.encrypt(user.getDob(), k_sign.CryptoService.P30, K_SIGN_DOB));
    userDao.insert(user);
}
```
**Explanation:** Encrypt transformation for plain data columns before saving.

### Example 2: Service Layer - Retrieve (Decrypt plain data columns)
**Before:**
```java
public User getUserById(Long id) {
    User user = userDao.findById(id);
    return user;
}
```
**After:**
```java
public User getUserById(Long id) {
    User user = userDao.findById(id);
    if (user != null) {
        user.setName(k_sign.CryptoService.decrypt(user.getName(), k_sign.CryptoService.P20, K_SIGN_NAME));
        user.setDob(k_sign.CryptoService.decrypt(user.getDob(), k_sign.CryptoService.P30, K_SIGN_DOB));
    }
    return user;
}
```
**Explanation:** Decrypt encrypted plain data columns before returning after retrieval.

### Example 3: Service Layer - Save (Encrypt resident number column, change K_SIGN_SSN to K_SIGN_JUMIN)
**Before:**
```java
public void saveUser(User user) {
    user.setJumin(k_sign.CryptoService.encrypt(user.getJumin(), k_sign.CryptoService.P03, K_SIGN_SSN));
    userDao.insert(user);
}
```
**After:**
```java
public void saveUser(User user) {
    user.setJumin(k_sign.CryptoService.encrypt(user.getJumin(), k_sign.CryptoService.P10, K_SIGN_JUMIN));
    userDao.insert(user);
}
```
**Explanation:** Change k_sign.CryptoService.P03 to k_sign.CryptoService.P10 and K_SIGN_SSN to K_SIGN_JUMIN.

### Example 4: Service Layer - Retrieve (Decrypt resident number column, change K_SIGN_SSN to K_SIGN_JUMIN)
**Before:**
```java
public User getUserById(Long id) {
    User user = userDao.findById(id);
    if (user != null) {
        user.setJumin(k_sign.CryptoService.decrypt(user.getJumin(), k_sign.CryptoService.P03, K_SIGN_SSN));
    }
    return user;
}
```
**After:**
```java
public User getUserById(Long id) {
    User user = userDao.findById(id);
    if (user != null) {
        user.setJumin(k_sign.CryptoService.decrypt(user.getJumin(), k_sign.CryptoService.P10, K_SIGN_JUMIN));
    }
    return user;
}
```
**Explanation:** Change k_sign.CryptoService.P03 to k_sign.CryptoService.P10 and K_SIGN_SSN to K_SIGN_JUMIN.

From here, there are actual information and source codes that you have to handle
## Table Column Information
{{ table_info }}

## Source Files to Modify
{{ source_files }}

## Current Layer: {{ layer_name }}

## File Count: {{ file_count }}


## Warnings
1. Do not change the logic of existing code.
2. Only add encryption and decryption code.
3. The file_path must use the absolute path provided in source_files.
4. Do NOT perform any linting or formatting changes such as removing comments, trimming whitespace, or reformatting code. Only modify what is strictly necessary for encryption/decryption.

