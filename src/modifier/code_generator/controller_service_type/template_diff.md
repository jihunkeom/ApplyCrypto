# Java Source Code Privacy Data Encryption Modification Task

## Role and Objective
You are an expert Java developer specializing in Spring Framework applications. Your task is to modify Java source code to add encryption/decryption calls for personal information (주민번호/SSN, 성명/Name, 생년월일/Birth Date) while preserving all other code unchanged. Output the unified diff of the modified source code. Do NOT output the full source code.

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
#### Types of modification
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

#### Modification Steps
Source code modifications must be approached through the following step-by-step process of thinking, execution, and verification:

1. Identify candidate codes.
First, you must identify candiate codes for change. The methods in the call stacks provided in "Call Stacks Information" section in below must be candidate codes. In those methods, there may be or may not be code blocks where variables inferred from column names specified in the "Table Column Information" section in below are used. For each selected candidate code, proceed with the modification work through steps 2-4 below.

2. Determin data flow type of the candidate codes.
Determine whether the data object used in the candidate code belongs to downstream or upstream. The process for making this determination is described in the sub-steps below.

2-1. The application is a backend application written in Java, and the framework can vary, including Spring, Anyframe, etc. The source code for each framework is divided into upper layer, middle layer, and lower layer. For example, as follows:

2-2. In the case of Spring framework, controller source files belong to the upper layer, service/service implementation belongs to the middle layer, and mapper or external interface source files belong to the lower layer. Among the lower layers, mapper is related to the database while external interface is not related to the database.

2-3. In the case of Anyframe framework, service/service implementation source files belong to the upper layer, business source files belong to the middle layer, and dem/dqm or external interface source files belong to the lower layer. Among the lower layers, dem/dqm is related to the database while external interface is not related to the database.

2-3. Downstream data flow means that data processing occurs as it is passed from top to bottom in the form of upper layer → middle layer → lower layer. In this case, the upper layer becomes the source layer and the lower layer becomes the destination layer. For example, if the data flow is downstream in Spring framework, the source layer is the controller and the destination layer is the mapper or external interface layer. Conversely, if it's upstream, the source layer is the mapper or external interface and the destination is the controller layer. The same approach applies to other frameworks. In Anyframe, if the data flow is downstream, the source layer is service/service implementation and the destination layer is dem/dqm or external interface. Conversely, if it's upstream, the source layer is dem/dqm or external interface layer and the destination layer is service/service implementation.

2-4. To determine whether the data flow processed in the candidate code belongs to downstream or upstream, you must identify the call relationships of the method containing the candidate code and verify the direction in which the data object is being passed. Perform this verification by comprehensively understanding the provided source code. Note that the method call relationships and data flow directions can differ.

Once the data flow of the candidate code is confirmed, perform the modification work according to the following steps:

3-1. If the data flow of the candidate code is downstream, determine whether the destination layer is a database-related layer. If it corresponds to this, proceed with modifications for encryption. If the destination layer is a layer unrelated to the database, no modifications should be made.

3-2. If the data flow of the candidate code is upstream, determine whether the source layer is a database-related layer. If it corresponds to this, proceed with modifications for decryption. If the source layer is a layer unrelated to the database, no modifications should be made.

3-3. When getter/setter methods need to be used for source code modification, you must use accurate method names. In order to do this, you have to examine provided DTO/DAO/VO class files.

4. Determine layers to change.
Applying encryption/descryption codes must not be duplacated accross layer source files. In candidate codes, if you find usage of variables inferred from column names specified in the "Table Column Information", chaning that layer file is preferred. If you can't find such codes over the layer files, the middle layer file is preferred for change. You can use each call stack provided in the "Call Stacks Information"

For example, in the case of Spring framework, the selection and modification of candidate code should be applied in either the controller layer or the service/service implementation layer, and should not occur redundantly in both layers. Similarly, in the case of Anyframe, it should be applied in either the service/service implementation layer or the business layer, and should not be applied redundantly in both layers.


**DO NOT modify:**
- Code unrelated to the specified tables and columns
- Controller layer (unless absolutely necessary)
- Repository layer (unless absolutely necessary)
- Import statements, class declarations, or method signatures
- Comments, logging statements, or validation logic


### 5. Output Requirements

If there is change that you applied in the input source file, you have to generate a Unified Diff representing the changes.
Do not output the full source file, only the diff.

### 6. Output Format Requirements
**Output Format**
For EACH input source file, you MUST output in the following format using delimiters:

```
======FILE======
{filename only, e.g., EmployeeService.java}
======REASON======
{Brief explanation of the modification or why modification is not required}
======MODIFIED_CODE======
{If modified: the Unified Diff of the changes}
{If not modified: leave this section empty}
======END======
```

The 'modification' should keep the same number of input source files in its list.
If there are modifications, ensure that "modified_code" contains the unified diff.
If you decide not to change any input source files, you must leave the "MODIFIED_CODE" section empty.

**Unified Diff Format Example:**
```diff
--- a/src/main/java/com/example/UserService.java
+++ b/src/main/java/com/example/UserService.java
@@ -32,7 +32,7 @@
     public void saveUser(UserDTO user) {
-        userMapper.insertUser(user);
+        user.setName(k_sign.CryptoService.encrypt(user.getName(), k_sign.CryptoService.P20, K_SIGN_NAME));
+        userMapper.insertUser(user);
     }
```

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
**After (Unified Diff):**
```diff
@@ -1,3 +1,5 @@
 public void saveUser(User user) {
+    user.setName(k_sign.CryptoService.encrypt(user.getName(), k_sign.CryptoService.P20, K_SIGN_NAME));
+    user.setDob(k_sign.CryptoService.encrypt(user.getDob(), k_sign.CryptoService.P30, K_SIGN_DOB));
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
**After (Unified Diff):**
```diff
@@ -2,4 +2,6 @@
     User user = userDao.findById(id);
-    return user;
+    if (user != null) {
+        user.setName(k_sign.CryptoService.decrypt(user.getName(), k_sign.CryptoService.P20, K_SIGN_NAME));
+        user.setDob(k_sign.CryptoService.decrypt(user.getDob(), k_sign.CryptoService.P30, K_SIGN_DOB));
+    }
+    return user;
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
**After (Unified Diff):**
```diff
@@ -2,1 +2,1 @@
-    user.setJumin(k_sign.CryptoService.encrypt(user.getJumin(), k_sign.CryptoService.P03, K_SIGN_SSN));
+    user.setJumin(k_sign.CryptoService.encrypt(user.getJumin(), k_sign.CryptoService.P10, K_SIGN_JUMIN));
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
**After (Unified Diff):**
```diff
@@ -4,1 +4,1 @@
-        user.setJumin(k_sign.CryptoService.decrypt(user.getJumin(), k_sign.CryptoService.P03, K_SIGN_SSN));
+        user.setJumin(k_sign.CryptoService.decrypt(user.getJumin(), k_sign.CryptoService.P10, K_SIGN_JUMIN));
```
**Explanation:** Change k_sign.CryptoService.P03 to k_sign.CryptoService.P10 and K_SIGN_SSN to K_SIGN_JUMIN.

From here, there are actual information and source codes that you have to handle
## Table Column Information
{{ table_info }}

## Source Files to Modify
{{ source_files }}

## Current Layer: {{ layer_name }}

## File Count: {{ file_count }}

## Call Stacks Information
The following call stacks show the method call relationships from the upper layer to lower layer methods. Methods of the upper and middle layer in each call stack should be candiate codes for applying encyption/decryption codes. You also have to use this information to understand the data flow direction when making modifications.
{{ call_stacks }}

## Warnings
1. Do not change the logic of existing code.
2. Only add encryption and decryption code.
3. The file_path must use the absolute path provided in source_files.
4. Do NOT perform any linting or formatting changes such as removing comments, trimming whitespace, or reformatting code. Only modify what is strictly necessary for encryption/decryption.
5. Do not remove or insert carrige return at the end of each source file. It should be as it is.

