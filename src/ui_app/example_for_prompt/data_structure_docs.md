# Data Structure Documentation

This document describes the structure and schema of the JSON files located in this directory: `call_graph.json` and `table_access_info.json`. These files serve as examples for prompt generation prompts and understanding the target project structure.

## 1. `call_graph.json`

This file represents the call graph of the application, detailing API endpoints and their corresponding execution flows (call trees) down to the database layer.

### Root Object
| Field | Type | Description |
|---|---|---|
| `endpoints` | Array | List of all identified API endpoints in the application. |
| `node_count` | Integer | Total number of nodes in the graph. |
| `edge_count` | Integer | Total number of edges in the graph. |
| `call_trees` | Array | Detailed call stacks for each entry point (endpoint). |

### `endpoints` Object
Represents a single API endpoint.
| Field | Type | Description |
|---|---|---|
| `path` | String | URI path of the endpoint (e.g., `/api/books`). |
| `http_method` | String | HTTP method (GET, POST, etc.). |
| `method_signature` | String | Signature of the entry method (e.g., `BookController.getBooks`). |
| `class_name` | String | Name of the class containing the endpoint. |
| `method_name` | String | Name of the method. |
| `file_path` | String | Absolute path to the source file. |

### `call_trees` Object (Node)
Represents a node in the call tree. This structure is recursive via the `children` field.
| Field | Type | Description |
|---|---|---|
| `method_signature` | String | Full signature of the method (e.g., `BookService.save`). |
| `layer` | String | Architectural layer (e.g., `Controller`, `Service`, `Repository`, `Mapper`, `Unknown`). |
| `is_circular` | Boolean | Indicates if this call creates a recursion/cycle. |
| `children` | Array | List of child nodes (methods called by this method). |
| `class_name` | String | (Optional) Class name. |
| `file_path` | String | (Optional) Source file path. |
| `line_number` | Integer | (Optional) Start line number of the method. |
| `end_line_number` | Integer | (Optional) End line number of the method. |
| `arguments` | Array | (Optional) List of arguments passed to the method. |
| `endpoint` | Object | (Root Node only) Copy of the endpoint info corresponding to this tree. |

**`arguments` Object:**
*   `name`: Argument name.
*   `type`: Argument data type.

---

## 2. `table_access_info.json`

This file details how the application interacts with database tables.

### Root Object
The root is an **Array** of objects, where each object represents access information for a specific database table.

### Table Access Object
| Field | Type | Description |
|---|---|---|
| `table_name` | String | Name of the database table (e.g., `tb_employee`). |
| `columns` | Array | List of columns accessed or involved. |
| `access_files` | Array | List of all file paths that participate in accessing this table. |
| `query_type` | String | Primary SQL operation type (e.g., `SELECT`, `INSERT`). |
| `sql_query` | String | Representative SQL query string. |
| `layer` | String | Layer where the access is defined (e.g., `Repository`). |
| `sql_queries` | Array | List of specific SQL query definitions mapped to this table. |
| `layer_files` | Object | Files categorized by their architectural layer. |
| `modified_files` | Array | List of files modified (if any context implies modification). |

### `columns` Object
| Field | Type | Description |
|---|---|---|
| `name` | String | Column name. |
| `new_column` | Boolean | Flag indicating if this is a new column proposal (conceptually). |

### `sql_queries` Object
Details individual queries defined in the code (e.g., MyBatis mappers).
| Field | Type | Description |
|---|---|---|
| `id` | String | ID of the query (e.g., mapper method name). |
| `query_type` | String | Type of SQL operation. |
| `sql` | String | The actual SQL statement. |
| `strategy_specific` | Object | Framework-specific metadata (e.g., MyBatis details). |

**`strategy_specific` Object (MyBatis example):**
*   `namespace`: Mapper namespace.
*   `parameter_type`: Java type for parameters.
*   `result_type`: Java type for results.
*   `result_map`: Result map reference.

### `layer_files` Object
A dictionary where keys are layer names and values are lists of file paths.
*   **Keys**: `Repository`, `service`, `controller`, etc.
*   **Values**: `[ "path/to/File.java", ... ]`
