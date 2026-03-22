# Remote API

LLMing-Lodge exposes a public REST API for managing droplets and sending chat messages programmatically. Authentication uses bearer tokens (API keys) stored in MongoDB.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
  - [API Key Format](#api-key-format)
  - [Creating API Keys](#creating-api-keys)
  - [Permissions](#permissions)
  - [Key Storage](#key-storage)
- [Droplet Endpoints](#droplet-endpoints)
  - [List Droplets](#list-droplets)
  - [Get Droplet](#get-droplet)
  - [Create Droplet](#create-droplet)
  - [Update Droplet](#update-droplet)
  - [Publish Droplet](#publish-droplet)
  - [Unpublish Droplet](#unpublish-droplet)
  - [Upload Droplet as ZIP](#upload-droplet-as-zip)
- [Chat Endpoints](#chat-endpoints)
  - [Send Message (SSE Streaming)](#send-message-sse-streaming)
  - [Send Message (Polling)](#send-message-polling)
  - [Poll Task Status](#poll-task-status)
- [Droplet Document Schema](#droplet-document-schema)
- [Remote Task System](#remote-task-system)
- [Setup](#setup)

---

## Base URL

All public API endpoints are prefixed with:

```
/api/llming/v1
```

## Authentication

Every request requires a bearer token in the `Authorization` header:

```
Authorization: Bearer llming_<hex>
```

### API Key Format

Keys use the `llming_` prefix followed by 32 hex characters:

```
llming_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
```

### Creating API Keys

API keys are managed through WebSocket messages in an active chat session.

**Create a key:**

The chat session sends a `apikeys:create` WebSocket message:

```json
{
  "type": "apikeys:create",
  "name": "My CLI Key",
  "permissions": ["manage_droplets", "automate_chat"]
}
```

Response (the full key is only returned once — the client must save it):

```json
{
  "type": "apikeys:created",
  "key_id": "uuid",
  "name": "My CLI Key",
  "full_key": "llming_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  "permissions": ["manage_droplets", "automate_chat"]
}
```

**List keys:**

```json
{"type": "apikeys:list"}
```

Response:

```json
{
  "type": "apikeys:list",
  "keys": [
    {
      "key_id": "uuid",
      "name": "My CLI Key",
      "key_prefix": "llming_a1b",
      "permissions": ["manage_droplets", "automate_chat"],
      "created_at": "2026-03-01T10:00:00Z"
    }
  ]
}
```

**Delete a key:**

```json
{"type": "apikeys:delete", "key_id": "uuid"}
```

### Permissions

| Permission | Grants access to |
|---|---|
| `manage_droplets` | All `/droplets` endpoints (CRUD, publish, ZIP upload) |
| `automate_chat` | `/chat/send` and `/chat/tasks/{id}` endpoints |

A key can have one or both permissions.

### Key Storage

Keys are stored in the `api_keys` MongoDB collection:

| Field | Type | Description |
|---|---|---|
| `key_id` | string | Unique identifier (UUID) |
| `user_email` | string | Owner's email address |
| `name` | string | Human-readable label |
| `key_prefix` | string | First 10 characters (for display) |
| `key_hash` | string | SHA-256 hash of the full key |
| `permissions` | array | List of permission strings |
| `created_at` | datetime | Creation timestamp |

Indexes: `key_id` (unique), `user_email`, `key_hash`.

The server never stores the full key — only the SHA-256 hash. Each request hashes the provided token and looks up the matching `key_hash`.

---

## Droplet Endpoints

### List Droplets

```
GET /api/llming/v1/droplets
Authorization: Bearer llming_...
```

Returns the authenticated user's droplets (dev mode, up to 200).

**Response:**

```json
{
  "droplets": [
    {
      "uid": "my-droplet-uid",
      "name": "Sales Assistant",
      "description": "Helps with sales queries",
      "icon": "trending_up",
      "category": "sales",
      "mode": "dev",
      "creator_email": "alice@example.com",
      "updated_at": "2026-03-01T12:00:00Z"
    }
  ]
}
```

### Get Droplet

```
GET /api/llming/v1/droplets/{uid}
Authorization: Bearer llming_...
```

Returns the full droplet document including files.

### Create Droplet

```
POST /api/llming/v1/droplets
Authorization: Bearer llming_...
Content-Type: application/json

{
  "name": "My Droplet",
  "description": "A helpful assistant",
  "system_prompt": "You are a helpful assistant specialized in...",
  "icon": "smart_toy",
  "category": "general"
}
```

The `type` field defaults to `"nudge"` and `creator_email` is set from the API key's owner.

### Update Droplet

```
PUT /api/llming/v1/droplets/{uid}
Authorization: Bearer llming_...
Content-Type: application/json

{
  "name": "Updated Name",
  "system_prompt": "New system prompt..."
}
```

Only provided fields are merged — omitted fields remain unchanged.

**Updatable fields:**

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name |
| `description` | string | Long-form description |
| `icon` | string | Material Design icon name or emoji |
| `category` | string | Category key |
| `sub_category` | string | Subcategory key |
| `system_prompt` | string | LLM system prompt |
| `model` | string | Default LLM model |
| `language` | string | Language code (e.g. "en", "de") |
| `team_id` | string | Team ownership ID |
| `visibility` | array | Email patterns for access control |
| `suggestions` | array | Quick-action suggestions |
| `capabilities` | array | Feature/tool toggles |

### Publish Droplet

```
POST /api/llming/v1/droplets/{uid}/publish
Authorization: Bearer llming_...
```

Copies the dev version to live. Users matching the `visibility` patterns can now discover and use this droplet.

**Response:**

```json
{"status": "published", "uid": "my-droplet-uid"}
```

### Unpublish Droplet

```
POST /api/llming/v1/droplets/{uid}/unpublish
Authorization: Bearer llming_...
```

Deletes the live version. The dev version is preserved.

**Response:**

```json
{"status": "unpublished", "uid": "my-droplet-uid"}
```

### Upload Droplet as ZIP

```
PUT /api/llming/v1/droplets/{uid}/zip
Authorization: Bearer llming_...
Content-Type: application/zip

<binary ZIP data>
```

Upload a complete droplet as a ZIP file. Use `_new` as the UID to create a new droplet.

```
PUT /api/llming/v1/droplets/_new/zip
```

#### ZIP Structure

```
my-droplet.zip
├── droplet.json           # Manifest (optional)
└── files/                 # Attachment files
    ├── index.js           # JS MCP entry point
    ├── utils.js           # Additional JS modules
    ├── data.csv           # Data file
    └── knowledge.pdf      # Knowledge document
```

**`droplet.json` manifest** — contains any droplet field (name, system_prompt, etc.). The `files` and `_id` fields are ignored in the manifest since files come from the ZIP's `files/` directory.

Example manifest:

```json
{
  "name": "Sales Assistant",
  "description": "Answers sales questions using attached data",
  "system_prompt": "You are a sales assistant. Use the provided tools to query the sales data.",
  "icon": "analytics",
  "category": "sales",
  "mcp_entry_point": "index.js"
}
```

#### Allowed File Extensions

| Category | Extensions |
|---|---|
| Knowledge files | `.pdf`, `.docx`, `.xlsx`, `.txt`, `.md`, `.csv` |
| MCP JavaScript | `.js`, `.mjs` |
| MCP data files | `.csv`, `.tsv`, `.json`, `.xml`, `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`, `.xls`, `.yaml`, `.yml` |

#### Size Limits

| Limit | Value |
|---|---|
| Single file | 5 MB |
| Total files | 10 MB |
| ZIP archive | 15 MB |
| Max file count | 40 |

#### Text Extraction

Binary files (PDF, DOCX, XLSX) are automatically run through text extraction. The extracted text is stored as `text_content` on each file entry and injected into the LLM's context when the droplet is activated. Text-based files (.txt, .md, .csv, .json, etc.) have their content stored directly.

#### File Storage Format

Files are stored in the nudge document as data URIs:

```json
{
  "file_id": "uuid",
  "name": "data.csv",
  "size": 1024,
  "mime_type": "text/csv",
  "content": "data:text/csv;base64,Y29sMSxjb2wyCnZhbDEsdmFsMg==",
  "text_content": "col1,col2\nval1,val2"
}
```

#### Response

```json
{
  "uid": "my-droplet-uid",
  "mode": "dev",
  "files_count": 3,
  "total_size": 45678
}
```

---

## Chat Endpoints

### Send Message (SSE Streaming)

```
POST /api/llming/v1/chat/send?stream=true
Authorization: Bearer llming_...
Content-Type: application/json

{
  "text": "What were our Q1 sales numbers?",
  "droplet_uid": "sales-assistant"
}
```

The `droplet_uid` is optional. If provided, the droplet is activated first (including its MCP tools and system prompt), then the message is sent.

**Response:** Server-Sent Events stream.

```
data: {"type":"chunk","text":"Based on"}

data: {"type":"chunk","text":" the data,"}

data: {"type":"chunk","text":" Q1 sales were $2.1M."}

data: {"type":"done","text":"Based on the data, Q1 sales were $2.1M.","model":"claude-sonnet-4-20250514"}
```

**Event types:**

| Type | Fields | Description |
|---|---|---|
| `chunk` | `text` | Incremental text fragment |
| `done` | `text`, `model` | Complete response with model name |
| `error` | `message` | Error description |

**Timeouts:**
- Droplet selection: 20 seconds
- Response streaming: 120 seconds

**Headers:**

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

### Send Message (Polling)

```
POST /api/llming/v1/chat/send?stream=false
Authorization: Bearer llming_...
Content-Type: application/json

{
  "text": "Summarize the attached report"
}
```

**Response:**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending"
}
```

Use the task ID to poll for results.

### Poll Task Status

```
GET /api/llming/v1/chat/tasks/{task_id}
Authorization: Bearer llming_...
```

**Response (pending):**

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "pending",
  "chunks": []
}
```

**Response (in progress):**

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "processing",
  "chunks": ["Based on", " the data,"]
}
```

**Response (completed):**

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "completed",
  "chunks": ["Based on", " the data,", " Q1 sales were $2.1M."],
  "response": {
    "text": "Based on the data, Q1 sales were $2.1M.",
    "model": "claude-sonnet-4-20250514"
  }
}
```

---

## Droplet Document Schema

A droplet (nudge) document in MongoDB contains:

| Field | Type | Description |
|---|---|---|
| `uid` | string | Unique identifier (user-facing) |
| `mode` | string | `"dev"` or `"live"` (composite key with uid) |
| `type` | string | Always `"nudge"` |
| `creator_email` | string | Creator's email |
| `name` | string | Display name |
| `description` | string | Long-form description |
| `icon` | string | Material Design icon name or emoji |
| `category` | string | Category key |
| `sub_category` | string | Subcategory |
| `system_prompt` | string | Injected into LLM context |
| `model` | string | Default LLM model |
| `language` | string | Language code |
| `team_id` | string | Team ownership |
| `visibility` | array | Email patterns for ACL (e.g. `["*@example.com", "bob@other.com"]`) |
| `suggestions` | array | Quick-action suggestion buttons |
| `capabilities` | dict | Tool name → enabled mapping (see below) |
| `mcp_entry_point` | string | JS entry point filename (default: `"index.js"`) |
| `files` | array | Attached files (see [file storage format](#file-storage-format)) |
| `is_master` | bool | Master nudge flag (requires team_id) |
| `auto_discover` | bool | Auto-discoverable in knowledge base |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last modification timestamp |

### Dev vs. Live

Every droplet exists in two modes:

- **dev** — the working copy, only visible to the creator (and team editors)
- **live** — the published copy, visible to users matching the `visibility` patterns

The `publish` endpoint copies dev → live. The `unpublish` endpoint deletes only the live copy.

### Capabilities

The `capabilities` field controls which tools are enabled when a droplet is activated:

```json
{
  "my_custom_tool": true,
  "another_tool": false,
  "optional_tool": null
}
```

| Value | Meaning |
|---|---|
| `true` | Force tool enabled |
| `false` | Force tool disabled |
| `null` | Use user's global default |

---

## Remote Task System

The chat endpoints use a task queue in MongoDB to bridge REST requests to active browser chat sessions.

### Flow

```
API Client                    MongoDB                     Browser Session
    │                            │                              │
    ├─ POST /chat/send ──────►  insert remote_task             │
    │                           (status: pending)               │
    │                            │                              │
    │                            │  ◄──── poll for pending ─────┤
    │                            │        claim task             │
    │                            │        (status: processing)   │
    │                            │                              │
    │                            │  ◄──── push chunks ──────────┤
    │  ◄── SSE: chunk ──────── read chunks                     │
    │  ◄── SSE: chunk ──────── read chunks                     │
    │                            │                              │
    │                            │  ◄──── set completed ────────┤
    │  ◄── SSE: done ───────── read response                   │
    │                            │                              │
```

### MongoDB Collection: `remote_tasks`

| Field | Type | Description |
|---|---|---|
| `task_id` | string | UUID, unique |
| `user_email` | string | Task owner (from API key) |
| `type` | string | `"send_message"` or `"select_droplet"` |
| `payload` | dict | `{"text": "..."}` or `{"droplet_uid": "..."}` |
| `status` | string | `"pending"` → `"processing"` → `"completed"` or `"error"` |
| `chunks` | array | Streaming text chunks (appended via `$push`) |
| `response` | dict | Final response: `{"text": "...", "model": "..."}` |
| `error` | string | Error message (if status is `"error"`) |
| `created_at` | datetime | Creation timestamp |

**TTL:** Documents auto-delete after 600 seconds (10 minutes).

**Indexes:** `task_id` (unique), `(user_email, status)`, `created_at` (TTL).

### Droplet Selection

When `droplet_uid` is provided in a `/chat/send` request:

1. A `select_droplet` task is inserted first
2. The browser session picks it up, activates the droplet (loads MCP tools, injects system prompt)
3. The API waits up to 20 seconds for the selection task to complete
4. Only then is the `send_message` task inserted
5. If selection times out or fails, the API returns a `504` or `400` error

### Browser-Side Polling

The browser session runs a background task poller that:

1. Atomically claims pending tasks via `find_one_and_update` (status: pending → processing)
2. Executes the task (send message, select droplet)
3. Pushes streaming chunks to the task document as they arrive
4. Sets status to `completed` with the final response when done

---

## Setup

### Mounting the API

The public API is mounted automatically by `setup_routes()` when a `nudge_store` is provided:

```python
from llming_lodge.server import setup_routes
from llming_lodge.nudge_store import NudgeStore

nudge_store = NudgeStore(mongo_uri, mongo_db)
setup_routes(app, nudge_store=nudge_store)
```

Or via `ChatPage` (which calls `setup_routes` internally):

```python
from llming_lodge.chat_page import ChatPage
from llming_lodge.chat_config import ChatAppConfig

chat_page = ChatPage(ChatAppConfig(
    nudge_mongo_uri="mongodb://...",
    nudge_mongo_db="my_app",
    ...
))
chat_page.ensure_routes()  # Call at startup, before NiceGUI's catch-all
```

### Required Infrastructure

- **MongoDB** — stores nudges, API keys, and remote tasks
- **Active browser session** — chat endpoints require a user to have the chat page open in a browser (the browser session processes remote tasks)

### curl Examples

**Create an API key** (via chat WebSocket, not REST).

**List droplets:**

```bash
curl https://example.com/api/llming/v1/droplets \
  -H "Authorization: Bearer llming_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
```

**Upload a droplet from a ZIP file:**

```bash
curl -X PUT https://example.com/api/llming/v1/droplets/_new/zip \
  -H "Authorization: Bearer llming_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" \
  -H "Content-Type: application/zip" \
  --data-binary @my-droplet.zip
```

**Send a chat message (streaming):**

```bash
curl -N https://example.com/api/llming/v1/chat/send \
  -H "Authorization: Bearer llming_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, what can you do?", "droplet_uid": "my-assistant"}'
```

**Send a chat message (polling):**

```bash
# Submit
TASK_ID=$(curl -s "https://example.com/api/llming/v1/chat/send?stream=false" \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize the report"}' | jq -r .task_id)

# Poll
curl "https://example.com/api/llming/v1/chat/tasks/$TASK_ID" \
  -H "Authorization: Bearer llming_..."
```
