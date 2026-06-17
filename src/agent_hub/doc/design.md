# Agent Hub 工程设计

## 1. 设计目标

Agent Hub 是 `agent_core` 的本地桌面产品层。设计时必须保持一个清晰边界：

- `agent_core` 负责 agent 运行语义、上下文树、Task DAG、worker 调度、工具执行、权限策略、provider 适配和持久化运行历史。
- `agent_hub.backend` 负责把 core 能力包装成桌面应用需要的 HTTP/SSE API、Hub UI 数据、权限桥接和进程生命周期。
- `agent_hub.frontend` 负责 IM 风格交互、worker 管理、消息展示、权限卡片、会话切换和 branch/fork 操作。

这个设计的关键约束是：Hub 不绕过 core。尤其是 `@worker`，它不是直接私聊 worker，而是给 Orchestrator 一个强约束，让 Orchestrator 必须把相关任务派给该 worker。

## 2. 总体架构

```text
┌──────────────────────────────────────────────────────────────────┐
│ Electron App                                                     │
│                                                                  │
│  ┌──────────────────────┐      HTTP/SSE       ┌───────────────┐  │
│  │ React Renderer        │ <────────────────> │ FastAPI Backend│  │
│  │ - conversations       │                    │ - API routes   │  │
│  │ - message stream      │                    │ - SSE hub      │  │
│  │ - worker panel        │                    │ - Hub DB       │  │
│  │ - permission cards    │                    │ - core bridge  │  │
│  └──────────────────────┘                    └───────┬───────┘  │
│                                                        │          │
│  ┌──────────────────────┐                              │          │
│  │ Electron Main         │ starts/stops                 │          │
│  │ - spawn backend       │                              │          │
│  │ - health wait         │                              │          │
│  │ - app lifecycle       │                              │          │
│  └──────────────────────┘                              │          │
└────────────────────────────────────────────────────────┼──────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ agent_core                                                       │
│ - AgentRuntime                                                   │
│ - Session/context tree                                           │
│ - Orchestrator mode                                              │
│ - AgentDefinition registry                                       │
│ - WorkerPoolRuntime                                              │
│ - Task DAG service                                               │
│ - Tool registry / permissions / hooks                            │
│ - Provider adapters                                              │
│ - SQLite session store                                           │
└──────────────────────────────────────────────────────────────────┘
```

Runtime ownership:

- Electron main process owns the Python backend process.
- FastAPI backend owns a long-lived `AgentRuntime`.
- Frontend never imports Python or calls `agent_core` directly.
- Frontend subscribes to SSE for conversation updates and sends commands through HTTP.

## 3. 目录结构

Planned layout:

```text
src/agent_hub/
  __init__.py
  doc/
    requirement.md
    design.md
  backend/
    __init__.py
    __main__.py
    app.py
    config.py
    database.py
    events.py
    permissions.py
    runtime.py
    schemas.py
    services/
      __init__.py
      conversations.py
      messages.py
      workers.py
      tools.py
      branch.py
    routes/
      __init__.py
      health.py
      conversations.py
      workers.py
      tools.py
      events.py
      permissions.py
  frontend/
    package.json
    tsconfig.json
    vite.config.ts
    index.html
    electron/
      main.ts
      preload.ts
      backendProcess.ts
    src/
      App.tsx
      main.tsx
      api/
        client.ts
        events.ts
        types.ts
      components/
        layout/
        conversations/
        messages/
        workers/
        permissions/
        input/
      state/
        appStore.ts
        eventReducer.ts
      styles/
        base.css
        theme.css
```

Repository helper:

```text
soong-hub
```

`soong-hub` 只负责开发期快速启动，不承担安装包职责。

## 4. 启动与生命周期

### 4.1 Backend 直接启动

```bash
PYTHONPATH=src python3 -m agent_hub.backend
```

Backend responsibilities:

1. Resolve `SOONG_AGENT_HOME` or default `~/.soong-agent`.
2. Ensure `config.toml` exists.
3. Open Hub DB.
4. Construct `AgentRuntime`.
5. Initialize core runtime lazily or eagerly during startup.
6. Load worker/agent definitions from built-in, config, JSON, dynamic DB.
7. Register FastAPI routes.
8. Serve API and SSE.

### 4.2 Electron 启动

Electron main process:

1. Finds repository root in dev mode.
2. Finds a free local port or uses configured port.
3. Spawns backend:

```bash
PYTHONPATH=src python3 -m agent_hub.backend --host 127.0.0.1 --port <port>
```

4. Polls `GET /health` until ready.
5. Opens BrowserWindow.
6. Passes backend base URL to renderer through preload.
7. On app quit, sends graceful shutdown or kills backend after timeout.

### 4.3 Config bootstrap

Hub backend must create default config only when missing:

```text
~/.soong-agent/config.toml
```

Rules:

- Existing config is never overwritten.
- Default config should come from core asset template, not duplicated literal strings in Hub.
- If default config creation fails, backend startup fails with a clear error.
- If config exists but is invalid, backend startup fails; Hub should not silently replace it.

### 4.4 Shutdown

Graceful shutdown sequence:

1. Stop accepting new send requests.
2. Mark outstanding Hub requests as shutting down.
3. Let active core runs cancel or finish according to core cancellation semantics.
4. Resolve pending permission futures as denied/cancelled if process is exiting.
5. Close core runtime.
6. Close Hub DB.
7. Exit backend process.

V1 may use a simple implementation, but the API boundaries should allow this sequence.

## 5. Core 扩展设计

Hub 需要新增少量 core 能力。原因是这些能力属于 core 语义，不应该只在 UI 或 backend 做软约束。

### 5.1 Run directives

Current `AgentRuntime.start`:

```python
async def start(
    self,
    message: str | UserMessage,
    session_id: str | None = None,
    mode: Literal["normal", "orchestrator"] = "normal",
) -> RunHandle:
```

Add:

```python
async def start(
    self,
    message: str | UserMessage,
    session_id: str | None = None,
    mode: Literal["normal", "orchestrator"] = "normal",
    directives: RunDirectives | dict[str, Any] | None = None,
) -> RunHandle:
```

Recommended type:

```python
class MentionedWorkerDirective(StrictModel):
    worker_id: str
    worker_agent_id: str
    worker_pool_id: str
    name: str
    agent_definition_id: str


class RunDirectives(StrictModel):
    mentioned_worker: MentionedWorkerDirective | None = None
```

Runtime storage:

- `RunHandle` gets `_directives`.
- `store.create_run` metadata should include sanitized directives.
- Runtime context should expose directives to tool execution context.
- Provider prompt composer receives an internal system/context block for directives.

Directive prompt block:

```text
The user explicitly mentioned worker `<worker_id>`.
You are the Orchestrator. You must dispatch the relevant work to this worker by using
`agent.dispatch_worker`. Do not dispatch to a different worker. If this worker cannot
perform the task, explain why. After the worker completes, summarize the result.
```

Important:

- This directive is not visible as a normal user message.
- It is persisted as run metadata for replay/debug.
- It must also be enforced in tool execution, because prompt-only constraints are insufficient.

### 5.2 Dispatch constraint enforcement

Current `agent.dispatch_worker` calls `runtime.run_worker_agent(...)`.

Add validation before launching worker:

```python
def validate_dispatch_against_directives(
    *,
    directives: RunDirectives | None,
    args: dict[str, Any],
    workers: WorkerPoolRuntime,
    session_id: str,
) -> DispatchValidationResult:
```

Rules when `mentioned_worker` exists:

- If `worker_agent_id` is present, it must equal the mentioned `worker_agent_id`.
- If `worker_id` is present, it must equal the mentioned `worker_id`.
- If `worker_pool_id` is present, it must match mentioned `worker_pool_id`.
- If no worker identity is supplied, core fills `worker_agent_id` from directive.
- Dispatching any other worker returns `validation_error` or a dedicated `worker_constraint_violation`.
- The error is returned as a tool result/error, not a Python crash.

This makes `@worker` a hard runtime contract.

### 5.3 Worker mention resolution

Core should expose an API so Hub does not duplicate worker lookup rules:

```python
class WorkerMentionResolution(StrictModel):
    worker_id: str
    worker_agent_id: str
    worker_pool_id: str
    name: str
    agent_definition_id: str
    status: str


def resolve_worker_mention(
    self,
    mention: str,
    *,
    session_id: str,
) -> WorkerMentionResolution:
```

Resolution priority:

1. exact `worker_id`
2. unique `name`
3. error if ambiguous
4. error if missing/disabled/deleted

The returned `worker_agent_id` is session-specific because current worker agent id is derived from `session_id + worker_id`.

### 5.4 Dynamic worker management

Core should own dynamic worker config because:

- `agent.list_workers` must see it.
- `agent.dispatch_worker` must enforce it.
- runtime model/tool resolution depends on it.
- JSON/TOML/SQLite precedence belongs near core config loading.

Proposed runtime API:

```python
def list_worker_configs(self, include_disabled: bool = True) -> list[WorkerConfigView]: ...

def get_worker_config(self, worker_id: str) -> WorkerConfigView | None: ...

def create_worker_config(self, request: WorkerConfigCreate) -> WorkerConfigView: ...

def update_worker_config(self, worker_id: str, request: WorkerConfigUpdate) -> WorkerConfigView: ...

def disable_worker_config(self, worker_id: str) -> WorkerConfigView: ...

def enable_worker_config(self, worker_id: str) -> WorkerConfigView: ...

def soft_delete_worker_config(self, worker_id: str) -> WorkerConfigView: ...

def reload_worker_configs(self) -> None: ...
```

After create/update/delete:

1. Persist config.
2. Rebuild effective AgentDefinition registry overlay.
3. Rebuild or mutate `WorkerPoolRuntime`.
4. Re-register agent tools if necessary.
5. Emit inspect/debug metadata.

### 5.5 Dynamic worker persistence

Existing core session store is SQLite. Dynamic worker config can be persisted in the same core SQLite database or in a core-owned adjacent SQLite file. The recommended implementation is core SQLite tables in the existing store because worker config affects core behavior.

Tables:

```sql
CREATE TABLE IF NOT EXISTS agent_definitions_dynamic (
  agent_definition_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  model_profile TEXT,
  model_json TEXT,
  system_prompt TEXT NOT NULL DEFAULT '',
  suggested_tools_json TEXT NOT NULL DEFAULT '[]',
  tags_json TEXT NOT NULL DEFAULT '[]',
  enabled INTEGER NOT NULL DEFAULT 1,
  deleted_at TEXT,
  source TEXT NOT NULL DEFAULT 'hub',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_configs_dynamic (
  worker_id TEXT PRIMARY KEY,
  worker_pool_id TEXT NOT NULL DEFAULT 'default',
  agent_definition_id TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  allowed_tools_json TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  deleted_at TEXT,
  source TEXT NOT NULL DEFAULT 'hub',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

Soft delete:

- `enabled = 0`
- `deleted_at = current timestamp`
- keep rows for historical display and replay.

### 5.6 JSON file loading

Core currently loads markdown agent definitions from:

```text
~/.soong-agent/agents/*.md
```

Add:

```text
~/.soong-agent/agents/*.json
~/.soong-agent/workers/*.json
```

JSON agent definition maps to `AgentDefinition`.

JSON worker config maps to worker pool entries and optional inline AgentDefinition.

Precedence:

```text
dynamic SQLite > user JSON > config.toml > built-in
```

Conflict rules:

- Same source duplicate id is an error.
- Higher precedence source can override lower precedence source.
- Override metadata should record final source and overridden source.
- Internal-only definitions such as `default_compact_agent` cannot be overridden.

### 5.7 Direct model config

Current core already has model config/profile concepts. Dynamic definitions need to support:

```json
{
  "model": {
    "provider": "openai",
    "name": "qwen2.5:7b",
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "temperature": 0.2
  }
}
```

Design rule:

- `model_profile` references a named profile.
- `model` is inline direct config.
- If both exist, inline `model` wins.

Implementation option:

1. Extend model resolver to accept inline model config directly.
2. Or synthesize a runtime model profile id such as `dynamic:<agent_definition_id>`.

Recommendation:

- Use direct resolver support if the model resolver already accepts inline partial model configs.
- Otherwise synthesize runtime profile ids as a smaller first implementation.

Security:

- Prefer `api_key_env`.
- If direct `api_key` is supported, never log it and never expose it through SSE.

### 5.8 Worker queue

Existing `WorkerPoolRuntime.select_worker()` raises `WORKER_BUSY` when a specified worker is busy.

For explicit mentioned worker dispatch, new behavior:

- Worker idle: start immediately.
- Worker busy: enqueue.
- Queue full: return `worker_queue_full`.
- Worker completes: dequeue next item.
- Queue item cancelled before start: skip.
- Queue is in-memory only.

Hard-coded V1 limit:

```python
MAX_WORKER_QUEUE_SIZE = 20
```

Recommended structures:

```python
@dataclass
class WorkerQueueItem:
    queue_id: str
    worker_id: str
    worker_agent_id: str
    session_id: str
    parent_run_id: str
    parent_agent_id: str
    task_id: str
    instruction: str
    worker_pool_id: str | None
    allowed_step_ids: list[str] | None
    dispatch_context: str | None
    constraints: dict[str, Any] | None
    allowed_tools: list[str] | None
    expected_output_schema: dict[str, Any] | None
    timeout_ms: int | None
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

Methods:

```python
def enqueue_worker_run(item: WorkerQueueItem) -> WorkerQueueItem: ...
def cancel_worker_queue_item(queue_id: str) -> bool: ...
def list_worker_queue(worker_id: str) -> list[WorkerQueueItem]: ...
async def start_next_worker_queue_item(worker_id: str) -> None: ...
```

Queue events should be emitted into core run streams and surfaced by Hub SSE:

- `worker_queued`
- `worker_dequeued`
- `worker_started`
- `worker_completed`
- `worker_failed`
- `worker_cancelled`

### 5.9 Session queue vs worker queue

Current runtime already has a per-session run queue:

```python
self._session_active
self._session_queues
```

This remains separate from worker queue.

Meaning:

- Session queue: multiple user messages in one core session.
- Worker queue: multiple dispatches targeting one busy worker.

Hub input can allow the user to keep typing while a run is active. Backend can create messages and rely on core session queue for same-session ordering. Worker-specific contention is handled by worker queue.

### 5.10 Branch/fork core APIs

Hub needs user-message node based APIs. Core already has session tree concepts; Hub should not edit DB directly.

Proposed APIs:

```python
async def list_branchable_nodes(
    self,
    session_id: str,
    *,
    node_type: Literal["user_message"] = "user_message",
    limit: int = 100,
) -> list[BranchableNodeView]: ...

async def set_active_path_from_node(
    self,
    session_id: str,
    node_id: str,
) -> ActivePathView: ...

async def fork_session_from_node(
    self,
    session_id: str,
    node_id: str,
    *,
    new_session_id: str | None = None,
) -> ForkSessionResult: ...
```

Hub should list only user message nodes by default.

Branch:

- Same core session.
- Updates active path.
- Hub conversation remains the same.

Fork:

- New core session.
- New Hub conversation.
- Copies or references context according to core implementation.

## 6. Hub Backend 设计

### 6.1 Backend modules

`app.py`

- Create FastAPI app.
- Wire routes.
- Configure startup/shutdown hooks.
- Store app state references.

`config.py`

- Resolve home/config paths.
- Bootstrap default config if missing.
- Read minimal config status for health.

`database.py`

- Open Hub SQLite DB.
- Run migrations.
- Provide async/sync repository helpers.

`runtime.py`

- Own `AgentRuntime`.
- Start/close runtime.
- Bridge core events into Hub events.
- Expose send/cancel/branch/fork helpers.

`events.py`

- SSE subscriber registry.
- Per-conversation fanout.
- Event id allocation.

`permissions.py`

- Permission callback implementation.
- Pending futures registry.
- Decision endpoint integration.

`services/conversations.py`

- Conversation CRUD.
- Send message orchestration.
- Cancel run.

`services/workers.py`

- Worker CRUD through core dynamic worker APIs.
- Queue operations.

`routes/*`

- Thin HTTP route layer.
- Validate input/output with Pydantic.

### 6.2 Backend app state

```python
class HubAppState:
    home_dir: Path
    config_path: Path
    db: HubDatabase
    runtime: AgentRuntime
    event_hub: HubEventHub
    permission_bridge: PermissionBridge
```

FastAPI routes access this via `app.state.hub`.

### 6.3 Hub DB

Hub DB stores UI concerns only.

Recommended path:

```text
~/.soong-agent/hub/hub.db
```

Tables:

```sql
CREATE TABLE IF NOT EXISTS conversations (
  conversation_id TEXT PRIMARY KEY,
  core_session_id TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL DEFAULT 'New conversation',
  status TEXT NOT NULL DEFAULT 'active',
  active_core_node_id TEXT,
  last_message_preview TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  message_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  parent_message_id TEXT,
  sender_type TEXT NOT NULL,
  sender_id TEXT,
  sender_name TEXT NOT NULL,
  target_type TEXT,
  target_id TEXT,
  original_text TEXT NOT NULL DEFAULT '',
  display_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'completed',
  core_session_id TEXT,
  core_run_id TEXT,
  core_node_id TEXT,
  child_run_id TEXT,
  task_id TEXT,
  worker_id TEXT,
  queue_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS permission_requests (
  permission_request_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  core_session_id TEXT NOT NULL,
  core_run_id TEXT,
  tool_name TEXT NOT NULL,
  permission TEXT NOT NULL,
  target_scope TEXT,
  args_summary TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending',
  decision TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL,
  worker_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

Why snapshots:

- Historical messages should still render worker name/description after soft delete or edits.
- Snapshots avoid rewriting old messages.

### 6.4 Message sender types

Allowed `sender_type`:

- `user`
- `orchestrator`
- `worker`
- `system`

Allowed `target_type`:

- `orchestrator`
- `worker`
- `none`

Allowed `message.status`:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

### 6.5 Conversation send flow

Request:

```http
POST /conversations/{conversation_id}/messages
```

```json
{
  "text": "@reviewer_worker inspect this diff"
}
```

Flow:

1. Load conversation.
2. Parse leading mention.
3. If mention is worker, call `runtime.resolve_worker_mention`.
4. Insert Hub user message with `status=completed`.
5. Emit `message_created`.
6. Start core run:

No mention:

```python
handle = await runtime.start(
    text,
    session_id=core_session_id,
    mode="orchestrator",
)
```

`@Orchestrator`:

```python
handle = await runtime.start(
    text_without_mention,
    session_id=core_session_id,
    mode="orchestrator",
)
```

`@worker`:

```python
handle = await runtime.start(
    text_without_mention,
    session_id=core_session_id,
    mode="orchestrator",
    directives={
        "mentioned_worker": resolved_worker.model_dump()
    },
)
```

7. Insert or update an Orchestrator placeholder message with `status=running`.
8. Store `core_run_id`.
9. Create background task to consume `handle.stream()`.
10. Convert core events into Hub events/messages.

Response:

```json
{
  "message_id": "msg_user_x",
  "conversation_id": "conv_x",
  "core_session_id": "sess_x",
  "core_run_id": "run_x",
  "status": "running"
}
```

### 6.6 Core event mapping

Hub should not forward every core event by default. It maps important events:

| Core event | Hub behavior |
| --- | --- |
| `run_started` | update Orchestrator message to running |
| `assistant_delta` / equivalent | append to Orchestrator message |
| `assistant_message` / equivalent | finalize Orchestrator message |
| `tool_call_started` | hidden by default, optional status |
| `tool_call_completed` | hidden by default |
| `permission_failed` | show system/error message |
| `worker_queued` | create/update worker status message |
| `worker_started` | show worker running status |
| `worker_completed` | create worker result message |
| `run_completed` | mark messages completed |
| `run_failed` | mark Orchestrator message failed |
| `run_cancelled` | mark messages cancelled |

Debug mode may forward raw core events under a separate event type:

```json
{
  "type": "debug_core_event",
  "payload": { "...": "..." }
}
```

Default UI ignores debug events.

### 6.7 SSE event hub

Route:

```http
GET /events?conversation_id=conv_x
```

Envelope:

```json
{
  "id": "evt_01",
  "type": "message_updated",
  "conversation_id": "conv_x",
  "payload": {},
  "created_at": "2026-06-17T12:00:00Z"
}
```

Important event types:

- `conversation_created`
- `conversation_updated`
- `message_created`
- `message_delta`
- `message_updated`
- `run_started`
- `run_completed`
- `run_failed`
- `run_cancelled`
- `worker_queued`
- `worker_started`
- `worker_completed`
- `worker_failed`
- `worker_cancelled`
- `permission_requested`
- `permission_resolved`
- `health_changed`

Implementation:

```python
class HubEventHub:
    def subscribe(self, conversation_id: str) -> AsyncIterator[HubEvent]: ...
    async def publish(self, event: HubEvent) -> None: ...
```

Use per-conversation `asyncio.Queue`.

Backpressure:

- Each subscriber queue has a bounded size.
- If full, drop debug/status-only events first or disconnect subscriber.
- Do not drop permission events.

### 6.8 Permission bridge

Core permission callback:

```python
async def permission_callback(request: PermissionRequest) -> PermissionDecision:
    return await permission_bridge.request_decision(conversation_id, request)
```

Challenge:

- Core callback only receives `PermissionRequest`.
- Hub needs conversation_id.

Solution:

- Runtime bridge stores mapping from active `core_run_id/session_id` to conversation_id.
- Permission callback resolves conversation by session/run context if request exposes enough metadata.
- If request lacks run metadata, core should include session/run in `PermissionRequest` metadata or callback context.

Flow:

1. Core calls callback.
2. Backend creates `permission_request_id`.
3. Backend stores pending future.
4. Backend inserts `permission_requests` row.
5. Backend publishes SSE:

```json
{
  "type": "permission_requested",
  "conversation_id": "conv_x",
  "payload": {
    "permission_request_id": "perm_x",
    "tool_name": "code.run_command",
    "permission": "write",
    "target_scope": "/Users/bytedance/proj/soong-agent",
    "args_summary": "ls -la",
    "suggested_decision": "deny"
  }
}
```

6. Frontend posts decision:

```http
POST /permissions/perm_x/decision
```

```json
{
  "decision": "allow_once"
}
```

7. Backend resolves future.
8. Core continues.

No timeout:

- The future waits until user decides or backend shuts down.
- If run is cancelled, backend resolves pending permission as deny/cancelled.

## 7. Backend API 详细设计

### 7.1 Health

```http
GET /health
```

Response:

```json
{
  "ok": true,
  "status": "ready",
  "config_path": "/Users/bytedance/.soong-agent/config.toml",
  "provider": "openai",
  "model": "qwen2.5:7b",
  "base_url": "http://127.0.0.1:11434/v1",
  "core_started": true,
  "hub_db_path": "/Users/bytedance/.soong-agent/hub/hub.db",
  "warnings": []
}
```

If backend is alive but core failed:

```json
{
  "ok": false,
  "status": "core_failed",
  "error": {
    "code": "config_error",
    "message": "orchestrator mode requires at least one configured worker pool"
  }
}
```

### 7.2 Conversations

Create:

```http
POST /conversations
```

```json
{
  "title": "New conversation"
}
```

Response:

```json
{
  "conversation_id": "conv_x",
  "core_session_id": "sess_x",
  "title": "New conversation",
  "status": "active",
  "created_at": "...",
  "updated_at": "..."
}
```

List:

```http
GET /conversations
```

Response:

```json
{
  "conversations": [
    {
      "conversation_id": "conv_x",
      "title": "New conversation",
      "status": "active",
      "last_message_preview": "inspect this diff",
      "updated_at": "..."
    }
  ]
}
```

Messages:

```http
GET /conversations/{conversation_id}/messages?limit=100&before=...
```

Response:

```json
{
  "messages": [
    {
      "message_id": "msg_x",
      "conversation_id": "conv_x",
      "sender_type": "user",
      "sender_id": "user",
      "sender_name": "You",
      "target_type": "worker",
      "target_id": "reviewer_worker",
      "display_text": "inspect this diff",
      "status": "completed",
      "core_node_id": "node_x",
      "created_at": "..."
    }
  ]
}
```

Send:

```http
POST /conversations/{conversation_id}/messages
```

```json
{
  "text": "@reviewer_worker inspect this diff"
}
```

Cancel:

```http
POST /conversations/{conversation_id}/cancel
```

```json
{
  "core_run_id": "run_x",
  "queue_id": null
}
```

Branch node list:

```http
GET /conversations/{conversation_id}/branchable-nodes
```

Response:

```json
{
  "nodes": [
    {
      "core_node_id": "node_x",
      "message_id": "msg_x",
      "preview": "inspect this diff",
      "created_at": "..."
    }
  ]
}
```

Branch:

```http
POST /conversations/{conversation_id}/branch
```

```json
{
  "core_node_id": "node_x"
}
```

Fork:

```http
POST /conversations/{conversation_id}/fork
```

```json
{
  "core_node_id": "node_x",
  "title": "Fork from inspect this diff"
}
```

Response:

```json
{
  "conversation_id": "conv_new",
  "core_session_id": "sess_new"
}
```

### 7.3 Workers

List:

```http
GET /workers?include_disabled=true&include_deleted=false
```

Response:

```json
{
  "workers": [
    {
      "worker_id": "reviewer_worker",
      "worker_agent_id": "agent_worker_x",
      "worker_pool_id": "default",
      "agent_definition_id": "code_reviewer",
      "name": "Code Reviewer",
      "description": "Reviews code changes.",
      "status": "idle",
      "enabled": true,
      "deleted_at": null,
      "queue_length": 0,
      "allowed_tools": ["code.read_file", "code.search"],
      "model": {
        "provider": "openai",
        "name": "qwen2.5:7b",
        "base_url": "http://127.0.0.1:11434/v1"
      }
    }
  ]
}
```

Create:

```http
POST /workers
```

```json
{
  "worker_id": "reviewer_worker",
  "worker_pool_id": "default",
  "name": "Code Reviewer",
  "description": "Reviews code changes.",
  "system_prompt": "You are a senior code reviewer.",
  "model": {
    "provider": "openai",
    "name": "qwen2.5:7b",
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "temperature": 0.2
  },
  "allowed_tools": ["code.read_file", "code.search"],
  "enabled": true
}
```

Update:

```http
PATCH /workers/{worker_id}
```

Delete:

```http
DELETE /workers/{worker_id}
```

Delete response:

```json
{
  "worker_id": "reviewer_worker",
  "enabled": false,
  "deleted_at": "..."
}
```

Queue:

```http
GET /workers/{worker_id}/queue
```

Cancel queued:

```http
POST /workers/{worker_id}/queue/{queue_id}/cancel
```

### 7.4 Tools

```http
GET /tools
```

Response:

```json
{
  "tools": [
    {
      "name": "code.read_file",
      "description": "Read a text file from the workspace.",
      "permission": "readonly",
      "tags": ["code", "readonly"],
      "enabled": true
    }
  ]
}
```

### 7.5 Permissions

```http
POST /permissions/{permission_request_id}/decision
```

Request:

```json
{
  "decision": "allow_once"
}
```

Allowed decisions:

- `allow_once`
- `allow_for_session`
- `deny`

Response:

```json
{
  "permission_request_id": "perm_x",
  "status": "allowed",
  "decision": "allow_once"
}
```

## 8. Frontend 设计

### 8.1 Electron main

Files:

```text
frontend/electron/main.ts
frontend/electron/backendProcess.ts
frontend/electron/preload.ts
```

Responsibilities:

- Start backend process in dev mode.
- Detect backend port.
- Wait for health.
- Create BrowserWindow.
- Provide backend base URL to renderer.
- Stop backend on quit.
- Show backend startup error screen when necessary.

The renderer should not use Node APIs directly.

### 8.2 React app layout

```text
+----------------------+----------------------------------+----------------------+
| ConversationList      | MessageStream                    | WorkerPanel          |
|                      |                                  |                      |
| New conversation      | user                             | worker list          |
| conversation rows     | Orchestrator                     | queue/status         |
| active indicator      | worker result                    | editor               |
|                      | permission card                  | tools selector       |
|                      |                                  |                      |
|                      | MentionInput                     |                      |
+----------------------+----------------------------------+----------------------+
```

Components:

```text
components/layout/AppShell.tsx
components/conversations/ConversationList.tsx
components/conversations/ConversationRow.tsx
components/messages/MessageStream.tsx
components/messages/MessageBubble.tsx
components/messages/MessageStatus.tsx
components/messages/BranchForkMenu.tsx
components/input/MentionInput.tsx
components/input/MentionMenu.tsx
components/workers/WorkerPanel.tsx
components/workers/WorkerRow.tsx
components/workers/WorkerEditor.tsx
components/workers/ToolMultiSelect.tsx
components/permissions/PermissionCard.tsx
components/permissions/PermissionStack.tsx
components/layout/HealthBanner.tsx
```

### 8.3 State management

Use a small store, for example Zustand or React reducer. V1 can start with reducer/context to avoid unnecessary dependencies.

State shape:

```ts
type AppState = {
  health: HealthStatus | null
  conversations: Conversation[]
  activeConversationId: string | null
  messagesByConversation: Record<string, Message[]>
  workers: WorkerView[]
  pendingPermissions: Record<string, PermissionRequestView>
  eventConnection: {
    status: "connecting" | "open" | "closed" | "error"
    lastEventId?: string
  }
}
```

Event reducer handles:

- `conversation_created`
- `conversation_updated`
- `message_created`
- `message_delta`
- `message_updated`
- `worker_queued`
- `worker_started`
- `worker_completed`
- `worker_failed`
- `permission_requested`
- `permission_resolved`

### 8.4 Mention input

Behavior:

- User types `@`.
- Menu opens with:
  - `Orchestrator`
  - enabled workers
- Up/down moves selected item.
- Enter/Tab selects item.
- Selection inserts `@worker_id `.
- Submit sends full raw text to backend.

Parsing remains authoritative in backend/core. Frontend suggestions are a UX helper only.

### 8.5 Sending while run is active

User requirement: waiting for a response must not block typing.

Frontend behavior:

- Input remains enabled while a run is active.
- Send button remains enabled unless backend says conversation is locked.
- Messages sent to the same conversation can be accepted and marked `queued` if core session is active.
- Backend/core session queue controls execution order.

UI should show:

- user message completed
- Orchestrator placeholder queued/running
- queued indicator if core returns queued handle

### 8.6 Branch/fork UI

Each user-visible user message can expose actions:

- Branch from here
- Fork as new conversation

Additionally, a command/menu can list branchable nodes:

- only user messages
- show node id short form
- show message preview
- allow keyboard navigation

Branch action:

- Calls backend branch API.
- Updates active conversation messages to reflect active path if backend returns refreshed messages.

Fork action:

- Calls backend fork API.
- Adds new conversation to list.
- Switches active conversation.

### 8.7 Worker editor

Fields:

- worker id
- name
- description
- system prompt
- worker pool id
- provider
- model name
- base URL
- api key env or api key
- temperature
- max output tokens
- allowed tools
- enabled

Validation:

- worker id required.
- worker id safe characters only.
- name required.
- model name required if overriding model.
- allowed tools must come from `/tools`.

### 8.8 Permission card

Permission card shows:

- tool name
- permission level
- target scope
- args summary
- reason if available
- actions:
  - Allow once
  - Allow for session
  - Deny

The card is inline, not a modal.

If multiple permissions are pending:

- Show a stack/list.
- Each decision applies to one permission request.

### 8.9 Visual style

Operational desktop app:

- Compact three-column layout.
- Clear sender colors:
  - user
  - Orchestrator
  - worker
  - system/status
- Status chips should be short and scannable.
- No landing page.
- No decorative hero.
- No nested card-heavy layout.
- Tool controls use familiar UI elements:
  - icons for actions
  - toggles for enable/disable
  - multiselect for tools
  - segmented controls where useful

## 9. Data flow

### 9.1 Normal message

```text
User input
  -> React POST /conversations/{id}/messages
  -> Backend parse mention: none
  -> Hub DB insert user message
  -> runtime.start(mode="orchestrator")
  -> Backend consumes RunHandle stream
  -> Core emits events
  -> Backend maps events to Hub messages
  -> SSE emits message deltas/status
  -> React updates MessageStream
```

### 9.2 Mentioned worker message

```text
User input "@reviewer inspect diff"
  -> Backend parse leading mention
  -> Core resolve_worker_mention("reviewer")
  -> runtime.start(mode="orchestrator", directives.mentioned_worker=reviewer)
  -> Orchestrator receives directive context
  -> Orchestrator calls agent.dispatch_worker
  -> Core validates dispatch target
  -> Worker runs or queues
  -> Worker result returns to Orchestrator
  -> Orchestrator summarizes
  -> Hub shows worker result + Orchestrator summary
```

### 9.3 Permission request

```text
Core tool call requires permission
  -> core permission_callback
  -> Backend PermissionBridge creates pending future
  -> Hub DB permission_requests insert
  -> SSE permission_requested
  -> React PermissionCard
  -> User chooses decision
  -> POST /permissions/{id}/decision
  -> Backend resolves future
  -> Core continues tool execution
  -> SSE permission_resolved
```

### 9.4 Worker queue

```text
Orchestrator dispatches mentioned worker
  -> Worker is busy
  -> Core enqueues WorkerQueueItem
  -> Hub receives worker_queued
  -> UI shows queue
  -> Current worker run completes
  -> Core dequeues next item
  -> Hub receives worker_started
  -> Worker completes
  -> Hub receives worker_completed
```

## 10. Error handling

### 10.1 Error envelope

Backend API errors:

```json
{
  "error": {
    "code": "worker_not_found",
    "message": "Worker not found: reviewer",
    "details": {}
  }
}
```

### 10.2 Common error codes

- `config_missing`
- `config_bootstrap_failed`
- `config_invalid`
- `core_start_failed`
- `conversation_not_found`
- `message_not_found`
- `worker_not_found`
- `worker_ambiguous`
- `worker_disabled`
- `worker_deleted`
- `worker_queue_full`
- `invalid_mention`
- `permission_not_found`
- `permission_already_resolved`
- `run_not_found`
- `run_cancelled`
- `provider_unavailable`
- `model_tool_calling_unsupported`
- `internal_error`

### 10.3 UI behavior

- Startup failure: full-window error.
- Runtime API failure: inline banner or system message.
- Send failure before core run starts: mark placeholder failed.
- Core run failure: mark Orchestrator message failed.
- Worker failure: mark worker message failed and let Orchestrator handle summary if possible.
- SSE disconnected: show connection banner and retry.

## 11. 安全与权限

### 11.1 Secrets

Rules:

- Prefer `api_key_env`.
- Direct `api_key` support must redact in:
  - logs
  - Hub DB snapshots where possible
  - SSE payloads
  - frontend display
- Health endpoint never returns secret.
- Worker list endpoint returns only redacted model info.

### 11.2 Tool permissions

Hub UI does not grant permission by itself. It only configures worker `allowed_tools`.

Actual permission decision remains core policy:

- readonly/write tool definition
- config permissions
- allowed write roots
- hooks
- permission callback
- session permission cache

### 11.3 Path and command safety

Hub must not add a backend endpoint that runs arbitrary shell commands outside core tools.

All file/command operations go through core tools so existing permission/hook/path policies apply.

## 12. Testing strategy

### 12.1 Core unit tests

Add tests for:

- JSON agent definition parsing.
- JSON worker parsing.
- dynamic worker SQLite CRUD.
- precedence: SQLite > JSON > TOML > built-in.
- `resolve_worker_mention` exact id.
- `resolve_worker_mention` unique name.
- ambiguous worker name error.
- disabled/deleted worker error.
- `runtime.start(..., directives=...)` stores directives.
- directive context appears in orchestrator context.
- `agent.dispatch_worker` fills mentioned worker when no worker id passed.
- `agent.dispatch_worker` rejects non-mentioned worker.
- busy mentioned worker enqueues.
- queue limit 20.
- queued job cancellation.
- worker completion starts next queued job.

### 12.2 Backend unit tests

Add tests for:

- config bootstrap only when missing.
- health response.
- Hub DB migrations.
- conversation CRUD.
- message CRUD.
- mention parser.
- send flow with fake/runtime stub.
- worker CRUD route calls core APIs.
- tools route reads core registry.
- SSE subscribe/publish.
- permission bridge waits and resolves.
- permission bridge cancellation on shutdown.
- branch/fork route calls core APIs.

### 12.3 Frontend tests

Add tests for:

- ConversationList rendering.
- MessageStream rendering by sender type.
- event reducer.
- MentionInput suggestions and keyboard selection.
- WorkerPanel status rendering.
- WorkerEditor validation.
- PermissionCard allow/deny actions.
- SSE reconnect state.

### 12.4 Integration tests

Python integration:

- Start FastAPI app with test client.
- Use temp `SOONG_AGENT_HOME`.
- Use fake provider for deterministic responses.
- Validate conversation send, events, worker CRUD.

Local Ollama E2E:

- Gate behind env var such as `SOONG_AGENT_REQUIRE_OLLAMA_E2E=1`.
- Use configured `~/.soong-agent/config.toml`.
- Do not hard-code model name in test unless explicitly creating isolated config.

Minimum Ollama scenario:

1. Backend health ready.
2. Create conversation.
3. Create worker.
4. Send normal message.
5. Send `@worker` message.
6. Observe worker dispatch.
7. Observe final Orchestrator summary.
8. Trigger permission request with a write/command tool if model/tool reliability allows.
9. Branch from user message.
10. Fork conversation.

### 12.5 Manual validation

Commands:

```bash
PYTHONPATH=src python3 -m agent_hub.backend
```

```bash
cd src/agent_hub/frontend
npm install
npm run dev
```

Future helper:

```bash
./soong-hub
```

## 13. Implementation phases

### Phase 1: Core worker foundations

Deliverables:

- Dynamic worker/agent config models.
- JSON loading from `~/.soong-agent/agents/*.json` and `~/.soong-agent/workers/*.json`.
- SQLite dynamic worker persistence.
- Worker config precedence.
- Runtime worker CRUD APIs.
- Worker mention resolution.
- Runtime reload of worker registry/pool.

Exit criteria:

- Tests can create/update/disable/delete workers through core.
- `agent.list_workers` sees dynamic workers.
- JSON workers load without Hub.

### Phase 2: Core directives and queue

Deliverables:

- `runtime.start(..., directives=...)`.
- `RunDirectives` model.
- Directive injection into Orchestrator context.
- Dispatch target validation.
- In-memory per-worker queue.
- Queue cancellation and status events.

Exit criteria:

- `@worker` semantics can be tested at core level without Hub.
- Dispatch to wrong worker fails.
- Busy worker queues up to 20.

### Phase 3: Hub backend foundation

Deliverables:

- FastAPI app.
- Config bootstrap.
- Hub DB migrations.
- Health route.
- Conversation/message routes.
- Worker routes.
- Tools route.
- SSE event hub.
- Permission bridge.

Exit criteria:

- Backend can run standalone.
- Tests cover API and SSE without Electron.

### Phase 4: Electron shell

Deliverables:

- Electron/Vite/React TypeScript project.
- Main process backend spawning.
- Health wait loop.
- Renderer preload config.
- Startup error screen.
- `./soong-hub` helper script.

Exit criteria:

- A developer can start the desktop app from repo root.
- Backend process exits when app quits.

### Phase 5: React UI

Deliverables:

- Three-column layout.
- Conversation list.
- Message stream.
- Mention input.
- Worker panel/editor.
- Permission cards.
- Branch/fork controls.
- SSE reducer.

Exit criteria:

- User can complete normal and `@worker` conversation flows.
- User can manage workers.
- User can handle permission requests.

### Phase 6: Integration and polish

Deliverables:

- Local Ollama validation.
- Error handling pass.
- UX pass for loading/queued/running states.
- README startup instructions.
- Test coverage for main paths.

Exit criteria:

- V1 acceptance criteria in `requirement.md` pass.

## 14. Compatibility with existing core

Known existing pieces:

- `AgentRuntime.start(message, session_id, mode)`
- `AgentRuntime.run_worker_agent(...)`
- `WorkerPoolRuntime`
- `agent.list_workers`
- `agent.dispatch_worker`
- `PermissionCallback`
- `PermissionSessionCache`
- core session SQLite store
- agent definition registry
- task service and task tools

Design impact:

- Most Hub behavior can be added without changing provider adapters.
- `runtime.start` signature needs a backward-compatible optional argument.
- `ToolExecutionContext` needs access to run directives.
- Worker runtime needs queue and dynamic config support.
- Agent definition registry needs JSON/dynamic overlays.
- Model resolver needs inline model config support or dynamic profile synthesis.

## 15. Risks and mitigations

### 15.1 Tool-calling reliability

Risk:

- Local Ollama model may not reliably call tools or follow Orchestrator directive.

Mitigation:

- Enforce dispatch target in core tools.
- Keep prompts explicit.
- Use config-driven model choice.
- E2E should report model/tool limitations clearly.

### 15.2 Core/Hub persistence split

Risk:

- Data split between core SQLite and Hub DB can drift.

Mitigation:

- Core remains source of truth for execution.
- Hub stores only display records.
- Store core ids on Hub messages.
- Provide replay fallback for debugging.

### 15.3 Dynamic worker reload

Risk:

- Updating worker config while a worker is running can create inconsistent runtime state.

Mitigation:

- Running worker keeps snapshot until run completion.
- New runs use new config.
- Soft delete prevents new dispatch but does not cancel active run unless user explicitly cancels.

### 15.4 Permission mapping

Risk:

- Permission callback may not know which conversation should receive the request.

Mitigation:

- Add run/session metadata to permission request path.
- Maintain backend run-to-conversation map.
- If mapping fails, publish global permission event and show it in active app-level stack.

### 15.5 Electron process cleanup

Risk:

- Python backend process may leak after Electron closes.

Mitigation:

- Main process tracks child PID.
- Graceful shutdown first.
- Kill after timeout.
- Dev script documents cleanup.

### 15.6 Queue is memory-only

Risk:

- Queued requests disappear on backend restart.

Mitigation:

- This is explicit V1 behavior.
- UI should mark queued jobs lost only if backend restarts while app is open.
- Durable queue remains future work.

## 16. Implementation notes

### 16.1 Do not implement direct worker chat

Avoid this shortcut:

```text
@worker -> runtime.run_worker_agent(...)
```

This breaks the product semantics. Worker dispatch must go through Orchestrator.

### 16.2 Do not duplicate tool permission logic in Hub

Hub may display and configure `allowed_tools`, but final enforcement belongs to core.

### 16.3 Keep route handlers thin

Routes should validate HTTP input and call services. Business logic belongs in service modules.

### 16.4 Use structured APIs, not string parsing

For:

- JSON configs
- model config
- worker configs
- SSE payloads
- permission decisions

Use Pydantic/typed models. Avoid ad hoc dicts once interfaces stabilize.

### 16.5 Avoid frontend hardcoded worker/tool truth

Frontend can hardcode layout and labels, but not the available workers/tools/model. Those come from backend/core.

## 17. Final V1 contract

When implementation is complete, the following contract should hold:

1. Hub starts locally as an Electron app.
2. Backend starts automatically and initializes core.
3. User can chat with Orchestrator.
4. User can `@worker`, and Orchestrator is forced to dispatch that worker.
5. Dynamic workers can be created and used without restart.
6. Worker queue handles busy mentioned workers.
7. Permissions are shown inline and wait forever for user decision.
8. Conversations are persisted in Hub DB and linked to core sessions.
9. Branch changes active path within the same session.
10. Fork creates a new session/conversation from a user message node.
11. Provider/model configuration comes from `~/.soong-agent/config.toml`.
12. Local Ollama can be used through config for real validation.
