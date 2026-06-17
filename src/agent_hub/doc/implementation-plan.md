# Agent Hub 完整 V1 决策版实施计划

## Summary

实现 `src/agent_hub` 作为 `agent_core` 的本地桌面协作层：Electron + React + TypeScript 前端，FastAPI + uvicorn 后端，默认由 Orchestrator 处理用户消息，`@worker` 只作为 Orchestrator 派工约束，不允许绕过 Orchestrator 直连 worker。

完整 V1 包括：core worker 动态配置、`@worker` directive、worker 内存队列、Hub backend、SSE、权限内联等待、Electron 启动后端、React 三栏 UI、会话历史、branch/fork，以及 `design.md` 中补充详细测试用例。

## Key Decisions

- 完整 V1：core、backend、Electron/React UI 都做。
- `@worker` busy 时进入队列，Orchestrator run 等待 queued worker 完成后继续总结。
- Worker 配置采用统一内联 schema，API 和 JSON 文件共用。
- Worker model 可选全量覆盖；不填则继承主模型配置。
- Hub 不自动导入已有 core sessions。
- Hub `project_dir` 使用启动 `./agenthub` 时的 cwd。
- 新建 conversation 立即创建 core session；删除/隐藏 Hub conversation 不删除 core session。
- Conversation title 用首条用户消息截断生成。
- 前端状态管理用 React reducer/context。
- 前端测试用 Vitest + React Testing Library + jsdom。
- Hub DB 使用 `aiosqlite`。

## Core Changes

- 为 `AgentRuntime.start()` 增加向后兼容参数：

```python
directives: RunDirectives | dict[str, Any] | None = None
```

- 新增类型：
  - `RunDirectives`
  - `MentionedWorkerDirective`
  - `WorkerConfigCreate`
  - `WorkerConfigUpdate`
  - `WorkerConfigView`
  - `WorkerMentionResolution`
  - `WorkerQueueItem`
  - `BranchableNodeView`

- `mentioned_worker` directive 必须：
  - 写入 run metadata。
  - 注入 Orchestrator context。
  - 传入 `ToolExecutionContext` 或 runtime service，供 `agent.dispatch_worker` 强校验。

- `agent.dispatch_worker` 语义：
  - 如果当前 run 有 `mentioned_worker`，未指定 worker 时自动填入该 worker。
  - 指定其他 worker 时返回 tool error。
  - worker idle 时直接运行。
  - worker busy 时创建 queue item，并让当前 `agent.dispatch_worker` 调用等待该 item 真正执行完成，再返回 worker result 给 Orchestrator。
  - queue item 被取消时，等待中的 dispatch 返回 cancelled/failed tool result。
  - 队列上限固定 20，超过返回 `worker_queue_full`。

- Dynamic worker config：
  - 支持 `~/.soong-agent/workers/*.json`。
  - 支持 `~/.soong-agent/agents/*.json`。
  - Hub UI 创建的 worker 持久化到 core SQLite。
  - 优先级：`dynamic SQLite > user JSON > config.toml > built-in`。
  - `DELETE /workers/{id}` 做软删除：`enabled=false` + `deleted_at`。
  - running worker 使用启动时快照；更新/删除只影响后续 run。

- 统一 WorkerConfig schema：
  - `worker_id`
  - `worker_pool_id`
  - `name`
  - `description`
  - `system_prompt`
  - `model`
  - `allowed_tools`
  - `enabled`
  - `metadata`
  - `agent_definition_id` 可由 core 自动生成；如显式提供则使用。
  - inline `model` 可覆盖 `provider/name/base_url/api_key_env/temperature/max_output_tokens`；不填继承主模型配置。

- Branch/fork：
  - `list_branchable_nodes()` 默认只返回 user message nodes。
  - `switch_node()` 用于同 session branch；session 有 active/queued run 时拒绝。
  - `fork_session()` 用于从 user node 创建新 core session。

## Hub Backend / UI

- Backend：
  - 新建 `agent_hub.backend` FastAPI app。
  - 启动命令：`PYTHONPATH=src python3 -m agent_hub.backend`。
  - `./agenthub` 从当前 cwd 启动，cwd 即 Hub V1 唯一 `project_dir`。
  - 缺失 `~/.soong-agent/config.toml` 时用 core 默认模板创建；已有不覆盖；非法配置启动失败。
  - Hub DB：`~/.soong-agent/hub/hub.db`，使用 `aiosqlite`。
  - Hub 只管理自己创建的 conversations，不扫描/导入已有 core sessions。

- Hub DB 保存：
  - conversations
  - messages
  - permission_requests
  - worker_snapshots

- Backend API：
  - `/health`
  - `/events?conversation_id=...`
  - conversation create/list/messages/send/cancel
  - branchable-nodes / branch / fork
  - worker list/create/update/delete/queue/cancel
  - tools list
  - permission decision

- Conversation lifecycle：
  - 新建 conversation 时立即创建 core session。
  - 首条用户消息生成标题：去掉 mention 后截断。
  - Hub 删除/隐藏 conversation 不删除 core session。
  - 已有 core sessions 不自动导入。

- Permission：
  - core permission callback 进入 backend PermissionBridge。
  - SSE 发 inline `permission_requested`。
  - 用户不决策就一直等待。
  - run cancel/backend shutdown 时 pending permission resolve 为 deny/cancelled。

- Frontend：
  - Electron + Vite + React + TypeScript。
  - 状态管理用 reducer/context。
  - 测试用 Vitest + React Testing Library + jsdom。
  - 三栏布局：conversation list / message stream / worker panel。
  - Mention input 支持 `@Orchestrator` 和 enabled workers，上下选择。
  - active run 时输入框保持可用，后续消息显示 queued/running。
  - Worker editor 同时提供表单模式和 JSON 原文模式，二者使用同一 WorkerConfig schema。
  - Permission card 内联展示，不用弹窗。
  - Branch/fork 菜单只列用户消息 node id 和摘要。

## Implementation Order

1. Core worker foundations
   - 动态 agent/worker models。
   - JSON 加载。
   - SQLite migration。
   - worker CRUD API。
   - worker config reload。
   - `agent.list_workers` 展示动态 worker。

2. Core directives and dispatch enforcement
   - `RunDirectives`。
   - `runtime.start(..., directives=...)`。
   - directive context injection。
   - `resolve_worker_mention()`。
   - `agent.dispatch_worker` mentioned-worker 校验。

3. Core worker queue and branch/fork helpers
   - per-worker queue。
   - queue cancel/list/status events。
   - worker complete 后启动下一项。
   - branchable user-node API。

4. Hub backend foundation
   - FastAPI app。
   - config bootstrap。
   - Hub DB migrations。
   - health/conversation/message routes。
   - runtime bridge。
   - SSE event hub。
   - permission bridge。

5. Hub worker/backend APIs
   - worker CRUD routes。
   - queue routes。
   - tools route。
   - branch/fork routes。
   - core event 到 Hub message/status 映射。

6. Electron shell
   - frontend package scaffolding。
   - Electron main/preload。
   - backend process spawn。
   - health wait。
   - startup error page。
   - `./agenthub` helper script。

7. React UI
   - 三栏 layout。
   - conversation list。
   - message stream。
   - mention input。
   - worker panel/editor。
   - permission cards。
   - branch/fork menu。
   - SSE reducer。

8. Integration and polish
   - 补 README 启动说明。
   - 跑 Python unit/integration。
   - 跑 frontend tests。
   - 本地 Ollama gated E2E。
   - 修正 loading/queued/running/error 体验。

## Test Plan

### Core Test Cases

| ID | Scenario | Expected |
| --- | --- | --- |
| C1 | 解析 `~/.soong-agent/agents/*.json` | 生成有效 AgentDefinition |
| C2 | 解析 `~/.soong-agent/workers/*.json` | 生成有效 WorkerConfig |
| C3 | SQLite dynamic worker 覆盖 JSON/TOML | effective worker 来自 SQLite |
| C4 | soft delete worker | 不能 mention，不能 dispatch，历史可显示 |
| C5 | `resolve_worker_mention` exact id | 返回唯一 worker |
| C6 | `resolve_worker_mention` unique name | 返回唯一 worker |
| C7 | ambiguous worker name | 返回 `worker_ambiguous` |
| C8 | disabled/deleted worker mention | 返回对应错误 |
| C9 | `runtime.start(..., directives=...)` | directives 写入 run metadata |
| C10 | directive context build | Orchestrator context 含 mentioned worker 约束 |
| C11 | dispatch 未指定 worker | 自动填入 mentioned worker |
| C12 | dispatch 到其他 worker | 返回 tool error |
| C13 | busy mentioned worker | 进入 queue |
| C14 | queue 第 21 个 item | 返回 `worker_queue_full` |
| C15 | cancel queued item | item 不会启动 |
| C16 | worker 完成 | 自动启动下一 queued item |
| C17 | branchable nodes | 只返回 user message nodes |
| C18 | active/queued run 时 branch | 返回 session active error |
| C19 | fork from user node | 创建新 session |

### Backend Test Cases

| ID | Scenario | Expected |
| --- | --- | --- |
| B1 | config 缺失 | 创建默认 config |
| B2 | config 已存在 | 不覆盖 |
| B3 | config 非法 | `/health` 返回 core/config error |
| B4 | create/list conversation | 返回 Hub conversation |
| B5 | list messages | 返回 conversation messages |
| B6 | send normal message | 启动 orchestrator run |
| B7 | send `@worker ...` | resolve mention 并传 directives |
| B8 | active run 时继续 send | 后续 run/message queued |
| B9 | cancel active run | run/message 标记 cancelled |
| B10 | cancel queued run | queued item 不启动 |
| B11 | worker CRUD routes | 调用 core worker API |
| B12 | worker queue routes | 返回 queue/cancel 状态 |
| B13 | `/tools` | 返回 effective tool catalog |
| B14 | SSE subscribe/publish | 前端可收到事件 |
| B15 | core event mapping | 生成 Hub message delta/status |
| B16 | permission bridge wait | 未决策前 core 等待 |
| B17 | permission decision | future resolve，core 继续 |
| B18 | shutdown pending permission | pending request deny/cancelled |
| B19 | branch route | 调 core switch active node |
| B20 | fork route | 创建新 conversation/session |

### Frontend Test Cases

| ID | Scenario | Expected |
| --- | --- | --- |
| F1 | app layout | 渲染三栏结构 |
| F2 | conversation switch | active conversation 更新 |
| F3 | message sender types | user/orchestrator/worker/system 样式不同 |
| F4 | `@` mention menu | 展示 Orchestrator 和 enabled workers |
| F5 | mention keyboard selection | 上下选择，Enter/Tab 插入 |
| F6 | active run 输入 | input 保持可用 |
| F7 | worker editor validation | required/safe id/model/tool 校验 |
| F8 | worker enable/disable/delete | 调正确 API 并更新 UI |
| F9 | permission card allow once | POST `allow_once` |
| F10 | permission card allow session | POST `allow_for_session` |
| F11 | permission card deny | POST `deny` |
| F12 | branch/fork menu | 只展示用户消息 node |
| F13 | SSE disconnected | 显示 reconnect/connection banner |

### Integration / E2E

| ID | Scenario | Expected |
| --- | --- | --- |
| I1 | FastAPI app + temp home | `/health` ready |
| I2 | create conversation -> send message | SSE 收到 completed/failed 终态 |
| I3 | create worker -> send `@worker` | Orchestrator dispatch 指定 worker |
| I4 | permission request round trip | run 等待，决策后继续 |
| I5 | branch from user message | 后续对话从该 node 延伸 |
| I6 | fork from user message | 新 conversation 绑定新 core session |
| E1 | Ollama gated normal flow | 使用 config provider/model，不硬编码模型 |
| E2 | Ollama gated `@worker` flow | worker dispatch 可观察 |
| E3 | Ollama gated permission flow | inline permission 可决策 |

## Assumptions

- V1 不做 installer/auto-update、多用户、云端、workspace switcher。
- V1 不做 web search、LSP、codesearch 新工具。
- Worker queue 是内存队列，backend/core 重启后丢失。
- `@worker` 只解析消息开头 mention。
- Direct `api_key` 如支持必须脱敏；推荐使用 `api_key_env`。
- Hub DB 是 UI 辅助状态；core 仍是 session tree、run、task、worker 语义的 source of truth。
