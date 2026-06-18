# Agent Hub 需求规格

## 1. 背景

`agent_core` 已经具备会话、上下文树、Orchestrator、Task DAG、worker pool、工具调用、权限回调、provider 配置等核心能力。当前缺少的是一个面向用户的桌面交互层，让用户可以像在 IM 群聊里一样和多个 agent 协作。

`agent_hub` 的目标不是重新实现 agent runtime，而是在 `agent_core` 之上提供一个本地桌面应用：

- 用户在一个会话里连续输入消息。
- 默认由 Orchestrator 处理请求。
- 用户可以通过 `@worker` 指定某个 worker，但请求仍由 Orchestrator 编排。
- 用户可以查看 Orchestrator、worker、状态事件、权限请求和历史会话。
- 用户可以管理 worker，并把 worker 配置持久化。
- 用户可以从历史消息分支或 fork 出新会话。

第一版优先服务本地开发体验，不做云端协作、账号体系和安装包发布。

## 2. 产品目标

Agent Hub 是一个本地桌面 IM 风格的 agent 协作工具。它应该让用户感觉自己在和一个由 Orchestrator 负责调度的 agent 团队协作，而不是在和单个普通 chatbot 对话。

核心目标：

1. 让 `agent_core` 的 orchestrator/worker 能力可视化、可操作。
2. 让用户能通过自然的 `@worker` 入口约束 Orchestrator 的派工对象。
3. 让 worker 配置可以通过 UI 创建、修改、禁用、软删除，也允许高级用户直接写 JSON。
4. 让权限请求以内联卡片的方式停住等待用户决策。
5. 让会话历史、分支、fork 能够以产品化方式被使用。
6. 用已有 `~/.soong-agent/config.toml` 控制 provider/model，不在 Hub 内硬编码模型。

## 3. 范围

### 3.1 V1 范围内

- Electron + React + TypeScript 桌面 UI。
- Python FastAPI + uvicorn 后端。
- Electron 启动时自动拉起 Python 后端。
- 单用户、本地运行。
- 使用现有 `agent_core` 作为执行层。
- 一个 Hub conversation 对应一个 `agent_core` session。
- 支持 SSE 实时事件。
- 支持普通消息、`@Orchestrator`、`@worker`。
- `@worker` 表示让 Orchestrator 指派指定 worker，不绕过 Orchestrator。
- 支持 worker 管理：
  - list
  - create
  - update
  - enable
  - disable
  - soft delete
- 支持 worker busy 时排队：
  - 内存 FIFO 队列
  - 队列可取消
  - 队列长度写死为 20
  - 重启后队列丢失
- 支持用户写 JSON 配置 worker/agent。
- 支持 Hub UI 创建的 worker 持久化到 core SQLite。
- 支持权限请求：
  - 后端等待
  - 前端内联卡片展示
  - 用户 allow/deny 后继续
  - 不设自动超时
- 支持基础会话功能：
  - 新建会话
  - 列出会话
  - 切换会话
  - 查看历史消息
  - 取消运行中请求或排队请求
- 支持上下文树相关操作：
  - 从当前 session 内某个用户消息 node 切换 active path，形成新的分支路径
  - 从当前 session 的某个用户消息 node fork 出一个新 session
- 支持本地 Ollama 真实验证，但模型通过配置读取。

### 3.2 V1 范围外

- 登录、账号、多用户协作。
- 远程部署或云端 Hub。
- worker marketplace。
- 完整 installer/auto-update。
- 多项目工作区管理。
- 完整上下文树可视化。
- durable worker queue。
- worker 队列跨进程恢复。
- WebSocket。
- 独立 web search、LSP、codesearch 等新公开工具。
- 绕过 Orchestrator 的点对点 `@worker` 直连模式。
- 在 Hub 内硬编码 Ollama 模型名。

## 4. 已确认技术决策

- 前端：Electron + React + TypeScript。
- 后端：FastAPI + uvicorn。
- Electron 主进程负责启动 Python 后端。
- 前端只访问 Hub backend API，不直接调用 `agent_core`。
- 实时事件使用 SSE。
- 使用已有 `~/.soong-agent/config.toml`。
- 如果配置文件不存在，Hub 启动时创建默认配置。
- 本地真实验证使用 Ollama，但通过 `openai` provider 的 OpenAI 格式 endpoint 或用户配置指定，不写死。
- 开发阶段启动命令：

```bash
PYTHONPATH=src python3 -m agent_hub.backend
```

- 未来增加仓库根目录脚本：

```bash
./agenthub
```

## 5. 术语

- Hub：`agent_hub` 桌面应用整体。
- Backend：Hub 的 FastAPI 后端。
- Frontend：Hub 的 Electron/React UI。
- Core：已有 `agent_core`。
- Conversation：Hub 侧对用户可见的会话。
- Session：`agent_core` 侧上下文树和运行历史的持久化单元。
- Run：一次 core 执行。
- Orchestrator：`agent_core` orchestrator mode 下的主 agent。
- Worker：由 worker pool 管理的可复用 worker agent。
- AgentDefinition：worker/sub/fork agent 的定义，包括 prompt、描述、工具偏好等。
- Task DAG：Orchestrator 拆解任务后用于派工和状态追踪的 DAG。
- Active path：core session 当前上下文树可见路径。
- Branch：在同一个 session 内切换到某个历史用户消息 node，并从该节点继续对话。
- Fork session：从某个历史用户消息 node 复制上下文，创建新的 core session 和新的 Hub conversation。

## 6. 核心用户旅程

### 6.1 首次启动

1. 用户运行 `./agenthub` 或前端 dev 命令。
2. Electron 主进程启动 Python 后端。
3. 后端检查 `~/.soong-agent/config.toml`。
4. 如果配置不存在，后端用内置默认模板创建。
5. 后端初始化 core runtime。
6. 后端加载内置、TOML、JSON、SQLite 中的 agent/worker 配置。
7. 前端等待 `/health` 成功后进入主界面。

验收：

- 没有配置文件时能自动创建默认配置。
- 有配置文件时不覆盖用户配置。
- 后端启动失败时前端展示明确错误，不进入空白页面。

### 6.2 普通对话

1. 用户在输入框输入一行消息。
2. 消息没有开头 mention。
3. Backend 创建 Hub user message。
4. Backend 调用 `agent_core.runtime.start(..., mode="orchestrator")`。
5. Orchestrator 正常处理。
6. 前端通过 SSE 显示运行中状态、最终回复、错误或取消结果。

验收：

- 用户消息立即显示。
- Orchestrator 回复显示为独立 sender。
- 工具/debug 细节默认不展开。
- 运行失败时显示可读错误。

### 6.3 指定 worker

1. 用户输入 `@reviewer 检查这次改动`。
2. Backend 只解析开头 mention。
3. Backend 调用 core 的 worker mention resolution。
4. 如果命中唯一 worker，Backend 启动 orchestrator run，并带上 `mentioned_worker` directive。
5. Orchestrator 仍负责创建任务、派工、汇总。
6. Core 强制 `agent.dispatch_worker` 只能派给被 mention 的 worker。
7. 指定 worker 完成后，Orchestrator 对用户总结。

验收：

- `@worker` 不直接启动 worker run。
- Orchestrator 如果尝试派给其他 worker，core 返回工具层校验错误。
- UI 能看到指定 worker 的 queued/running/completed 状态。
- worker 最终结果和 Orchestrator 总结都能在消息流中体现。

### 6.4 Worker busy 排队

1. 用户向一个 busy worker 发送 `@worker` 请求。
2. Backend/core 不重新选择其他 worker。
3. 请求进入该 worker 的内存 FIFO 队列。
4. UI 显示 queued。
5. 用户可以取消排队项。
6. worker idle 后自动取下一个 queued job 运行。

验收：

- 每个 worker 队列上限为 20。
- 超过上限返回 `worker_queue_full`。
- 取消 queued job 后不会启动。
- 重启后队列清空符合预期。

### 6.5 管理 worker

1. 用户打开右侧 worker panel。
2. 查看 worker 列表、状态、队列、工具权限。
3. 用户创建或编辑 worker。
4. Backend 调用 core worker management API。
5. Core 持久化动态配置并刷新 runtime worker pool。

验收：

- 创建后无需重启即可用于 mention。
- 禁用 worker 后不能被 mention 或 dispatch。
- 删除是软删除，历史消息仍显示原 worker 名称。
- JSON 文件和 SQLite 动态配置的优先级符合定义。

### 6.6 权限请求

1. Core 工具执行需要权限。
2. Backend permission callback 创建 request id。
3. Backend 通过 SSE 发出 `permission_requested`。
4. Frontend 在消息流或底部区域展示内联 permission card。
5. 用户选择 allow once、allow for session 或 deny。
6. Backend 解除等待，core 继续工具执行或返回 permission denied。

验收：

- 不使用系统弹窗。
- 用户不决策时运行停住等待。
- 不自动超时。
- allow for session 使用 core 现有 session permission cache。

### 6.7 分支和 fork

Branch：

1. 用户在历史用户消息上选择 branch。
2. UI 列出当前 session 的用户消息 node。
3. 用户用鼠标或键盘选择目标 node。
4. Backend 调用 core active path 切换能力。
5. 当前 conversation 继续从该 node 后面产生新路径。

Fork session：

1. 用户在历史用户消息上选择 fork。
2. Backend 从该 node 创建新的 core session。
3. Backend 创建新的 Hub conversation。
4. UI 切到新 conversation。

验收：

- Branch 是同一 session 内的 active path 切换。
- Fork 是新 session，不要求用户手动输入 source session id。
- Branch/Fork 的选择列表只展示用户消息，并显示 node id 和内容摘要。

## 7. Conversation 与 Session 关系

每个 Hub conversation 一对一绑定一个 core session。

Core 是以下数据的权威来源：

- session tree
- active path
- run/event replay
- task DAG
- worker run
- tool execution
- artifact
- permission behavior

Hub 是以下数据的权威来源：

- conversation title
- UI message order
- sender display name
- mention target
- 简化状态
- 前端展示偏好

Hub message 必须尽可能关联 core id：

- `core_session_id`
- `core_run_id`
- `core_node_id`
- `child_run_id`
- `task_id`
- `worker_id`
- `queue_id`

如果 Hub message 表损坏，后端可以从 core replay 构造只读 fallback，但正常渲染以 Hub message 为准。

## 8. Mention 规则

### 8.1 支持形式

- 无 mention：发送给 Orchestrator。
- `@Orchestrator <message>`：显式发送给 Orchestrator。
- `@worker_id <message>`：请求 Orchestrator 指派指定 worker。
- `@worker_name <message>`：当 name 唯一时可用。

### 8.2 解析规则

- 只解析消息开头的 mention。
- mention 必须是第一段非空文本。
- 中间出现的 `@xxx` 只是普通文本。
- `@@`、`//` 之类特殊逃逸语义不在 Agent Hub v1 设计中。
- 如果用户只输入 `@worker` 但没有正文，前端应提示需要输入任务内容。

### 8.3 worker resolution 优先级

1. exact `worker_id`
2. unique `name`
3. ambiguous error
4. not found error

### 8.4 错误

- `worker_not_found`
- `worker_ambiguous`
- `worker_disabled`
- `worker_deleted`
- `worker_queue_full`

错误应该以内联系统消息或输入区域校验提示展示，不应该让应用崩溃。

## 9. Orchestrator 与 Worker 语义

用户指定 worker 时，语义是：

> Orchestrator，请让这个指定 worker 完成适合它的部分，并把结果总结给我。

不是：

> 用户直接和 worker 私聊。

因此：

- `@worker` 仍启动 orchestrator mode。
- Orchestrator 仍是任务拆解和最终回复主体。
- Worker 只执行 Orchestrator 分派的 task/step。
- Core 要在工具层强校验 dispatch 目标。
- UI 可以展示 worker 的中间状态，但最终用户回复仍由 Orchestrator 收口。

## 10. Worker 配置

### 10.1 Worker 字段

V1 UI 支持：

- `worker_id`
- `name`
- `description`
- `system_prompt`
- `worker_pool_id`
- `model.provider`
- `model.name`
- `model.base_url`
- `model.api_key` 或 `api_key_env`
- `model.temperature`
- `model.max_output_tokens`
- `allowed_tools`
- `enabled`

说明：

- `worker_id` 用于 mention、历史引用、队列归属。
- `name` 用于展示和可选 mention。
- `description` 用于 Orchestrator 选择 worker 的上下文。
- `system_prompt` 进入 worker agent definition body。
- `allowed_tools` 是 worker 能力上限，不能扩大 runtime/policy/mode 的限制。
- `enabled=false` 时 worker 不可被新任务使用。

### 10.2 动态配置与文件配置

Core 应支持：

- UI 创建的 SQLite 动态配置。
- 用户手写 JSON 配置。
- 现有 `config.toml` worker pool。
- 内置 AgentDefinition。

优先级：

```text
SQLite dynamic > user JSON > config.toml > built-in defaults
```

### 10.3 JSON 目录

用户级目录：

```text
~/.soong-agent/agents/*.json
~/.soong-agent/workers/*.json
```

也可以继续兼容现有：

```text
~/.soong-agent/agents/*.md
```

### 10.4 AgentDefinition JSON 示例

```json
{
  "agent_definition_id": "code_reviewer",
  "name": "Code Reviewer",
  "description": "Reviews code changes and identifies correctness, safety, and maintainability issues.",
  "model": {
    "provider": "openai",
    "name": "qwen2.5:7b",
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "temperature": 0.2,
    "max_output_tokens": 4096
  },
  "system_prompt": "You are a senior code reviewer. Focus on concrete bugs, regressions, missing tests, and risky assumptions.",
  "suggested_tools": ["code.read_file", "code.search"],
  "tags": ["review", "code"]
}
```

### 10.5 Worker JSON 引用 agent

```json
{
  "worker_id": "reviewer_worker",
  "worker_pool_id": "default",
  "agent_definition_id": "code_reviewer",
  "enabled": true,
  "allowed_tools": ["code.read_file", "code.search"]
}
```

### 10.6 Worker JSON 内联 agent

```json
{
  "worker_id": "reviewer_worker",
  "worker_pool_id": "default",
  "enabled": true,
  "agent": {
    "agent_definition_id": "code_reviewer",
    "name": "Code Reviewer",
    "description": "Reviews code changes.",
    "model": {
      "provider": "openai",
      "name": "qwen2.5:7b",
      "base_url": "http://127.0.0.1:11434/v1",
      "api_key": "ollama",
      "temperature": 0.2
    },
    "system_prompt": "You are a senior code reviewer.",
    "suggested_tools": ["code.read_file", "code.search"],
    "tags": ["review"]
  },
  "allowed_tools": ["code.read_file", "code.search"]
}
```

### 10.7 Model 配置规则

- `model_profile` 可以引用 `config.toml` 里的命名模型配置。
- `model` 可以直接写 provider/base_url/api_key/model 参数。
- 如果同时存在 `model_profile` 和 `model`，直接写的 `model` 获胜。
- Hub UI 不写死 Ollama；示例可以用 Ollama 的 OpenAI 格式接口配置。

## 11. Tool 选择

Backend 提供 `GET /tools`。

Frontend 在 worker editor 中提供 `allowed_tools` 多选。

默认建议安全工具：

- `code.read_file`
- `code.list_dir`
- `code.search`

用户可手动开启写入或命令工具：

- `code.write_file`
- `code.edit_file`
- `code.run_command`

要求：

- 工具列表必须来自 core registry。
- UI 不维护硬编码完整工具表。
- `allowed_tools` 为空或 null 时使用 core 默认策略。
- worker 的 `allowed_tools` 只能收窄，不能绕过 permission policy。

## 12. UI 需求

### 12.1 主布局

三栏桌面布局：

```text
+----------------------+----------------------------------+----------------------+
| Conversations         | Messages                         | Workers              |
|                      |                                  |                      |
| 会话列表              | 用户 / Orchestrator / worker      | worker 状态           |
| 搜索/新建             | 权限卡片 / 状态事件               | 队列 / 编辑            |
|                      | 输入框                           | 工具权限              |
+----------------------+----------------------------------+----------------------+
```

### 12.2 左侧会话栏

功能：

- 新建 conversation。
- 列出 conversation。
- 切换 conversation。
- 显示标题、更新时间、最后一条摘要。
- 显示运行中状态。
- 支持从当前 conversation fork 后自动切换。

### 12.3 中间消息流

显示：

- user message
- Orchestrator reply
- worker result
- queued/running/completed/failed/cancelled status
- permission card
- branch/fork 操作入口

默认隐藏：

- raw provider payload
- context_built
- debug event
- raw tool call 参数

可以预留 debug 展开入口，但 V1 不要求完整 debug 面板。

### 12.4 输入框

要求：

- 支持一行一轮提交。
- 等待模型回复时仍允许输入，输入进入前端 pending 队列或创建新请求，具体由后端并发/排队策略决定。
- 输入 `@` 后展示 Orchestrator 和可用 worker 候选。
- 候选可用上下键选择。
- 选择 worker 后插入 `@worker_id `。
- 对 `@worker` 空正文做本地校验。

### 12.5 右侧 worker 面板

显示：

- worker name/id
- enabled/disabled/deleted 状态
- idle/busy/unavailable
- queue length
- current run/task/step 摘要
- allowed tools
- model/provider 摘要

操作：

- 创建 worker
- 编辑 worker
- 启用/禁用 worker
- 软删除 worker
- 查看队列
- 取消 queued job

### 12.6 视觉风格

这是工作型桌面工具，不是 landing page。

要求：

- 信息密度适中。
- 颜色克制。
- 状态清晰。
- Orchestrator、worker、user 有明显视觉区分。
- 权限卡片醒目但不阻塞整个窗口。
- 不使用营销式 hero、装饰卡片或过度渐变。

## 13. Backend API 范围

### 13.1 Health / Config

- `GET /health`
- `GET /config/status`

返回：

- backend ready
- config path
- provider
- model
- core initialized
- frontend 可展示的警告

### 13.2 Conversations

- `POST /conversations`
- `GET /conversations`
- `GET /conversations/{conversation_id}`
- `GET /conversations/{conversation_id}/messages`
- `POST /conversations/{conversation_id}/messages`
- `POST /conversations/{conversation_id}/cancel`
- `POST /conversations/{conversation_id}/branch`
- `POST /conversations/{conversation_id}/fork`

### 13.3 Workers

- `GET /workers`
- `POST /workers`
- `GET /workers/{worker_id}`
- `PATCH /workers/{worker_id}`
- `DELETE /workers/{worker_id}`
- `POST /workers/{worker_id}/enable`
- `POST /workers/{worker_id}/disable`
- `GET /workers/{worker_id}/queue`
- `POST /workers/{worker_id}/queue/{queue_id}/cancel`

### 13.4 Tools

- `GET /tools`

### 13.5 Events

- `GET /events?conversation_id=...`

### 13.6 Permissions

- `POST /permissions/{permission_request_id}/decision`

## 14. 状态模型

### 14.1 Conversation status

- `active`
- `archived`

V1 可以只实现 `active`。

### 14.2 Message status

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

### 14.3 Worker status

- `idle`
- `busy`
- `disabled`
- `deleted`
- `unavailable`

### 14.4 Queue item status

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

### 14.5 Permission status

- `pending`
- `allowed`
- `denied`
- `cancelled`

## 15. 错误处理需求

错误必须可读、可恢复。

常见错误：

- config missing bootstrap failed
- config parse failed
- backend DB open failed
- core runtime init failed
- provider unavailable
- model does not support tool calling
- worker not found
- worker ambiguous
- worker disabled
- worker queue full
- permission denied
- run cancelled
- core run failed
- SSE disconnected

UI 展示要求：

- 后端启动失败：可以使用独立错误页。
- 运行时错误：以内联消息或 banner 展示。
- worker 表单错误：字段级展示。
- permission denied：作为工具执行结果相关消息展示。

## 16. 配置与本地 Ollama

Hub 使用 `~/.soong-agent/config.toml`。

Ollama 验证推荐配置形态：

```toml
[model]
provider = "openai"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
name = "qwen2.5:7b"
```

注意：

- 这是示例，不是 Hub 内硬编码。
- 用户可以换成 OpenAI、Claude 或其他 provider 配置。
- E2E 使用本地 Ollama 时，具体模型名从配置读取。
- 如果配置不可用，health API 必须给出明确提示。

## 17. 非功能需求

### 17.1 响应性

- 用户发送消息后 UI 立即显示 user message。
- SSE 断开后 UI 显示连接状态并尝试重连。
- 长时间运行时 UI 不冻结。
- 权限等待期间应用可继续操作其他会话。

### 17.2 可观察性

- Backend 记录关键 lifecycle log：
  - startup
  - runtime init
  - conversation create
  - run start/end
  - worker dispatch
  - queue enqueue/dequeue/cancel
  - permission request/decision
- Debug 细节默认不展示给普通 UI，但后续可以接 debug panel。

### 17.3 安全与权限

- Hub 不绕过 core permission policy。
- Hub 不把 API key 明文写进日志。
- Hub DB 不保存不必要 secret。
- UI worker editor 如果允许输入 `api_key`，需要在存储前明确策略：
  - 优先推荐 `api_key_env`
  - 如果保存 direct api_key，需要避免日志和 SSE 泄漏

### 17.4 可测试性

- Backend 业务逻辑要能脱离 Electron 测试。
- Mention parser、worker resolution、DB、SSE、permission bridge 都应有单测。
- Frontend API client 和核心组件应可单测。
- 本地 Ollama E2E 作为真实验证，不替代单元测试。

## 18. 验收标准

V1 完成时至少满足：

1. `./agenthub` 可以启动桌面应用和后端。
2. 缺少 `~/.soong-agent/config.toml` 时自动创建默认配置。
3. `/health` 返回 provider/model/config 状态。
4. 可以创建新 conversation。
5. 可以发送普通消息并收到 Orchestrator 回复。
6. 可以创建 worker。
7. 输入 `@worker` 可以让 Orchestrator 指派指定 worker。
8. 指定 worker busy 时进入队列，队列上限 20。
9. 可以取消 queued job。
10. 权限请求以内联卡片展示，并一直等待用户选择。
11. 可以从用户消息 node branch。
12. 可以从用户消息 node fork 出新 conversation。
13. 可以切换历史 conversation。
14. Worker 软删除后历史消息仍能展示 worker 信息。
15. 本地 Ollama 通过配置完成至少一条真实运行链路验证。

## 19. 需要在实现前再次确认的细节

以下不是阻塞需求文档的未决项，而是进入实现计划时需要落到具体代码的选择：

- Hub DB 是否完全独立于 core SQLite，还是复用同一个 SQLite 文件的不同表。
- Dynamic worker 的 direct `api_key` 是否允许写入 SQLite，还是只允许 `api_key_env`。
- Worker 创建表单第一版是否暴露完整 model 字段，还是先只暴露 provider/name/base_url/api_key_env。
- Branch/Fork 的 core API 当前是否足够，需要补哪些最小接口。
- 前端等待模型回复时允许继续输入的具体策略：同 conversation 串行队列，还是允许多个 active runs。
