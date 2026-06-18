import { X } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { ToolView, WorkerConfigPayload, WorkerView } from "../../types";

type WorkerEditorDialogProps = {
  worker: WorkerView | null;
  kind: "internal" | "external";
  tools: ToolView[];
  onClose: () => void;
  onSaved: () => Promise<void> | void;
};

const safeIdPattern = /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/;

const emptyWorkerJson = {
  worker_id: "",
  name: "",
  description: "",
  system_prompt: "",
  worker_pool_id: "default",
  model: {},
  allowed_tools: ["code.read_file", "code.list_dir", "code.search"],
  enabled: true,
};

const emptyExternalWorkerJson = {
  worker_id: "",
  name: "",
  description: "",
  system_prompt: "You are an external worker. Treat the orchestrator dispatch as the user's request and return the result clearly.",
  worker_pool_id: "default",
  allowed_tools: [],
  enabled: true,
  metadata: {
    worker_executor: {
      type: "",
      config: {},
    },
  },
};

const supportedExternalExecutors = [
  {
    label: "OpenCode",
    type: "opencode",
    description: "OpenCode adapter",
  },
  {
    label: "Codex",
    type: "codex_pty",
    description: "Codex PTY adapter",
  },
];

export function WorkerEditorDialog({ worker, kind, tools, onClose, onSaved }: WorkerEditorDialogProps) {
  const [jsonText, setJsonText] = useState(initialJsonText(worker, kind));
  const [editorMode, setEditorMode] = useState<"form" | "json">("form");
  const [error, setError] = useState<string | null>(null);
  const editor = safeJsonObject(jsonText);
  const title = worker ? `Edit ${worker.worker_id}` : "Create Worker";
  const editorKind = worker ? workerKind(worker) : kind;
  const executor = workerExecutor(editor);
  const executorConfigText = executor.configText;
  const executorConfigError = executor.configError;
  const allowedToolsText = allowedToolsValue(editor).join("\n");

  useEffect(() => {
    setJsonText(initialJsonText(worker, kind));
    setEditorMode("form");
    setError(null);
  }, [worker, kind]);

  async function submitWorker(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const payload = parseWorkerJson(jsonText);
      if (worker) {
        const { worker_id: _workerId, ...updatePayload } = payload;
        await api.updateWorker(worker.worker_id, updatePayload);
      } else {
        await api.createWorker(payload);
      }
      await onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function updateJsonField(key: string, value: unknown) {
    try {
      const current = JSON.parse(jsonText) as Record<string, unknown>;
      current[key] = value;
      setJsonText(JSON.stringify(current, null, 2));
      setError(null);
    } catch {
      setError("Switch to valid JSON before editing form fields.");
    }
  }

  function updateModelField(key: string, value: unknown) {
    try {
      const current = JSON.parse(jsonText) as Record<string, unknown>;
      const model = current.model && typeof current.model === "object" && !Array.isArray(current.model) ? { ...(current.model as Record<string, unknown>) } : {};
      if (value === "") {
        delete model[key];
      } else {
        model[key] = value;
      }
      current.model = model;
      setJsonText(JSON.stringify(current, null, 2));
      setError(null);
    } catch {
      setError("Switch to valid JSON before editing form fields.");
    }
  }

  function updateWorkerExecutorType(value: string) {
    try {
      const current = JSON.parse(jsonText) as Record<string, unknown>;
      const metadata = current.metadata && typeof current.metadata === "object" && !Array.isArray(current.metadata) ? { ...(current.metadata as Record<string, unknown>) } : {};
      const existing = metadata.worker_executor && typeof metadata.worker_executor === "object" && !Array.isArray(metadata.worker_executor)
        ? { ...(metadata.worker_executor as Record<string, unknown>) }
        : {};
      if (!value.trim()) {
        delete metadata.worker_executor;
      } else {
        metadata.worker_executor = { ...existing, type: value.trim(), config: executor.config };
      }
      current.metadata = metadata;
      setJsonText(JSON.stringify(current, null, 2));
      setError(null);
    } catch {
      setError("Switch to valid JSON before editing form fields.");
    }
  }

  function updateWorkerExecutorConfig(value: string) {
    try {
      const parsedConfig = value.trim() ? JSON.parse(value) : {};
      if (!parsedConfig || typeof parsedConfig !== "object" || Array.isArray(parsedConfig)) {
        setError("External executor config must be a JSON object.");
        return;
      }
      const current = JSON.parse(jsonText) as Record<string, unknown>;
      const metadata = current.metadata && typeof current.metadata === "object" && !Array.isArray(current.metadata) ? { ...(current.metadata as Record<string, unknown>) } : {};
      const existing = metadata.worker_executor && typeof metadata.worker_executor === "object" && !Array.isArray(metadata.worker_executor)
        ? { ...(metadata.worker_executor as Record<string, unknown>) }
        : {};
      metadata.worker_executor = { ...existing, type: executor.type, config: parsedConfig };
      current.metadata = metadata;
      setJsonText(JSON.stringify(current, null, 2));
      setError(null);
    } catch {
      setError("External executor config must be valid JSON.");
    }
  }

  function updateAllowedToolsText(value: string) {
    updateJsonField(
      "allowed_tools",
      value
        .split(/\r?\n|,/)
        .map((tool) => tool.trim())
        .filter(Boolean),
    );
  }

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.currentTarget === event.target && onClose()}>
      <section className="worker-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="worker-editor-title">
        <div className="dialog-header">
          <div>
            <strong id="worker-editor-title">{title}</strong>
            <small>{worker ? worker.name || worker.worker_id : editorKind === "external" ? "External worker" : "Internal worker"}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close worker editor">
            <X size={15} />
          </button>
        </div>
        <form className="worker-editor worker-editor-dialog-form" onSubmit={submitWorker}>
          <div className="editor-title">
            <span>Configuration</span>
            <div className="segmented">
              <button type="button" className={editorMode === "form" ? "active" : ""} onClick={() => setEditorMode("form")}>
                Form
              </button>
              <button type="button" className={editorMode === "json" ? "active" : ""} onClick={() => setEditorMode("json")}>
                JSON
              </button>
            </div>
          </div>
          <div className="worker-editor-body">
            {editorMode === "form" ? (
              <>
                <label>
                  Worker ID
                  <input
                    value={String(editor.worker_id ?? "")}
                    disabled={Boolean(worker)}
                    onChange={(event) => updateJsonField("worker_id", event.target.value)}
                    placeholder="reviewer_worker"
                  />
                </label>
                <label>
                  Name
                  <input value={String(editor.name ?? "")} onChange={(event) => updateJsonField("name", event.target.value)} placeholder="Reviewer" />
                </label>
                <label>
                  Description
                  <input
                    value={String(editor.description ?? "")}
                    onChange={(event) => updateJsonField("description", event.target.value)}
                    placeholder="Reviews code changes."
                  />
                </label>
                <label>
                  System Prompt
                  <textarea
                    value={String(editor.system_prompt ?? "")}
                    onChange={(event) => updateJsonField("system_prompt", event.target.value)}
                    placeholder="You are a focused worker."
                  />
                </label>
                {editorKind === "internal" ? (
                  <>
                    <fieldset className="model-fieldset">
                      <legend>Model</legend>
                      <label>
                        Provider
                        <input value={String(modelField(editor, "provider") ?? "")} onChange={(event) => updateModelField("provider", event.target.value)} placeholder="openai" />
                      </label>
                      <label>
                        Model Name
                        <input value={String(modelField(editor, "name") ?? "")} onChange={(event) => updateModelField("name", event.target.value)} placeholder="qwen2.5:7b" />
                      </label>
                      <label>
                        Base URL
                        <input
                          value={String(modelField(editor, "base_url") ?? "")}
                          onChange={(event) => updateModelField("base_url", event.target.value)}
                          placeholder="http://127.0.0.1:11434/v1"
                        />
                      </label>
                      <label>
                        API Key
                        <input value={String(modelField(editor, "api_key") ?? "")} onChange={(event) => updateModelField("api_key", event.target.value)} placeholder="ollama" />
                      </label>
                      <label>
                        API Key Env
                        <input
                          value={String(modelField(editor, "api_key_env") ?? "")}
                          onChange={(event) => updateModelField("api_key_env", event.target.value)}
                          placeholder="OPENAI_API_KEY"
                        />
                      </label>
                      <label>
                        Temperature
                        <input
                          type="number"
                          step="0.1"
                          value={String(modelField(editor, "temperature") ?? "")}
                          onChange={(event) => updateModelField("temperature", event.target.value === "" ? "" : Number(event.target.value))}
                          placeholder="0.2"
                        />
                      </label>
                      <label>
                        Max Output Tokens
                        <input
                          type="number"
                          step="1"
                          value={String(modelField(editor, "max_output_tokens") ?? "")}
                          onChange={(event) => updateModelField("max_output_tokens", event.target.value === "" ? "" : Number(event.target.value))}
                          placeholder="4096"
                        />
                      </label>
                    </fieldset>
                    <label>
                      Allowed Tools From Registry
                      <select
                        multiple
                        value={allowedToolsValue(editor).filter((tool) => tools.some((known) => known.name === tool))}
                        onChange={(event) =>
                          updateJsonField(
                            "allowed_tools",
                            mergeTools(
                              allowedToolsValue(editor).filter((tool) => !tools.some((known) => known.name === tool)),
                              Array.from(event.currentTarget.selectedOptions).map((option) => option.value),
                            ),
                          )
                        }
                      >
                        {tools.map((tool) => (
                          <option key={tool.name} value={tool.name}>
                            {tool.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                ) : (
                  <>
                    <fieldset className="model-fieldset">
                      <legend>External Executor</legend>
                      <div className="external-executor-support" aria-label="Supported external workers">
                        <div className="external-executor-support-header">
                          <span>Supported External Workers</span>
                          <small>Other external workers need an adapter first.</small>
                        </div>
                        <div className="external-executor-options">
                          {supportedExternalExecutors.map((option) => (
                            <button
                              type="button"
                              key={option.type}
                              className={executor.type === option.type ? "executor-option active" : "executor-option"}
                              aria-pressed={executor.type === option.type}
                              aria-label={`Use ${option.label} external worker`}
                              onClick={() => updateWorkerExecutorType(option.type)}
                            >
                              <strong>{option.label}</strong>
                              <span>{option.description}</span>
                              <code>{option.type}</code>
                            </button>
                          ))}
                        </div>
                      </div>
                      <label>
                        Executor Type
                        <input
                          value={executor.type}
                          onChange={(event) => updateWorkerExecutorType(event.target.value)}
                          placeholder="opencode"
                        />
                      </label>
                      <label>
                        Executor Config JSON
                        <textarea
                          className="executor-config-editor"
                          value={executorConfigText}
                          onChange={(event) => updateWorkerExecutorConfig(event.target.value)}
                          placeholder={'{\n  "binary": "opencode",\n  "cwd": "/path/to/project",\n  "args": []\n}'}
                        />
                      </label>
                      {executorConfigError ? <span className="field-error">{executorConfigError}</span> : null}
                    </fieldset>
                    <label>
                      Allowed Tools Override
                      <textarea
                        className="executor-config-editor"
                        value={allowedToolsText}
                        onChange={(event) => updateAllowedToolsText(event.target.value)}
                        placeholder={"opencode.acp"}
                      />
                    </label>
                  </>
                )}
              </>
            ) : (
              <label>
                Worker JSON
                <textarea className="json-editor" value={jsonText} onChange={(event) => setJsonText(event.target.value)} />
              </label>
            )}
          </div>
          {error ? <div className="form-error">{error}</div> : null}
          <div className="worker-editor-actions">
            <button type="button" onClick={onClose}>
              Cancel
            </button>
            <button type="submit">{worker ? "Update Worker" : "Create Worker"}</button>
          </div>
        </form>
      </section>
    </div>
  );
}

function initialJsonText(worker: WorkerView | null, kind: "internal" | "external"): string {
  return JSON.stringify(worker ? workerToEditorJson(worker) : kind === "external" ? emptyExternalWorkerJson : emptyWorkerJson, null, 2);
}

function workerToEditorJson(worker: WorkerView): Record<string, unknown> {
  return {
    worker_id: worker.worker_id,
    worker_pool_id: worker.worker_pool_id,
    agent_definition_id: worker.system_prompt ? undefined : worker.agent_definition_id,
    name: worker.name,
    description: worker.description,
    system_prompt: worker.system_prompt ?? "",
    model_profile: worker.model_profile ?? undefined,
    model: worker.model ?? undefined,
    allowed_tools: worker.allowed_tools ?? [],
    enabled: worker.enabled,
    metadata: worker.metadata ?? {},
  };
}

function safeJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function modelField(editor: Record<string, unknown>, key: string): unknown {
  return editor.model && typeof editor.model === "object" && !Array.isArray(editor.model) ? (editor.model as Record<string, unknown>)[key] : undefined;
}

function workerKind(worker: WorkerView): "internal" | "external" {
  return worker.metadata && typeof worker.metadata === "object" && workerExecutor({ metadata: worker.metadata }).type ? "external" : "internal";
}

function allowedToolsValue(editor: Record<string, unknown>): string[] {
  return Array.isArray(editor.allowed_tools) ? editor.allowed_tools.map(String) : [];
}

function mergeTools(left: string[], right: string[]): string[] {
  return Array.from(new Set([...left, ...right].map((tool) => tool.trim()).filter(Boolean)));
}

function workerExecutor(editor: Record<string, unknown>): {
  type: string;
  config: Record<string, unknown>;
  configText: string;
  configError: string | null;
} {
  const metadata = editor.metadata && typeof editor.metadata === "object" && !Array.isArray(editor.metadata) ? editor.metadata as Record<string, unknown> : {};
  const rawExecutor = metadata.worker_executor;
  if (!rawExecutor || typeof rawExecutor !== "object" || Array.isArray(rawExecutor)) {
    return { type: "", config: {}, configText: "{}", configError: null };
  }
  const executor = rawExecutor as Record<string, unknown>;
  const rawConfig = executor.config;
  const config = rawConfig && typeof rawConfig === "object" && !Array.isArray(rawConfig) ? rawConfig as Record<string, unknown> : {};
  return {
    type: typeof executor.type === "string" ? executor.type : "",
    config,
    configText: JSON.stringify(config, null, 2),
    configError: rawConfig !== undefined && rawConfig !== null && (typeof rawConfig !== "object" || Array.isArray(rawConfig))
      ? "metadata.worker_executor.config must be an object."
      : null,
  };
}

function parseWorkerJson(value: string): WorkerConfigPayload {
  const parsed = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Worker JSON must be an object.");
  }
  const worker = parsed as Record<string, unknown>;
  const workerId = String(worker.worker_id ?? "").trim();
  const workerPoolId = String(worker.worker_pool_id ?? "default").trim();
  const agentDefinitionId = worker.agent_definition_id === undefined || worker.agent_definition_id === null ? "" : String(worker.agent_definition_id).trim();
  const inlineAgent = worker.agent;
  if (!workerId) {
    throw new Error("worker_id is required.");
  }
  if (!safeIdPattern.test(workerId)) {
    throw new Error("worker_id contains unsafe characters.");
  }
  if (workerPoolId && !safeIdPattern.test(workerPoolId)) {
    throw new Error("worker_pool_id contains unsafe characters.");
  }
  if (agentDefinitionId && !safeIdPattern.test(agentDefinitionId)) {
    throw new Error("agent_definition_id contains unsafe characters.");
  }
  if (inlineAgent !== undefined && inlineAgent !== null) {
    validateInlineAgent(inlineAgent);
  }
  if (!String(worker.system_prompt ?? "").trim() && !agentDefinitionId && !inlineAgent) {
    throw new Error("system_prompt, agent_definition_id, or agent is required.");
  }
  if (worker.model !== undefined && worker.model !== null && (typeof worker.model !== "object" || Array.isArray(worker.model))) {
    throw new Error("model must be an object.");
  }
  if (worker.allowed_tools !== undefined && worker.allowed_tools !== null) {
    if (!Array.isArray(worker.allowed_tools) || !worker.allowed_tools.every((tool: unknown) => typeof tool === "string" && tool.trim())) {
      throw new Error("allowed_tools must be a list of tool names.");
    }
  }
  validateWorkerExecutor(worker.metadata);
  if (worker.model !== undefined && worker.model !== null && Object.keys(worker.model as Record<string, unknown>).length === 0) {
    delete worker.model;
  }
  return worker as WorkerConfigPayload;
}

function validateWorkerExecutor(value: unknown) {
  if (value === undefined || value === null) return;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("metadata must be an object.");
  }
  const metadata = value as Record<string, unknown>;
  const rawExecutor = metadata.worker_executor;
  if (rawExecutor === undefined || rawExecutor === null) return;
  if (typeof rawExecutor !== "object" || Array.isArray(rawExecutor)) {
    throw new Error("metadata.worker_executor must be an object.");
  }
  const executor = rawExecutor as Record<string, unknown>;
  if (typeof executor.type !== "string" || !executor.type.trim()) {
    throw new Error("metadata.worker_executor.type is required.");
  }
  if (!safeIdPattern.test(executor.type.trim())) {
    throw new Error("metadata.worker_executor.type contains unsafe characters.");
  }
  if (executor.config !== undefined && executor.config !== null && (typeof executor.config !== "object" || Array.isArray(executor.config))) {
    throw new Error("metadata.worker_executor.config must be an object.");
  }
}

function validateInlineAgent(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("agent must be an object.");
  }
  const agent = value as Record<string, unknown>;
  const agentDefinitionId = String(agent.agent_definition_id ?? agent.id ?? "").trim();
  if (!agentDefinitionId) {
    throw new Error("agent.agent_definition_id is required.");
  }
  if (!safeIdPattern.test(agentDefinitionId)) {
    throw new Error("agent.agent_definition_id contains unsafe characters.");
  }
  if (!String(agent.name ?? "").trim()) {
    throw new Error("agent.name is required.");
  }
  if (!String(agent.body ?? agent.system_prompt ?? "").trim()) {
    throw new Error("agent.system_prompt is required.");
  }
  if (agent.model !== undefined && agent.model !== null && (typeof agent.model !== "object" || Array.isArray(agent.model))) {
    throw new Error("agent.model must be an object.");
  }
  for (const key of ["suggested_tools", "tags"]) {
    const list = agent[key];
    if (list !== undefined && list !== null && (!Array.isArray(list) || !list.every((item: unknown) => typeof item === "string" && item.trim()))) {
      throw new Error(`agent.${key} must be a list of strings.`);
    }
  }
}
