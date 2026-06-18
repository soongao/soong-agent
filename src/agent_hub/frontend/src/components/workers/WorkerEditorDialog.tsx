import { X } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { ToolView, WorkerConfigPayload, WorkerView } from "../../types";

type WorkerEditorDialogProps = {
  worker: WorkerView | null;
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

export function WorkerEditorDialog({ worker, tools, onClose, onSaved }: WorkerEditorDialogProps) {
  const [jsonText, setJsonText] = useState(initialJsonText(worker));
  const [editorMode, setEditorMode] = useState<"form" | "json">("form");
  const [error, setError] = useState<string | null>(null);
  const editor = safeJsonObject(jsonText);
  const title = worker ? `Edit ${worker.worker_id}` : "Create Worker";

  useEffect(() => {
    setJsonText(initialJsonText(worker));
    setEditorMode("form");
    setError(null);
  }, [worker]);

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

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.currentTarget === event.target && onClose()}>
      <section className="worker-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="worker-editor-title">
        <div className="dialog-header">
          <div>
            <strong id="worker-editor-title">{title}</strong>
            <small>{worker ? worker.name || worker.worker_id : "Dynamic worker"}</small>
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
                  Allowed Tools
                  <select
                    multiple
                    value={Array.isArray(editor.allowed_tools) ? editor.allowed_tools.map(String) : []}
                    onChange={(event) =>
                      updateJsonField(
                        "allowed_tools",
                        Array.from(event.currentTarget.selectedOptions).map((option) => option.value),
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

function initialJsonText(worker: WorkerView | null): string {
  return JSON.stringify(worker ? workerToEditorJson(worker) : emptyWorkerJson, null, 2);
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
  if (worker.model !== undefined && worker.model !== null && Object.keys(worker.model as Record<string, unknown>).length === 0) {
    delete worker.model;
  }
  return worker as WorkerConfigPayload;
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
