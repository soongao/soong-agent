import { Check, ChevronDown, ChevronRight, Minus, Pencil, Plus, Trash2, X } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import { useAppDispatch, useAppState } from "../../state/store";
import type { WorkerQueueItem, WorkerView } from "../../types";
import { WorkerEditorDialog } from "./WorkerEditorDialog";

export function WorkerPanel() {
  const { activeConversationId, conversationWorkersByConversation, workers, tools } = useAppState();
  const dispatch = useAppDispatch();
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
  const [editingWorkerId, setEditingWorkerId] = useState<string | null | undefined>(undefined);
  const [editorKind, setEditorKind] = useState<"internal" | "external">("internal");
  const [queueItems, setQueueItems] = useState<WorkerQueueItem[]>([]);
  const [collapsedSections, setCollapsedSections] = useState({
    conversation: false,
    available: true,
  });
  const selectedWorker = useMemo(
    () => workers.find((worker) => worker.worker_id === selectedWorkerId) ?? null,
    [selectedWorkerId, workers],
  );
  const editingWorker = useMemo(
    () => (editingWorkerId ? workers.find((worker) => worker.worker_id === editingWorkerId) ?? null : null),
    [editingWorkerId, workers],
  );
  const editorOpen = editingWorkerId !== undefined;
  const conversationWorkers = activeConversationId ? conversationWorkersByConversation[activeConversationId] ?? [] : [];
  const conversationWorkerIds = useMemo(() => new Set(conversationWorkers.map((worker) => worker.worker_id)), [conversationWorkers]);
  const availableWorkers = useMemo(
    () => workers.filter((worker) => !worker.deleted_at && !conversationWorkerIds.has(worker.worker_id)),
    [conversationWorkerIds, workers],
  );

  useEffect(() => {
    if (!selectedWorker) {
      setQueueItems([]);
      return;
    }
    void refreshQueue(selectedWorker.worker_id);
  }, [selectedWorker]);

  async function refresh() {
    const response = await api.workers();
    dispatch({ type: "workers", workers: response.workers });
    if (activeConversationId) {
      const conversationWorkersResponse = await api.conversationWorkers(activeConversationId);
      dispatch({ type: "conversationWorkers", conversationId: activeConversationId, workers: conversationWorkersResponse.workers });
    }
  }

  async function refreshQueue(workerId = selectedWorker?.worker_id) {
    if (!workerId) return;
    const response = await api.workerQueue(workerId);
    setQueueItems(Array.isArray(response.queue) ? response.queue : []);
  }

  async function deleteWorker(workerId: string) {
    await api.deleteWorker(workerId);
    if (selectedWorkerId === workerId) {
      setSelectedWorkerId(null);
    }
    if (editingWorkerId === workerId) {
      setEditingWorkerId(undefined);
    }
    await refresh();
  }

  async function toggleWorker(worker: WorkerView) {
    if (worker.enabled) {
      await api.disableWorker(worker.worker_id);
    } else {
      await api.enableWorker(worker.worker_id);
    }
    await refresh();
  }

  async function cancelQueueItem(item: WorkerQueueItem) {
    await api.cancelWorkerQueue(item.worker_id, item.queue_id);
    await refreshQueue(item.worker_id);
    await refresh();
  }

  async function addToConversation(workerId: string) {
    if (!activeConversationId) return;
    await api.addConversationWorker(activeConversationId, workerId);
    await refresh();
  }

  async function removeFromConversation(workerId: string) {
    if (!activeConversationId) return;
    await api.removeConversationWorker(activeConversationId, workerId);
    if (selectedWorkerId === workerId) {
      setSelectedWorkerId(null);
    }
    await refresh();
  }

  function toggleSection(section: keyof typeof collapsedSections) {
    setCollapsedSections((current) => ({ ...current, [section]: !current[section] }));
  }

  return (
    <aside className="worker-panel">
      <div className="pane-header">
        <span>Workers</span>
        <div className="header-actions">
          <button
            type="button"
            className="header-action"
            onClick={() => {
              setEditorKind("internal");
              setEditingWorkerId(null);
            }}
          >
            <Plus size={14} />
            New Internal
          </button>
          <button
            type="button"
            className="header-action"
            onClick={() => {
              setEditorKind("external");
              setEditingWorkerId(null);
            }}
          >
            <Plus size={14} />
            New External
          </button>
        </div>
      </div>
      <div className="worker-list">
        <div className="worker-list-section">
          <WorkerSectionHeader
            title="Conversation workers"
            count={conversationWorkers.length}
            collapsed={collapsedSections.conversation}
            onToggle={() => toggleSection("conversation")}
          />
          {!collapsedSections.conversation ? (
            <>
              {conversationWorkers.length === 0 ? <small className="worker-empty">No workers added to this conversation.</small> : null}
              {conversationWorkers.map((worker) => (
                <WorkerRow
                  key={worker.worker_id}
                  worker={worker}
                  active={selectedWorkerId === worker.worker_id}
                  onSelect={() => setSelectedWorkerId(worker.worker_id)}
                  actions={
                    <>
                      <button className="icon-button" onClick={() => removeFromConversation(worker.worker_id)} aria-label={`Remove ${worker.worker_id} from conversation`}>
                        <Minus size={14} />
                      </button>
                      <WorkerManagementActions
                        worker={worker}
                        onEdit={() => {
                          setSelectedWorkerId(worker.worker_id);
                          setEditorKind(workerKind(worker));
                          setEditingWorkerId(worker.worker_id);
                        }}
                        onToggle={() => toggleWorker(worker)}
                        onDelete={() => deleteWorker(worker.worker_id)}
                      />
                    </>
                  }
                />
              ))}
            </>
          ) : null}
        </div>
        <div className="worker-list-section">
          <WorkerSectionHeader
            title="Available workers"
            count={availableWorkers.length}
            collapsed={collapsedSections.available}
            onToggle={() => toggleSection("available")}
          />
          {!collapsedSections.available ? (
            <>
              {availableWorkers.length === 0 ? <small className="worker-empty">No available workers.</small> : null}
              {availableWorkers.map((worker) => (
                <WorkerRow
                  key={worker.worker_id}
                  worker={worker}
                  active={selectedWorkerId === worker.worker_id}
                  onSelect={() => setSelectedWorkerId(worker.worker_id)}
                  actions={
                    <>
                      <button
                        className="icon-button"
                        disabled={!activeConversationId || !worker.enabled}
                        onClick={() => addToConversation(worker.worker_id)}
                        aria-label={`Add ${worker.worker_id} to conversation`}
                      >
                        <Plus size={14} />
                      </button>
                      <WorkerManagementActions
                        worker={worker}
                        onEdit={() => {
                          setSelectedWorkerId(worker.worker_id);
                          setEditorKind(workerKind(worker));
                          setEditingWorkerId(worker.worker_id);
                        }}
                        onToggle={() => toggleWorker(worker)}
                        onDelete={() => deleteWorker(worker.worker_id)}
                      />
                    </>
                  }
                />
              ))}
            </>
          ) : null}
        </div>
      </div>
      {selectedWorker ? (
        <section className="worker-queue">
          <div className="queue-header">
            <strong>Queue</strong>
            <button onClick={() => refreshQueue()} type="button">
              Refresh
            </button>
          </div>
          {queueItems.length === 0 ? <small>No queued items.</small> : null}
          {queueItems.map((item) => (
            <div key={item.queue_id} className="queue-row">
              <span>{item.task_id}</span>
              <small>
                {item.status} · {item.queue_id}
              </small>
              {item.status === "queued" ? (
                <button type="button" onClick={() => cancelQueueItem(item)}>
                  Cancel
                </button>
              ) : null}
            </div>
          ))}
        </section>
      ) : null}
      {editorOpen ? <WorkerEditorDialog worker={editingWorker} kind={editorKind} tools={tools} onClose={() => setEditingWorkerId(undefined)} onSaved={refresh} /> : null}
    </aside>
  );
}

function WorkerSectionHeader({
  collapsed,
  count,
  onToggle,
  title,
}: {
  collapsed: boolean;
  count: number;
  onToggle: () => void;
  title: string;
}) {
  return (
    <button type="button" className="worker-list-title" aria-expanded={!collapsed} onClick={onToggle}>
      {collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
      <span>{title}</span>
      <span className="worker-list-count">{count}</span>
    </button>
  );
}

function WorkerRow({
  worker,
  active,
  actions,
  onSelect,
}: {
  worker: WorkerView;
  active: boolean;
  actions: ReactNode;
  onSelect: () => void;
}) {
  return (
    <div className={`worker-row${active ? " active" : ""}`}>
      <button className="worker-select" onClick={onSelect} aria-label={`View ${worker.worker_id}`}>
        <strong>{worker.name || worker.worker_id}</strong>
        <small>
          {worker.worker_id} · {worker.enabled ? worker.status : "disabled"} · queue {worker.queue_length}
        </small>
        <small>{workerRuntimeSummary(worker)}</small>
        <small>{workerModelSummary(worker)}</small>
      </button>
      <div className="worker-actions">{actions}</div>
    </div>
  );
}

function WorkerManagementActions({
  worker,
  onDelete,
  onEdit,
  onToggle,
}: {
  worker: WorkerView;
  onDelete: () => void;
  onEdit: () => void;
  onToggle: () => void;
}) {
  if (worker.source !== "dynamic" || worker.deleted_at) return null;
  return (
    <>
      <button className="icon-button" onClick={onEdit} aria-label={`Edit ${worker.worker_id}`}>
        <Pencil size={14} />
      </button>
      <button className="icon-button" onClick={onToggle} aria-label={`${worker.enabled ? "Disable" : "Enable"} ${worker.worker_id}`}>
        {worker.enabled ? <X size={14} /> : <Check size={14} />}
      </button>
      <button className="icon-button" onClick={onDelete} aria-label={`Delete ${worker.worker_id}`}>
        <Trash2 size={14} />
      </button>
    </>
  );
}

function workerKind(worker: WorkerView): "internal" | "external" {
  const metadata = worker.metadata;
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) return "internal";
  const executor = metadata.worker_executor;
  if (!executor || typeof executor !== "object" || Array.isArray(executor)) return "internal";
  const executorRecord = executor as Record<string, unknown>;
  return typeof executorRecord.type === "string" && executorRecord.type.trim() ? "external" : "internal";
}

function workerRuntimeSummary(worker: WorkerView): string {
  const parts = [
    worker.current_task_id ? `task ${worker.current_task_id}` : "",
    worker.current_step_id ? `step ${worker.current_step_id}` : "",
    worker.current_run_id ? `run ${worker.current_run_id}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "no active task";
}

function workerModelSummary(worker: WorkerView): string {
  if (worker.model && typeof worker.model === "object") {
    const provider = typeof worker.model.provider === "string" ? worker.model.provider : "";
    const name = typeof worker.model.name === "string" ? worker.model.name : "";
    return [provider, name].filter(Boolean).join(" · ") || "inherits default model";
  }
  return worker.model_profile ? `profile ${worker.model_profile}` : "inherits default model";
}
