import { Check, Pencil, Plus, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import { useAppDispatch, useAppState } from "../../state/store";
import type { WorkerQueueItem, WorkerView } from "../../types";
import { WorkerEditorDialog } from "./WorkerEditorDialog";

export function WorkerPanel() {
  const { workers, tools } = useAppState();
  const dispatch = useAppDispatch();
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
  const [editingWorkerId, setEditingWorkerId] = useState<string | null | undefined>(undefined);
  const [queueItems, setQueueItems] = useState<WorkerQueueItem[]>([]);
  const selectedWorker = useMemo(
    () => workers.find((worker) => worker.worker_id === selectedWorkerId) ?? null,
    [selectedWorkerId, workers],
  );
  const editingWorker = useMemo(
    () => (editingWorkerId ? workers.find((worker) => worker.worker_id === editingWorkerId) ?? null : null),
    [editingWorkerId, workers],
  );
  const editorOpen = editingWorkerId !== undefined;

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
  }

  async function refreshQueue(workerId = selectedWorker?.worker_id) {
    if (!workerId) return;
    const response = await api.workerQueue(workerId);
    setQueueItems(response.queue);
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

  return (
    <aside className="worker-panel">
      <div className="pane-header">
        <span>Workers</span>
        <button type="button" className="header-action" onClick={() => setEditingWorkerId(null)}>
          <Plus size={14} />
          New Worker
        </button>
      </div>
      <div className="worker-list">
        {workers.map((worker) => (
          <div key={worker.worker_id} className={`worker-row${selectedWorkerId === worker.worker_id ? " active" : ""}`}>
            <button className="worker-select" onClick={() => setSelectedWorkerId(worker.worker_id)} aria-label={`View ${worker.worker_id}`}>
              <strong>{worker.name || worker.worker_id}</strong>
              <small>
                {worker.worker_id} · {worker.enabled ? worker.status : "disabled"} · queue {worker.queue_length}
              </small>
              <small>{workerRuntimeSummary(worker)}</small>
              <small>{workerModelSummary(worker)}</small>
            </button>
            <div className="worker-actions">
              {worker.source === "dynamic" && !worker.deleted_at ? (
                <>
                  <button
                    className="icon-button"
                    onClick={() => {
                      setSelectedWorkerId(worker.worker_id);
                      setEditingWorkerId(worker.worker_id);
                    }}
                    aria-label={`Edit ${worker.worker_id}`}
                  >
                    <Pencil size={14} />
                  </button>
                  <button className="icon-button" onClick={() => toggleWorker(worker)} aria-label={`${worker.enabled ? "Disable" : "Enable"} ${worker.worker_id}`}>
                    {worker.enabled ? <X size={14} /> : <Check size={14} />}
                  </button>
                  <button className="icon-button" onClick={() => deleteWorker(worker.worker_id)} aria-label={`Delete ${worker.worker_id}`}>
                    <Trash2 size={14} />
                  </button>
                </>
              ) : null}
            </div>
          </div>
        ))}
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
      {editorOpen ? <WorkerEditorDialog worker={editingWorker} tools={tools} onClose={() => setEditingWorkerId(undefined)} onSaved={refresh} /> : null}
    </aside>
  );
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
