import type { BranchableNode, Conversation, HealthStatus, Message, ToolView, WorkerConfigPayload, WorkerQueueItem, WorkerView } from "../types";

function configuredBackendBaseUrl(): string {
  const injected = window.agentHub?.backendBaseUrl?.trim();
  if (injected) {
    console.info(`[agenthub-api] backend url from preload: ${injected}`);
    return injected;
  }
  const viteValue = import.meta.env.VITE_AGENTHUB_BACKEND_URL?.trim();
  if (viteValue) {
    console.info(`[agenthub-api] backend url from vite env: ${viteValue}`);
    return viteValue;
  }
  console.warn("[agenthub-api] backend url fallback: http://127.0.0.1:8765");
  return "http://127.0.0.1:8765";
}

export const backendBaseUrl = configuredBackendBaseUrl();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${backendBaseUrl}${path}`;
  console.info(`[agenthub-api] ${init?.method ?? "GET"} ${url}`);
  const response = await fetch(url, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  console.info(`[agenthub-api] ${response.status} ${init?.method ?? "GET"} ${url}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json() as Promise<T>;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { error?: { code?: string; message?: string } };
    if (data.error?.code && data.error.message) {
      return `${data.error.code}: ${data.error.message}`;
    }
    if (data.error?.message) {
      return data.error.message;
    }
  } catch {
    // Fall back to the HTTP status when the backend cannot return JSON.
  }
  return `${response.status} ${response.statusText}`;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  configStatus: () => request<HealthStatus>("/config/status"),
  conversations: () => request<{ conversations: Conversation[] }>("/conversations"),
  conversation: (conversationId: string) => request<Conversation>(`/conversations/${conversationId}`),
  createConversation: (title = "New conversation") =>
    request<Conversation>("/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  deleteConversation: (conversationId: string) => request<Conversation>(`/conversations/${conversationId}`, { method: "DELETE" }),
  messages: (conversationId: string) => request<{ messages: Message[] }>(`/conversations/${conversationId}/messages`),
  sendMessage: (conversationId: string, text: string) =>
    request<{ message_id: string; conversation_id: string; core_session_id: string; core_run_id: string; status: string }>(
      `/conversations/${conversationId}/messages`,
      { method: "POST", body: JSON.stringify({ text }) },
    ),
  loadSkill: (conversationId: string, skillName: string) =>
    request<Record<string, unknown>>(`/conversations/${conversationId}/skills/${encodeURIComponent(skillName)}/load`, {
      method: "POST",
      body: JSON.stringify({ name: skillName }),
    }),
  cancelConversation: (conversationId: string, payload: { core_run_id?: string | null; queue_id?: string | null }) =>
    request<Record<string, unknown>>(`/conversations/${conversationId}/cancel`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  workers: () => request<{ workers: WorkerView[] }>("/workers"),
  worker: (workerId: string) => request<WorkerView>(`/workers/${workerId}`),
  createWorker: (worker: WorkerConfigPayload) =>
    request<WorkerView>("/workers", { method: "POST", body: JSON.stringify(worker) }),
  updateWorker: (workerId: string, worker: Partial<WorkerConfigPayload>) =>
    request<WorkerView>(`/workers/${workerId}`, { method: "PATCH", body: JSON.stringify(worker) }),
  enableWorker: (workerId: string) => request<WorkerView>(`/workers/${workerId}/enable`, { method: "POST" }),
  disableWorker: (workerId: string) => request<WorkerView>(`/workers/${workerId}/disable`, { method: "POST" }),
  deleteWorker: (workerId: string) => request<WorkerView>(`/workers/${workerId}`, { method: "DELETE" }),
  workerQueue: (workerId: string) => request<{ queue: WorkerQueueItem[] }>(`/workers/${workerId}/queue`),
  cancelWorkerQueue: (workerId: string, queueId: string) =>
    request<Record<string, unknown>>(`/workers/${workerId}/queue/${queueId}/cancel`, { method: "POST" }),
  tools: () => request<{ tools: ToolView[] }>("/tools"),
  decidePermission: (permissionRequestId: string, decision: "allow_once" | "allow_for_session" | "deny") =>
    request<Record<string, unknown>>(`/permissions/${permissionRequestId}/decision`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    }),
  branchableNodes: (conversationId: string) =>
    request<{ nodes: BranchableNode[] }>(`/conversations/${conversationId}/branchable-nodes`),
  branch: (conversationId: string, coreNodeId: string) =>
    request<Record<string, unknown>>(`/conversations/${conversationId}/branch`, {
      method: "POST",
      body: JSON.stringify({ core_node_id: coreNodeId }),
    }),
  fork: (conversationId: string, coreNodeId: string, title?: string) =>
    request<{ conversation_id: string; core_session_id: string }>(`/conversations/${conversationId}/fork`, {
      method: "POST",
      body: JSON.stringify({ core_node_id: coreNodeId, title }),
    }),
  eventSource: (conversationId?: string | null) =>
    new EventSource(`${backendBaseUrl}/events${conversationId ? `?conversation_id=${encodeURIComponent(conversationId)}` : ""}`),
};
