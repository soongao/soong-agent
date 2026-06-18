import type { Message, WorkerView } from "../types";

export function localUserMessage(conversationId: string, text: string, workers: WorkerView[]): Message {
  const now = new Date().toISOString();
  const target = localMessageTarget(text, workers);
  return {
    message_id: `local_send_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    conversation_id: conversationId,
    sender_type: "user",
    sender_name: "You",
    target_type: target.targetType,
    target_id: target.targetId,
    original_text: text,
    display_text: target.displayText,
    status: "sending",
    metadata: {
      optimistic: true,
      ...(target.targetName ? { target_name: target.targetName } : {}),
    },
    created_at: now,
    updated_at: now,
  };
}

export function failedLocalMessage(message: Message, error: unknown): Message {
  return {
    ...message,
    status: "failed",
    metadata: {
      ...(message.metadata ?? {}),
      error: error instanceof Error ? error.message : "Failed to send message.",
    },
    updated_at: new Date().toISOString(),
  };
}

function localMessageTarget(text: string, workers: WorkerView[]) {
  const stripped = text.trim();
  if (!stripped.startsWith("@")) {
    return { targetType: "orchestrator", targetId: "orchestrator", targetName: "Orchestrator", displayText: text };
  }
  const [mention, ...rest] = stripped.split(/\s+/);
  const name = mention.slice(1);
  const displayText = rest.join(" ").trim() || text;
  if (name.toLowerCase() === "orchestrator") {
    return { targetType: "orchestrator", targetId: "orchestrator", targetName: "Orchestrator", displayText };
  }
  const worker = workers.find((item) => item.worker_id === name || item.name === name);
  return {
    targetType: "worker",
    targetId: worker?.worker_id ?? name,
    targetName: worker?.name ?? name,
    displayText,
  };
}
