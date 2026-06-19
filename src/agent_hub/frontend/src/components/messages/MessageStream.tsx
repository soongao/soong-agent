import { GitBranch, GitFork, Reply, Send, Square, X } from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api/client";
import { failedLocalMessage, localUserMessage } from "../../messages/localMessages";
import { useAppDispatch, useAppState } from "../../state/store";
import type { BranchableNode, Message, PermissionRequest } from "../../types";

export function MessageStream() {
  const { activeConversationId, conversationWorkersByConversation, messagesByConversation, permissionsByConversation } = useAppState();
  const dispatch = useAppDispatch();
  const messages = activeConversationId ? messagesByConversation[activeConversationId] ?? [] : [];
  const permissions = activeConversationId ? permissionsByConversation[activeConversationId] ?? [] : [];
  const conversationWorkers = activeConversationId ? conversationWorkersByConversation[activeConversationId] ?? [] : [];
  const streamEndRef = useRef<HTMLDivElement | null>(null);
  const scrollKey = useMemo(
    () => messages.map((message) => `${message.message_id}:${message.updated_at}:${message.display_text.length}:${message.status}`).join("|"),
    [messages],
  );
  const [branchNodes, setBranchNodes] = useState<BranchableNode[]>([]);
  const [branchMode, setBranchMode] = useState<"branch" | "fork">("branch");
  const [showBranchPicker, setShowBranchPicker] = useState(false);
  const [selectedNode, setSelectedNode] = useState(0);
  const [replyingToMessageId, setReplyingToMessageId] = useState<string | null>(null);
  const [replyText, setReplyText] = useState("");
  const [replyError, setReplyError] = useState<string | null>(null);
  const replyInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setShowBranchPicker(false);
    setBranchNodes([]);
    setSelectedNode(0);
    setReplyingToMessageId(null);
    setReplyText("");
    setReplyError(null);
  }, [activeConversationId]);

  useEffect(() => {
    streamEndRef.current?.scrollIntoView?.({ block: "end" });
  }, [activeConversationId, permissions.length, scrollKey]);

  useEffect(() => {
    if (!replyingToMessageId) return;
    requestAnimationFrame(() => replyInputRef.current?.focus());
  }, [replyingToMessageId]);

  async function refreshMessages() {
    if (!activeConversationId) return;
    const response = await api.messages(activeConversationId);
    dispatch({ type: "messages", conversationId: activeConversationId, messages: response.messages });
  }

  async function cancelMessage(messageCoreRunId?: string | null, queueId?: string | null) {
    if (!activeConversationId || (!messageCoreRunId && !queueId)) return;
    await api.cancelConversation(activeConversationId, { core_run_id: messageCoreRunId, queue_id: queueId });
    await refreshMessages();
  }

  async function branchFrom(nodeId: string) {
    if (!activeConversationId) return;
    await api.branch(activeConversationId, nodeId);
    await refreshMessages();
  }

  async function forkFrom(nodeId: string) {
    if (!activeConversationId) return;
    const forked = await api.fork(activeConversationId, nodeId);
    const conversations = await api.conversations();
    dispatch({ type: "conversations", conversations: conversations.conversations });
    dispatch({ type: "activeConversation", conversationId: forked.conversation_id });
    const messages = await api.messages(forked.conversation_id);
    dispatch({ type: "messages", conversationId: forked.conversation_id, messages: messages.messages });
  }

  async function openNodePicker(mode: "branch" | "fork") {
    if (!activeConversationId) return;
    const response = await api.branchableNodes(activeConversationId);
    setBranchNodes(response.nodes);
    setBranchMode(mode);
    setSelectedNode(0);
    setShowBranchPicker(true);
  }

  async function applySelectedNode() {
    const node = branchNodes[selectedNode];
    if (!node) return;
    setShowBranchPicker(false);
    if (branchMode === "branch") {
      await branchFrom(node.core_node_id);
      return;
    }
    await forkFrom(node.core_node_id);
  }

  async function handlePickerKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelectedNode((value) => Math.min(branchNodes.length - 1, value + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelectedNode((value) => Math.max(0, value - 1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      await applySelectedNode();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setShowBranchPicker(false);
    }
  }

  function openWorkerReply(message: Message) {
    setReplyingToMessageId(message.message_id);
    setReplyText("");
    setReplyError(null);
  }

  async function sendWorkerReply(event: FormEvent, replyTarget: WorkerReplyTarget) {
    event.preventDefault();
    if (!activeConversationId || !replyText.trim()) return;
    const outgoingText = `@${replyTarget.workerId} ${replyText.trim()}`;
    const pendingMessage = localUserMessage(activeConversationId, outgoingText, conversationWorkers);
    try {
      setReplyError(null);
      dispatch({ type: "localMessage", conversationId: activeConversationId, message: pendingMessage });
      setReplyText("");
      setReplyingToMessageId(null);
      await api.sendMessage(activeConversationId, outgoingText);
    } catch (err) {
      dispatch({ type: "localMessage", conversationId: activeConversationId, message: failedLocalMessage(pendingMessage, err) });
      setReplyError(err instanceof Error ? err.message : "Failed to send reply.");
    }
  }

  return (
    <div className="message-stream">
      <div className="stream-toolbar">
        <button disabled={!activeConversationId} onClick={() => openNodePicker("branch")}>
          <GitBranch size={14} />
          Branch
        </button>
        <button disabled={!activeConversationId} onClick={() => openNodePicker("fork")}>
          <GitFork size={14} />
          Fork
        </button>
      </div>
      {showBranchPicker ? (
        <div
          className="branch-picker"
          role="dialog"
          aria-label={branchMode === "branch" ? "Branch from user message" : "Fork from user message"}
          aria-modal="false"
          tabIndex={-1}
          onKeyDown={handlePickerKeyDown}
        >
          <div className="branch-picker-header">
            <strong>{branchMode === "branch" ? "Branch from user message" : "Fork from user message"}</strong>
            <button onClick={() => setShowBranchPicker(false)}>Close</button>
          </div>
          <div className="branch-node-list">
            {branchNodes.map((node, index) => (
              <button
                key={node.core_node_id}
                className={index === selectedNode ? "selected" : ""}
                aria-selected={index === selectedNode}
                onClick={() => setSelectedNode(index)}
                onDoubleClick={applySelectedNode}
              >
                <span>{node.core_node_id}</span>
                <small>{node.preview}</small>
              </button>
            ))}
            {branchNodes.length === 0 ? <small>No user message nodes yet.</small> : null}
          </div>
          <div className="branch-picker-actions">
            <button
              disabled={selectedNode <= 0}
              onClick={() => setSelectedNode((value) => Math.max(0, value - 1))}
            >
              Up
            </button>
            <button
              disabled={selectedNode >= branchNodes.length - 1}
              onClick={() => setSelectedNode((value) => Math.min(branchNodes.length - 1, value + 1))}
            >
              Down
            </button>
            <button disabled={branchNodes.length === 0} onClick={applySelectedNode}>
              {branchMode === "branch" ? "Branch" : "Fork"}
            </button>
          </div>
        </div>
      ) : null}
      {messages.length === 0 ? <div className="empty-state">No messages yet.</div> : null}
      {messages.map((message) => {
        const replyTarget = workerReplyTarget(message);
        return (
          <article key={message.message_id} className={`message-bubble ${message.sender_type}`}>
            <div className="message-rail" aria-hidden="true" />
            <div className="message-meta">
              <span>{messageTitle(message)}</span>
              <span>{message.status}</span>
            </div>
            <p className={messageText(message) ? undefined : "message-placeholder"}>{messageText(message) || placeholderText(message)}</p>
            {message.sender_type !== "user" && ["queued", "running"].includes(message.status) ? (
              <div className="message-actions">
                <button onClick={() => cancelMessage(message.core_run_id, message.queue_id)} title="Cancel">
                  <Square size={14} />
                </button>
              </div>
            ) : null}
            {replyTarget ? (
              <div className="message-actions">
                <button
                  type="button"
                  className="reply-action"
                  onClick={() => openWorkerReply(message)}
                  title={`Reply to ${replyTarget.name}`}
                >
                  <Reply size={14} />
                  <span>Reply</span>
                </button>
              </div>
            ) : null}
            {replyingToMessageId === message.message_id && replyTarget ? (
              <form className="inline-reply" onSubmit={(event) => sendWorkerReply(event, replyTarget)}>
                {replyError ? <div className="inline-reply-error">{replyError}</div> : null}
                <div className="inline-reply-header">
                  <span>Reply to {replyTarget.name}</span>
                  <button type="button" onClick={() => setReplyingToMessageId(null)} aria-label="Cancel reply">
                    <X size={14} />
                  </button>
                </div>
                <div className="inline-reply-row">
                  <input
                    ref={replyInputRef}
                    value={replyText}
                    onChange={(event) => {
                      setReplyText(event.target.value);
                      setReplyError(null);
                    }}
                    placeholder={`Message ${replyTarget.name}`}
                  />
                  <button type="submit" disabled={!replyText.trim()} aria-label={`Send reply to ${replyTarget.name}`}>
                    <Send size={14} />
                  </button>
                </div>
              </form>
            ) : null}
            {message.sender_type === "user" && message.core_node_id ? (
              <div className="message-actions">
                <button onClick={() => branchFrom(message.core_node_id!)} title="Branch from here">
                  <GitBranch size={14} />
                </button>
                <button onClick={() => forkFrom(message.core_node_id!)} title="Fork conversation">
                  <GitFork size={14} />
                </button>
              </div>
            ) : null}
          </article>
        );
      })}
      {permissions.map((permission) => (
        <PermissionCard key={permission.permission_request_id} permission={permission} />
      ))}
      <div ref={streamEndRef} aria-hidden="true" />
    </div>
  );
}

function messageText(message: { display_text: string; original_text: string }) {
  return message.display_text || message.original_text;
}

function placeholderText(message: { sender_type: string; status: string }) {
  if (message.sender_type !== "user" && ["queued", "running"].includes(message.status)) return "Generating...";
  if (message.status === "sending") return "Sending...";
  return "No content.";
}

function messageTitle(message: { sender_name: string; sender_type: string; target_type?: string | null; metadata?: Record<string, unknown> }) {
  if (message.sender_type === "user" && message.target_type === "worker") {
    const targetName = typeof message.metadata?.target_name === "string" ? message.metadata.target_name : undefined;
    return `You -> ${targetName || "Worker"}`;
  }
  if (message.sender_type === "orchestrator") {
    const targetWorkerName = typeof message.metadata?.target_worker_name === "string" ? message.metadata.target_worker_name : undefined;
    if (targetWorkerName) {
      return `Orchestrator -> ${targetWorkerName}`;
    }
  }
  return message.sender_name;
}

type WorkerReplyTarget = {
  workerId: string;
  name: string;
};

function workerReplyTarget(message: Message): WorkerReplyTarget | null {
  if (message.sender_type === "user" || ["queued", "running"].includes(message.status)) return null;
  const metadata = message.metadata ?? {};
  const snapshot = metadata.worker_snapshot && typeof metadata.worker_snapshot === "object" && !Array.isArray(metadata.worker_snapshot)
    ? (metadata.worker_snapshot as Record<string, unknown>)
    : {};
  const workerId = firstString(message.worker_id, metadata.worker_id, metadata.target_worker_id, snapshot.worker_id);
  if (!workerId) return null;
  const name = firstString(
    message.sender_type === "worker" && message.sender_name !== "Worker" ? message.sender_name : undefined,
    metadata.target_worker_name,
    snapshot.name,
    workerId,
  );
  return { workerId, name: name ?? workerId };
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

function PermissionCard({ permission }: { permission: PermissionRequest }) {
  async function decide(decision: "allow_once" | "allow_for_session" | "deny") {
    await api.decidePermission(permission.permission_request_id, decision);
  }

  return (
    <article className="permission-card">
      <div>
        <strong>{permission.tool_name}</strong>
        <small>
          {permission.permission}
          {permission.target_scope ? ` · ${permission.target_scope}` : ""}
        </small>
      </div>
      <p>{permission.args_summary}</p>
      <div className="permission-actions">
        <button onClick={() => decide("allow_once")}>Allow Once</button>
        <button onClick={() => decide("allow_for_session")}>Allow Session</button>
        <button onClick={() => decide("deny")}>Deny</button>
      </div>
    </article>
  );
}
