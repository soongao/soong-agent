import { GitBranch, GitFork, Square } from "lucide-react";
import { KeyboardEvent, useEffect, useState } from "react";
import { api } from "../../api/client";
import { useAppDispatch, useAppState } from "../../state/store";
import type { BranchableNode, PermissionRequest } from "../../types";

export function MessageStream() {
  const { activeConversationId, messagesByConversation, permissionsByConversation } = useAppState();
  const dispatch = useAppDispatch();
  const messages = activeConversationId ? messagesByConversation[activeConversationId] ?? [] : [];
  const permissions = activeConversationId ? permissionsByConversation[activeConversationId] ?? [] : [];
  const [branchNodes, setBranchNodes] = useState<BranchableNode[]>([]);
  const [branchMode, setBranchMode] = useState<"branch" | "fork">("branch");
  const [showBranchPicker, setShowBranchPicker] = useState(false);
  const [selectedNode, setSelectedNode] = useState(0);

  useEffect(() => {
    setShowBranchPicker(false);
    setBranchNodes([]);
    setSelectedNode(0);
  }, [activeConversationId]);

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
      {messages.map((message) => (
        <article key={message.message_id} className={`message-bubble ${message.sender_type}`}>
          <div className="message-meta">
            <span>{messageTitle(message)}</span>
            <span>{message.status}</span>
          </div>
          <p>{message.display_text || message.original_text}</p>
          {message.sender_type !== "user" && ["queued", "running"].includes(message.status) ? (
            <div className="message-actions">
              <button onClick={() => cancelMessage(message.core_run_id, message.queue_id)} title="Cancel">
                <Square size={14} />
              </button>
            </div>
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
      ))}
      {permissions.map((permission) => (
        <PermissionCard key={permission.permission_request_id} permission={permission} />
      ))}
    </div>
  );
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
