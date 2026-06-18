import { Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { api } from "../../api/client";
import { useAppDispatch, useAppState } from "../../state/store";

export function ConversationList() {
  const { conversations, activeConversationId } = useAppState();
  const dispatch = useAppDispatch();
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function createConversation() {
    setCreating(true);
    setError(null);
    try {
      const conversation = await api.createConversation();
      dispatch({ type: "conversations", conversations: [conversation, ...conversations] });
      dispatch({ type: "activeConversation", conversationId: conversation.conversation_id });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create conversation.");
    } finally {
      setCreating(false);
    }
  }

  async function deleteConversation(conversationId: string) {
    setError(null);
    try {
      await api.deleteConversation(conversationId);
      const nextConversations = conversations.filter((conversation) => conversation.conversation_id !== conversationId);
      dispatch({ type: "conversations", conversations: nextConversations });
      if (activeConversationId === conversationId) {
        dispatch({ type: "activeConversation", conversationId: nextConversations[0]?.conversation_id ?? null });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete conversation.");
    }
  }

  return (
    <aside className="conversation-list">
      <div className="pane-header">
        <span>Conversations</span>
        <button className="icon-button" onClick={createConversation} aria-label="New conversation" disabled={creating}>
          <Plus size={16} />
        </button>
      </div>
      {error ? (
        <div className="pane-error" role="alert">
          {error}
        </div>
      ) : null}
      <div className="conversation-rows">
        {conversations.map((conversation) => (
          <div key={conversation.conversation_id} className={`conversation-row ${conversation.conversation_id === activeConversationId ? "active" : ""}`}>
            <button className="conversation-select" onClick={() => dispatch({ type: "activeConversation", conversationId: conversation.conversation_id })}>
              <span>{conversation.title}</span>
              <small>{conversation.last_message_preview}</small>
            </button>
            <button
              className="icon-button"
              onClick={() => deleteConversation(conversation.conversation_id)}
              aria-label={`Delete ${conversation.title}`}
              title="Delete conversation"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
