import { useEffect } from "react";
import { api, backendBaseUrl } from "./api/client";
import { ConversationList } from "./components/conversations/ConversationList";
import { MessageStream } from "./components/messages/MessageStream";
import { MentionInput } from "./components/input/MentionInput";
import { WorkerPanel } from "./components/workers/WorkerPanel";
import { HealthBanner } from "./components/layout/HealthBanner";
import { useAppDispatch, useAppState } from "./state/store";

export function App() {
  const state = useAppState();
  const dispatch = useAppDispatch();
  const activeConversationId = state.activeConversationId;

  useEffect(() => {
    void Promise.all([api.health(), api.conversations(), api.workers(), api.tools()]).then(([health, conversations, workers, tools]) => {
      dispatch({ type: "health", health });
      dispatch({ type: "backendBaseUrl", backendBaseUrl });
      dispatch({ type: "conversations", conversations: conversations.conversations });
      dispatch({ type: "workers", workers: workers.workers });
      dispatch({ type: "tools", tools: tools.tools });
    });
  }, [dispatch]);

  useEffect(() => {
    if (!activeConversationId) return;
    void api.messages(activeConversationId).then((response) =>
      dispatch({ type: "messages", conversationId: activeConversationId, messages: response.messages }),
    );
    dispatch({ type: "eventConnection", status: "connecting" });
    const events = api.eventSource(activeConversationId);
    events.onopen = () => dispatch({ type: "eventConnection", status: "open" });
    events.onerror = () => dispatch({ type: "eventConnection", status: "error" });
    const handle = (event: MessageEvent) => {
      dispatch({ type: "hubEvent", event: JSON.parse(event.data) });
    };
    [
      "conversation_created",
      "conversation_deleted",
      "conversation_updated",
      "message_created",
      "message_delta",
      "message_updated",
      "run_started",
      "run_completed",
      "run_failed",
      "run_cancelled",
      "worker_queued",
      "worker_started",
      "worker_completed",
      "worker_failed",
      "worker_cancelled",
      "permission_requested",
      "permission_resolved",
      "permission_failed",
    ].forEach((type) => events.addEventListener(type, handle));
    return () => events.close();
  }, [activeConversationId, dispatch]);

  return (
    <div className="app-shell">
      <HealthBanner />
      <main className="workspace">
        <ConversationList />
        <section className="chat-pane">
          <MessageStream />
          <MentionInput />
        </section>
        <WorkerPanel />
      </main>
    </div>
  );
}
