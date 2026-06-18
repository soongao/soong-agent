import React, { createContext, useContext, useMemo, useReducer } from "react";
import type { Conversation, HealthStatus, HubEvent, Message, PermissionRequest, ToolView, WorkerView } from "../types";

type AppState = {
  health: HealthStatus | null;
  backendBaseUrl: string;
  conversations: Conversation[];
  activeConversationId: string | null;
  messagesByConversation: Record<string, Message[]>;
  conversationWorkersByConversation: Record<string, WorkerView[]>;
  workers: WorkerView[];
  tools: ToolView[];
  permissionsByConversation: Record<string, PermissionRequest[]>;
  eventConnection: "closed" | "connecting" | "open" | "error";
};

type Action =
  | { type: "health"; health: HealthStatus }
  | { type: "backendBaseUrl"; backendBaseUrl: string }
  | { type: "conversations"; conversations: Conversation[] }
  | { type: "activeConversation"; conversationId: string | null }
  | { type: "messages"; conversationId: string; messages: Message[] }
  | { type: "conversationWorkers"; conversationId: string; workers: WorkerView[] }
  | { type: "workers"; workers: WorkerView[] }
  | { type: "tools"; tools: ToolView[] }
  | { type: "eventConnection"; status: AppState["eventConnection"] }
  | { type: "localMessage"; conversationId: string; message: Message }
  | { type: "hubEvent"; event: HubEvent };

const initialState: AppState = {
  health: null,
  backendBaseUrl: "",
  conversations: [],
  activeConversationId: null,
  messagesByConversation: {},
  conversationWorkersByConversation: {},
  workers: [],
  tools: [],
  permissionsByConversation: {},
  eventConnection: "closed",
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "health":
      return { ...state, health: action.health };
    case "backendBaseUrl":
      return { ...state, backendBaseUrl: action.backendBaseUrl };
    case "conversations":
      return {
        ...state,
        conversations: action.conversations,
        activeConversationId: state.activeConversationId ?? action.conversations[0]?.conversation_id ?? null,
      };
    case "activeConversation":
      return { ...state, activeConversationId: action.conversationId };
    case "messages":
      return {
        ...state,
        messagesByConversation: { ...state.messagesByConversation, [action.conversationId]: action.messages },
      };
    case "conversationWorkers":
      return {
        ...state,
        conversationWorkersByConversation: { ...state.conversationWorkersByConversation, [action.conversationId]: action.workers },
      };
    case "workers":
      return { ...state, workers: action.workers };
    case "tools":
      return { ...state, tools: action.tools };
    case "eventConnection":
      return { ...state, eventConnection: action.status };
    case "localMessage": {
      const current = state.messagesByConversation[action.conversationId] ?? [];
      return {
        ...state,
        messagesByConversation: {
          ...state.messagesByConversation,
          [action.conversationId]: sortMessages(mergeById(current, action.message, "message_id")),
        },
      };
    }
    case "hubEvent":
      return applyHubEvent(state, action.event);
    default:
      return state;
  }
}

function applyHubEvent(state: AppState, event: HubEvent): AppState {
  const payload = event.payload as Record<string, unknown>;
  if (event.type === "conversation_created" && payload.conversation_id) {
    const conversation = payload as unknown as Conversation;
    return { ...state, conversations: mergeById(state.conversations, conversation, "conversation_id") };
  }
  if (event.type === "conversation_deleted" && event.conversation_id) {
    return removeConversation(state, event.conversation_id);
  }
  if (event.type === "conversation_updated" && event.conversation_id) {
    const current = state.conversations.find((conversation) => conversation.conversation_id === event.conversation_id);
    if (!current) return state;
    const conversation = { ...current, ...payload } as Conversation;
    return { ...state, conversations: mergeById(state.conversations, conversation, "conversation_id") };
  }
  if (
    (
      event.type === "message_created" ||
      event.type === "message_updated" ||
      event.type === "run_started" ||
      event.type === "run_completed" ||
      event.type === "worker_queued" ||
      event.type === "worker_started" ||
      event.type === "worker_completed" ||
      event.type === "worker_failed" ||
      event.type === "worker_cancelled" ||
      event.type === "permission_failed"
    ) &&
    event.conversation_id
  ) {
    const message = (payload.message ?? payload) as Message;
    if (!message?.message_id) return state;
    const current = state.messagesByConversation[event.conversation_id] ?? [];
    return {
      ...state,
      messagesByConversation: {
        ...state.messagesByConversation,
        [event.conversation_id]: sortMessages(upsertMessage(current, message)),
      },
    };
  }
  if (event.type === "message_delta" && event.conversation_id) {
    const message = payload.message as Message | undefined;
    if (!message?.message_id) return state;
    const current = state.messagesByConversation[event.conversation_id] ?? [];
    return {
      ...state,
      messagesByConversation: {
        ...state.messagesByConversation,
        [event.conversation_id]: appendMessageDelta(current, message, String(payload.delta ?? "")),
      },
    };
  }
  if (event.type === "run_failed" && event.conversation_id) {
    const message = (payload.message ?? payload) as Message;
    if (!message?.message_id) return state;
    const current = state.messagesByConversation[event.conversation_id] ?? [];
    return {
      ...state,
      messagesByConversation: {
        ...state.messagesByConversation,
        [event.conversation_id]: sortMessages(mergeById(current, message, "message_id")),
      },
    };
  }
  if (event.type === "permission_requested" && event.conversation_id) {
    const request = payload as unknown as PermissionRequest;
    if (!request.permission_request_id) return state;
    const current = state.permissionsByConversation[event.conversation_id] ?? [];
    return {
      ...state,
      permissionsByConversation: {
        ...state.permissionsByConversation,
        [event.conversation_id]: mergeById(current, request, "permission_request_id"),
      },
    };
  }
  if (event.type === "permission_resolved" && event.conversation_id) {
    const permissionRequestId = String(payload.permission_request_id ?? "");
    const current = state.permissionsByConversation[event.conversation_id] ?? [];
    return {
      ...state,
      permissionsByConversation: {
        ...state.permissionsByConversation,
        [event.conversation_id]: current.filter((request) => request.permission_request_id !== permissionRequestId),
      },
    };
  }
  return state;
}

function removeConversation(state: AppState, conversationId: string): AppState {
  const conversations = state.conversations.filter((conversation) => conversation.conversation_id !== conversationId);
  const { [conversationId]: _messages, ...messagesByConversation } = state.messagesByConversation;
  const { [conversationId]: _conversationWorkers, ...conversationWorkersByConversation } = state.conversationWorkersByConversation;
  const { [conversationId]: _permissions, ...permissionsByConversation } = state.permissionsByConversation;
  return {
    ...state,
    conversations,
    messagesByConversation,
    conversationWorkersByConversation,
    permissionsByConversation,
    activeConversationId: state.activeConversationId === conversationId ? conversations[0]?.conversation_id ?? null : state.activeConversationId,
  };
}

function mergeById<T extends Record<string, unknown>>(items: T[], item: T, key: keyof T): T[] {
  const index = items.findIndex((existing) => existing[key] === item[key]);
  if (index === -1) return [...items, item];
  return items.map((existing, i) => (i === index ? { ...existing, ...item } : existing));
}

function appendMessageDelta(messages: Message[], message: Message, delta: string): Message[] {
  const index = messages.findIndex((existing) => existing.message_id === message.message_id);
  if (index === -1) return [...messages, message];
  return messages.map((existing, i) => {
    if (i !== index) return existing;
    const nextText = message.display_text || (delta ? `${existing.display_text ?? ""}${delta}` : existing.display_text);
    return { ...existing, ...message, display_text: nextText };
  });
}

function upsertMessage(messages: Message[], message: Message): Message[] {
  const withoutOptimistic = message.sender_type === "user" ? removeMatchingOptimisticUserMessage(messages, message) : messages;
  return mergeById(withoutOptimistic, message, "message_id");
}

function removeMatchingOptimisticUserMessage(messages: Message[], message: Message): Message[] {
  return messages.filter((existing) => {
    if (existing.message_id === message.message_id) return true;
    if (existing.sender_type !== "user" || existing.metadata?.optimistic !== true) return true;
    const sameText = existing.original_text === message.original_text || existing.display_text === message.display_text;
    const sameTarget = !message.target_id || !existing.target_id || existing.target_id === message.target_id;
    return !(sameText && sameTarget);
  });
}

function sortMessages(messages: Message[]): Message[] {
  const groupStart = new Map<string, string>();
  for (const message of messages) {
    const group = messageGroupKey(message);
    const current = groupStart.get(group);
    if (!current || message.created_at < current) groupStart.set(group, message.created_at);
  }
  return [...messages].sort((left, right) => {
    const leftGroup = groupStart.get(messageGroupKey(left)) ?? left.created_at;
    const rightGroup = groupStart.get(messageGroupKey(right)) ?? right.created_at;
    return (
      leftGroup.localeCompare(rightGroup) ||
      senderRank(left.sender_type) - senderRank(right.sender_type) ||
      messageSortTime(left).localeCompare(messageSortTime(right)) ||
      left.message_id.localeCompare(right.message_id)
    );
  });
}

function messageGroupKey(message: Message): string {
  return message.core_run_id ? `run:${message.core_run_id}` : `message:${message.message_id}`;
}

function messageSortTime(message: Message): string {
  return message.sender_type === "user" ? message.created_at : message.updated_at;
}

function senderRank(senderType: string): number {
  return { user: 0, worker: 1, orchestrator: 2, system: 3 }[senderType] ?? 4;
}

const AppStateContext = createContext<AppState | null>(null);
const DispatchContext = createContext<React.Dispatch<Action> | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const memoState = useMemo(() => state, [state]);
  return (
    <AppStateContext.Provider value={memoState}>
      <DispatchContext.Provider value={dispatch}>{children}</DispatchContext.Provider>
    </AppStateContext.Provider>
  );
}

export function useAppState(): AppState {
  const state = useContext(AppStateContext);
  if (!state) throw new Error("useAppState must be used inside AppStateProvider");
  return state;
}

export function useAppDispatch(): React.Dispatch<Action> {
  const dispatch = useContext(DispatchContext);
  if (!dispatch) throw new Error("useAppDispatch must be used inside AppStateProvider");
  return dispatch;
}
