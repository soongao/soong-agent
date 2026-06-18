import { Send } from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api/client";
import { useAppDispatch, useAppState } from "../../state/store";
import type { Message } from "../../types";

type SlashOption = {
  name: string;
  usage: string;
  description: string;
  completion: string;
  kind: "command" | "skill";
};

const SLASH_COMMANDS: SlashOption[] = [
  { name: "help", usage: "/help", description: "Show available slash commands.", completion: "/help", kind: "command" },
  { name: "new", usage: "/new", description: "Create a new conversation.", completion: "/new", kind: "command" },
  { name: "clear", usage: "/clear", description: "Clear visible messages for this conversation.", completion: "/clear", kind: "command" },
  { name: "delete", usage: "/delete", description: "Delete the current conversation.", completion: "/delete", kind: "command" },
  { name: "cancel", usage: "/cancel", description: "Cancel the first queued or running message.", completion: "/cancel", kind: "command" },
  { name: "plan", usage: "/plan <goal>", description: "Create and write a plan for a goal.", completion: "/plan", kind: "command" },
  { name: "branch", usage: "/branch <node_id>", description: "Switch this conversation to a user node.", completion: "/branch", kind: "command" },
  { name: "fork", usage: "/fork <node_id>", description: "Fork this conversation from a user node.", completion: "/fork", kind: "command" },
  { name: "sessions", usage: "/sessions", description: "List conversations.", completion: "/sessions", kind: "command" },
  { name: "use", usage: "/use <conversation_id>", description: "Switch to a conversation.", completion: "/use", kind: "command" },
  { name: "workers", usage: "/workers", description: "List available workers.", completion: "/workers", kind: "command" },
  { name: "skills", usage: "/skills", description: "List available skills.", completion: "/skills", kind: "command" },
  { name: "config", usage: "/config", description: "Show backend, model, and context status.", completion: "/config", kind: "command" },
];

export function MentionInput() {
  const { activeConversationId, conversations, health, messagesByConversation, workers } = useAppState();
  const dispatch = useAppDispatch();
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const selectedMentionRef = useRef<HTMLButtonElement | null>(null);
  const selectedSlashRef = useRef<HTMLButtonElement | null>(null);
  const activeConversation = conversations.find((conversation) => conversation.conversation_id === activeConversationId);
  const showMentionMenu = text.startsWith("@") && !text.includes(" ");
  const showSlashMenu = text.startsWith("/") && !text.includes(" ");
  const options = useMemo(() => ["Orchestrator", ...workers.filter((worker) => worker.enabled && !worker.deleted_at).map((worker) => worker.worker_id)], [workers]);
  const slashOptions = useMemo(() => buildSlashOptions(text, health?.context?.skills ?? []), [health?.context?.skills, text]);

  useEffect(() => {
    selectedMentionRef.current?.scrollIntoView?.({ block: "nearest" });
    selectedSlashRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!activeConversationId || !text.trim()) return;
    setError(null);
    const trimmed = text.trim();
    if (trimmed.startsWith("@") && !trimmed.includes(" ")) {
      setError("Message text is required after a worker mention.");
      return;
    }
    if (trimmed.startsWith("/")) {
      await handleSlashCommand(trimmed);
      return;
    }
    try {
      await api.sendMessage(activeConversationId, text);
      const messages = await api.messages(activeConversationId);
      dispatch({ type: "messages", conversationId: activeConversationId, messages: messages.messages });
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message.");
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (!showMentionMenu && !showSlashMenu) return;
    const itemCount = showSlashMenu ? slashOptions.length : options.length;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelected((value) => (itemCount ? (value + 1) % itemCount : 0));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelected((value) => (itemCount ? (value - 1 + itemCount) % itemCount : 0));
    } else if (event.key === "Enter" || event.key === "Tab") {
      if (showSlashMenu) {
        const option = slashOptions[selected];
        if (option) {
          if (event.key === "Enter" && text.trim() === option.completion && option.kind === "command") {
            return;
          }
          event.preventDefault();
          setText(option.kind === "skill" ? `${option.completion} ` : option.completion);
        }
        return;
      }
      if (options[selected]) {
        event.preventDefault();
        setText(`@${options[selected]} `);
      }
    }
  }

  async function handleSlashCommand(trimmed: string) {
    const commandText = trimmed.slice(1);
    const [command = "", ...rest] = commandText.split(/\s+/);
    const argument = rest.join(" ").trim();
    const skillNames = new Set((health?.context?.skills ?? []).map((skill) => skill.name));
    try {
      if (command === "help") {
        addSystemMessage(slashHelpText(health?.context?.skills ?? []));
        setText("");
        return;
      }
      if (command === "new") {
        const conversation = await api.createConversation();
        dispatch({ type: "conversations", conversations: [conversation, ...conversations] });
        dispatch({ type: "activeConversation", conversationId: conversation.conversation_id });
        setText("");
        return;
      }
      if (command === "clear") {
        dispatch({ type: "messages", conversationId: activeConversationId!, messages: [] });
        setText("");
        return;
      }
      if (command === "delete") {
        await api.deleteConversation(activeConversationId!);
        const next = conversations.filter((conversation) => conversation.conversation_id !== activeConversationId);
        dispatch({ type: "conversations", conversations: next });
        dispatch({ type: "activeConversation", conversationId: next[0]?.conversation_id ?? null });
        setText("");
        return;
      }
      if (command === "cancel") {
        const activeMessage = (messagesByConversation[activeConversationId!] ?? []).find(
          (message) => message.sender_type !== "user" && ["queued", "running"].includes(message.status) && (message.core_run_id || message.queue_id),
        );
        if (!activeMessage) {
          addSystemMessage("No queued or running message to cancel.");
        } else {
          await api.cancelConversation(activeConversationId!, { core_run_id: activeMessage.core_run_id, queue_id: activeMessage.queue_id });
          const messages = await api.messages(activeConversationId!);
          dispatch({ type: "messages", conversationId: activeConversationId!, messages: messages.messages });
        }
        setText("");
        return;
      }
      if (command === "plan") {
        if (!argument) {
          setError("usage: /plan <goal>");
          return;
        }
        await api.sendMessage(activeConversationId!, planRequestMessage(argument));
        const messages = await api.messages(activeConversationId!);
        dispatch({ type: "messages", conversationId: activeConversationId!, messages: messages.messages });
        setText("");
        return;
      }
      if (command === "branch") {
        if (!argument) {
          addSystemMessage(await branchableNodesText());
          setText("");
          return;
        }
        await api.branch(activeConversationId!, argument);
        const messages = await api.messages(activeConversationId!);
        dispatch({ type: "messages", conversationId: activeConversationId!, messages: messages.messages });
        setText("");
        return;
      }
      if (command === "fork") {
        if (!argument) {
          addSystemMessage(await branchableNodesText());
          setText("");
          return;
        }
        const forked = await api.fork(activeConversationId!, argument);
        const response = await api.conversations();
        dispatch({ type: "conversations", conversations: response.conversations });
        dispatch({ type: "activeConversation", conversationId: forked.conversation_id });
        setText("");
        return;
      }
      if (command === "sessions") {
        addSystemMessage(
          conversations.length
            ? conversations.map((conversation) => `${conversation.conversation_id} - ${conversation.title}`).join("\n")
            : "No conversations found.",
        );
        setText("");
        return;
      }
      if (command === "use") {
        if (!argument) {
          setError("Conversation id is required after /use.");
          return;
        }
        const conversation = conversations.find((item) => item.conversation_id === argument);
        if (!conversation) {
          setError(`Conversation not found: ${argument}`);
          return;
        }
        dispatch({ type: "activeConversation", conversationId: conversation.conversation_id });
        setText("");
        return;
      }
      if (command === "workers") {
        addSystemMessage(workers.length ? workers.map((worker) => `@${worker.worker_id} - ${worker.description || worker.name}`).join("\n") : "No workers found.");
        setText("");
        return;
      }
      if (command === "skills") {
        const skills = health?.context?.skills ?? [];
        addSystemMessage(skills.length ? skills.map((skill) => `/${skill.name} - ${skill.description || "No description."}`).join("\n") : "No skills found.");
        setText("");
        return;
      }
      if (command === "config") {
        addSystemMessage(configText());
        setText("");
        return;
      }
      if (skillNames.has(command)) {
        if (!argument) {
          setError(`Message text is required after /${command}.`);
          return;
        }
        await api.loadSkill(activeConversationId!, command);
        await api.sendMessage(activeConversationId!, argument);
        const messages = await api.messages(activeConversationId!);
        dispatch({ type: "messages", conversationId: activeConversationId!, messages: messages.messages });
        setText("");
        return;
      }
      setError(`Unknown slash command: /${command}. Use /help for commands.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Slash command failed.");
    }
  }

  async function branchableNodesText() {
    if (!activeConversationId) return "No active conversation.";
    const response = await api.branchableNodes(activeConversationId);
    if (!response.nodes.length) return "No user message nodes yet.";
    return ["User message nodes", ...response.nodes.map((node) => `${node.core_node_id}${node.active ? " *" : ""} - ${node.preview}`)].join("\n");
  }

  function addSystemMessage(displayText: string) {
    if (!activeConversationId) return;
    dispatch({
      type: "localMessage",
      conversationId: activeConversationId,
      message: localSystemMessage(activeConversationId, displayText),
    });
  }

  function configText() {
    const instructions = health?.context?.auto_instruction_paths ?? [];
    return [
      `conversation: ${activeConversation?.title ?? activeConversationId}`,
      `backend: ${health?.base_url ?? "unknown"}`,
      `model: ${health?.provider ?? "unknown"} / ${health?.model ?? "unknown"}`,
      `CLAUDE.md: ${instructions.length ? instructions.join(", ") : "none"}`,
      `skills: ${health?.context?.skill_count ?? 0}`,
    ].join("\n");
  }

  return (
    <form className="mention-input" onSubmit={submit}>
      {error ? <div className="input-error">{error}</div> : null}
      {showSlashMenu ? (
        <div className="slash-menu">
          {slashOptions.map((option, index) => (
            <button
              type="button"
              key={`${option.kind}:${option.name}`}
              ref={index === selected ? selectedSlashRef : undefined}
              className={index === selected ? "selected" : ""}
              onMouseDown={(event) => {
                event.preventDefault();
                setText(option.kind === "skill" ? `${option.completion} ` : option.completion);
              }}
            >
              <span>{option.usage}</span>
              <small>{option.description}</small>
            </button>
          ))}
        </div>
      ) : null}
      {showMentionMenu ? (
        <div className="mention-menu">
          {options.map((option, index) => (
            <button
              type="button"
              key={option}
              ref={index === selected ? selectedMentionRef : undefined}
              className={index === selected ? "selected" : ""}
              onMouseDown={(event) => {
                event.preventDefault();
                setText(`@${option} `);
              }}
            >
              @{option}
            </button>
          ))}
        </div>
      ) : null}
      <input
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          setSelected(0);
          setError(null);
        }}
        onKeyDown={handleKeyDown}
        placeholder={activeConversationId ? "Message, @worker, or /help" : "Create a conversation first"}
        disabled={!activeConversationId}
      />
      <button className="send-button" disabled={!activeConversationId || !text.trim()} aria-label="Send">
        <Send size={16} />
      </button>
    </form>
  );
}

function planRequestMessage(goal: string) {
  return `Create a plan for: ${goal}. Use agent.plan_template, then write the plan Markdown to the suggested project plan directory.`;
}

function buildSlashOptions(text: string, skills: { name: string; description?: string }[]): SlashOption[] {
  const commandText = text.startsWith("/") ? text.slice(1).toLowerCase() : "";
  const commandNames = new Set(SLASH_COMMANDS.map((command) => command.name));
  const skillOptions = skills
    .filter((skill) => !commandNames.has(skill.name))
    .map((skill) => ({
      name: skill.name,
      usage: `/${skill.name} <message>`,
      description: skill.description || "Load this skill and send a message.",
      completion: `/${skill.name}`,
      kind: "skill" as const,
    }));
  return [...SLASH_COMMANDS, ...skillOptions].filter((option) => option.name.toLowerCase().startsWith(commandText));
}

function slashHelpText(skills: { name: string; description?: string }[]) {
  const lines = ["Slash commands", ...SLASH_COMMANDS.map((command) => `${command.usage} - ${command.description}`)];
  if (skills.length) {
    lines.push("", "Skills");
    lines.push(...skills.map((skill) => `/${skill.name} <message> - ${skill.description || "Load this skill and send a message."}`));
  }
  return lines.join("\n");
}

function localSystemMessage(conversationId: string, displayText: string): Message {
  const now = new Date().toISOString();
  return {
    message_id: `local_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    conversation_id: conversationId,
    sender_type: "system",
    sender_name: "Agent Hub",
    target_type: "none",
    original_text: "",
    display_text: displayText,
    status: "completed",
    created_at: now,
    updated_at: now,
  };
}
