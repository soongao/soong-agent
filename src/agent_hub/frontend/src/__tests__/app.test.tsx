import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { AppStateProvider } from "../state/store";
import type { WorkerView } from "../types";

const mockHealth = {
  ok: true,
  status: "ready",
  config_path: "/tmp/config.toml",
  provider: "openai",
  model: "qwen2.5:7b",
  base_url: "http://127.0.0.1:11434/v1",
  core_started: true,
  hub_db_path: "/tmp/hub.db",
  project_dir: "/tmp/project",
  context: {
    auto_instruction_paths: ["/tmp/project/CLAUDE.md"],
    skill_count: 1,
    skills: [{ name: "brainstorming", description: "Think first" }],
  },
  warnings: [],
};

const mockConversation = {
  conversation_id: "conv_1",
  core_session_id: "sess_1",
  title: "Test conversation",
  status: "active",
  active_core_node_id: null,
  last_message_preview: "hello",
  created_at: "2026-06-17T00:00:00Z",
  updated_at: "2026-06-17T00:00:00Z",
};

const secondConversation = {
  ...mockConversation,
  conversation_id: "conv_2",
  core_session_id: "sess_2",
  title: "Second conversation",
  last_message_preview: "second",
};

const forkedConversation = {
  ...mockConversation,
  conversation_id: "conv_fork",
  core_session_id: "sess_fork",
  title: "Fork from node_user",
  last_message_preview: "forked",
};

const mockWorker: WorkerView = {
  worker_id: "reviewer",
  worker_pool_id: "default",
  agent_definition_id: "worker.reviewer",
  name: "Reviewer",
  description: "Reviews code.",
  system_prompt: "Review code.",
  allowed_tools: ["code.read_file"],
  enabled: true,
  deleted_at: null,
  source: "dynamic",
  status: "idle",
  queue_length: 0,
  current_task_id: null,
  current_run_id: null,
  current_step_id: null,
};

const mockExternalWorker: WorkerView = {
  ...mockWorker,
  worker_id: "opencode_worker",
  agent_definition_id: "worker.opencode",
  name: "OpenCode Worker",
  description: "Uses an external executor.",
  system_prompt: "External worker.",
  allowed_tools: ["opencode.acp"],
  metadata: {
    worker_executor: {
      type: "opencode",
      config: { binary: "opencode", cwd: "/tmp/project", args: ["--pure"] },
    },
  },
};

let responses: Record<string, unknown>;

function defaultResponses(): Record<string, unknown> {
  return {
    "/health": mockHealth,
    "/conversations": { conversations: [mockConversation, secondConversation] },
    "/workers": { workers: [mockWorker] },
    "/conversations/conv_1/workers": { workers: [mockWorker] },
    "/conversations/conv_2/workers": { workers: [] },
    "/conversations/conv_new/workers": { workers: [] },
    "/conversations/conv_fork/workers": { workers: [] },
    "/tools": {
      tools: [
        {
          name: "code.read_file",
          description: "Read a file.",
          permission: "readonly",
          tags: ["code"],
          enabled: true,
        },
      ],
    },
    "/conversations/conv_1/messages": {
      messages: [
        {
          message_id: "msg_1",
          conversation_id: "conv_1",
          sender_type: "user",
          sender_name: "You",
          original_text: "hello",
          display_text: "hello",
          status: "completed",
          target_type: "worker",
          target_id: "reviewer",
          core_node_id: "node_user_1",
          created_at: "2026-06-17T00:00:00Z",
          updated_at: "2026-06-17T00:00:00Z",
          metadata: { target_name: "Reviewer" },
        },
        {
          message_id: "msg_2",
          conversation_id: "conv_1",
          sender_type: "orchestrator",
          sender_name: "Orchestrator",
          original_text: "",
          display_text: "working",
          status: "running",
          core_run_id: "run_1",
          metadata: { target_worker_id: "reviewer", target_worker_name: "Reviewer" },
          created_at: "2026-06-17T00:00:01Z",
          updated_at: "2026-06-17T00:00:01Z",
        },
      ],
    },
    "/conversations/conv_1/branchable-nodes": {
      nodes: [
        { core_node_id: "node_user_1", preview: "hello", created_at: "2026-06-17T00:00:00Z", active: true },
        { core_node_id: "node_user_2", preview: "next turn", created_at: "2026-06-17T00:01:00Z", active: false },
      ],
    },
    "/conversations/conv_2/messages": {
      messages: [
        {
          message_id: "msg_conv_2_user",
          conversation_id: "conv_2",
          sender_type: "user",
          sender_name: "You",
          original_text: "second",
          display_text: "second conversation message",
          status: "completed",
          core_node_id: "node_user_2",
          created_at: "2026-06-17T00:00:00Z",
          updated_at: "2026-06-17T00:00:00Z",
        },
      ],
    },
    "/conversations/conv_new/messages": { messages: [] },
    "/conversations/conv_fork/messages": {
      messages: [
        {
          message_id: "msg_fork_user",
          conversation_id: "conv_fork",
          sender_type: "user",
          sender_name: "You",
          original_text: "hello",
          display_text: "hello",
          status: "completed",
          core_node_id: "node_user_1",
          created_at: "2026-06-17T00:00:00Z",
          updated_at: "2026-06-17T00:00:00Z",
        },
      ],
    },
    "/workers/reviewer/queue": {
      queue: [
        {
          queue_id: "worker_queue_1",
          worker_id: "reviewer",
          worker_agent_id: "worker_sess_reviewer",
          session_id: "sess_1",
          parent_run_id: "run_parent",
          parent_agent_id: "agent_parent",
          task_id: "task_1",
          status: "queued",
          position: 1,
          created_at: "2026-06-17T00:00:00Z",
          updated_at: "2026-06-17T00:00:00Z",
          cancelled: false,
        },
      ],
    },
  };
}

const eventSources: MockEventSource[] = [];

class MockEventSource extends EventTarget {
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public url: string) {
    super();
    eventSources.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }

  close() {}
}

function renderApp() {
  return render(
    <AppStateProvider>
      <App />
    </AppStateProvider>,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  responses = defaultResponses();
  eventSources.length = 0;
  vi.stubGlobal("EventSource", MockEventSource);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const path = new URL(url).pathname;
      if (init?.method === "POST" && path.endsWith("/decision")) {
        return jsonResponse({ ok: true });
      }
      if (init?.method === "POST" && path === "/conversations/conv_1/cancel") {
        return jsonResponse({ cancelled: true });
      }
      if (init?.method === "POST" && path === "/conversations") {
        return jsonResponse({
          ...mockConversation,
          conversation_id: "conv_new",
          core_session_id: "sess_new",
          title: "New conversation",
          last_message_preview: "",
        });
      }
      if (init?.method === "DELETE" && path === "/conversations/conv_1") {
        responses["/conversations"] = { conversations: [secondConversation] };
        return jsonResponse({ ...mockConversation, status: "deleted" });
      }
      if (init?.method === "POST" && path === "/conversations/conv_1/messages") {
        const payload = JSON.parse(String(init.body ?? "{}")) as { text?: string };
        if (payload.text?.startsWith("@missing")) {
          return jsonResponse(
            { error: { code: "worker_not_found", message: "worker not found: missing", details: {} } },
            { ok: false, status: 404, statusText: "Not Found" },
          );
        }
        return jsonResponse({ message_id: "msg_sent", conversation_id: "conv_1", core_session_id: "sess_1", core_run_id: "run_sent", status: "queued" });
      }
      if (init?.method === "POST" && path.endsWith("/workers") && path.startsWith("/conversations/")) {
        const payload = JSON.parse(String(init.body ?? "{}")) as { worker_id?: string };
        const worker = workersResponse().find((item) => item.worker_id === payload.worker_id) ?? mockWorker;
        const key = path;
        const existing = ((responses[key] as { workers?: typeof mockWorker[] } | undefined)?.workers ?? []) as typeof mockWorker[];
        responses[key] = { workers: [...existing.filter((item) => item.worker_id !== worker.worker_id), worker] };
        return jsonResponse(worker);
      }
      if (init?.method === "DELETE" && path.startsWith("/conversations/") && path.includes("/workers/")) {
        const workersPath = path.replace(/\/workers\/[^/]+$/, "/workers");
        const workerId = decodeURIComponent(path.split("/").at(-1) ?? "");
        const existing = ((responses[workersPath] as { workers?: typeof mockWorker[] } | undefined)?.workers ?? []) as typeof mockWorker[];
        responses[workersPath] = { workers: existing.filter((item) => item.worker_id !== workerId) };
        return jsonResponse({ removed: true, worker_id: workerId });
      }
      if (init?.method === "POST" && path === "/conversations/conv_1/skills/brainstorming/load") {
        return jsonResponse({
          session_id: "sess_1",
          name: "brainstorming",
          path: "/tmp/skills/brainstorming/SKILL.md",
          loaded: true,
          already_loaded: false,
        });
      }
      if (init?.method === "POST" && path === "/conversations/conv_1/fork") {
        responses["/conversations"] = { conversations: [forkedConversation, mockConversation, secondConversation] };
        return jsonResponse({ conversation_id: "conv_fork", core_session_id: "sess_fork" });
      }
      if (init?.method === "POST" && path === "/conversations/conv_1/branch") {
        return jsonResponse({ switched: true });
      }
      if (init?.method === "POST" && path === "/workers/reviewer/queue/worker_queue_1/cancel") {
        return jsonResponse({ cancelled: true });
      }
      if (init?.method === "POST" && path === "/workers/reviewer/disable") {
        const worker = { ...mockWorker, enabled: false };
        setWorkerResponses([worker]);
        return jsonResponse(worker);
      }
      if (init?.method === "POST" && path === "/workers/reviewer/enable") {
        const worker = { ...mockWorker, enabled: true };
        setWorkerResponses([worker]);
        return jsonResponse(worker);
      }
      if (init?.method === "DELETE" && path === "/workers/reviewer") {
        const worker = { ...mockWorker, enabled: false, deleted_at: "2026-06-17T00:00:00Z" };
        setWorkerResponses([]);
        return jsonResponse(worker);
      }
      if (init?.method === "PATCH" && path === "/workers/reviewer") {
        return jsonResponse(mockWorker);
      }
      if (init?.method === "POST" && path === "/workers") {
        return jsonResponse(mockWorker);
      }
      return jsonResponse(responses[path] ?? {});
    }),
  );
});

describe("Agent Hub app", () => {
  it("renders the three main panes and loaded data", async () => {
    renderApp();
    expect(await screen.findByText("Conversations")).toBeInTheDocument();
    expect(screen.getByText("Workers")).toBeInTheDocument();
    expect(await screen.findByText("http://127.0.0.1:8765")).toBeInTheDocument();
    expect(await screen.findByText("CLAUDE.md 1 · skills 1")).toBeInTheDocument();
    expect(await screen.findByText("Test conversation")).toBeInTheDocument();
    expect(await screen.findByText("Second conversation")).toBeInTheDocument();
    expect(await screen.findByText("Reviewer")).toBeInTheDocument();
  });

  it("shows worker runtime and model summaries", async () => {
    const runningWorker = {
      ...mockWorker,
      status: "running",
      queue_length: 2,
      current_task_id: "task_review",
      current_run_id: "run_review",
      current_step_id: "step_review",
      model: { provider: "openai", name: "qwen2.5:7b", api_key: "***" },
    };
    setWorkerResponses([runningWorker]);
    renderApp();
    await screen.findByText("reviewer · running · queue 2");
    const workerButton = screen.getByRole("button", { name: /View reviewer/i });
    expect(within(workerButton).getByText("reviewer · running · queue 2")).toBeInTheDocument();
    expect(within(workerButton).getByText("task task_review · step step_review · run run_review")).toBeInTheDocument();
    expect(within(workerButton).getByText("openai · qwen2.5:7b")).toBeInTheDocument();
  });

  it("switches conversations and loads the selected message history", async () => {
    renderApp();
    fireEvent.click(await screen.findByText("Second conversation"));
    expect(await screen.findByText("second conversation message")).toBeInTheDocument();
    expect(screen.getByText("Second conversation").closest(".conversation-row")).toHaveClass("active");
  });

  it("creates a new conversation from the sidebar", async () => {
    renderApp();
    fireEvent.click(await screen.findByLabelText("New conversation"));
    expect(await screen.findByText("New conversation")).toBeInTheDocument();
    expect(screen.getByText("New conversation").closest(".conversation-row")).toHaveClass("active");
  });

  it("deletes a conversation from the sidebar", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    fireEvent.click(screen.getByLabelText("Delete Test conversation"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/conversations/conv_1", expect.objectContaining({ method: "DELETE" }));
    });
    expect(screen.queryByText("Test conversation")).not.toBeInTheDocument();
    expect(screen.getByText("Second conversation").closest(".conversation-row")).toHaveClass("active");
  });

  it("shows create conversation errors in the sidebar", async () => {
    vi.mocked(fetch).mockImplementationOnce(async () => jsonResponse(mockHealth));
    vi.mocked(fetch).mockImplementationOnce(async () => jsonResponse({ conversations: [] }));
    vi.mocked(fetch).mockImplementationOnce(async () => jsonResponse({ workers: [] }));
    vi.mocked(fetch).mockImplementationOnce(async () => jsonResponse({ tools: [] }));
    vi.mocked(fetch).mockImplementationOnce(async () =>
      jsonResponse(
        { error: { code: "config_invalid", message: "invalid config", details: {} } },
        { ok: false, status: 503, statusText: "Service Unavailable" },
      ),
    );
    renderApp();
    fireEvent.click(await screen.findByLabelText("New conversation"));
    expect(await screen.findByText("config_invalid: invalid config")).toBeInTheDocument();
  });

  it("shows startup error details from health", async () => {
    responses["/health"] = {
      ...mockHealth,
      ok: false,
      status: "core_failed",
      core_started: false,
      error: { code: "config_invalid", message: "invalid provider config", details: {} },
    };
    renderApp();
    expect(await screen.findByText("config_invalid: invalid provider config")).toBeInTheDocument();
  });

  it("shows mention candidates and inserts selected worker", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "@" } });
    expect(await screen.findByText("@Orchestrator")).toBeInTheDocument();
    expect(screen.getByText("@reviewer")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(input).toHaveValue("@reviewer ");
  });

  it("starts new conversations without workers and lets users add one", async () => {
    renderApp();
    fireEvent.click(await screen.findByLabelText("New conversation"));
    expect(await screen.findByText("No workers added to this conversation.")).toBeInTheDocument();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "@" } });
    expect(await screen.findByText("@Orchestrator")).toBeInTheDocument();
    expect(screen.queryByText("@reviewer")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Available workers/ }));
    fireEvent.click(screen.getByLabelText("Add reviewer to conversation"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_new/workers",
        expect.objectContaining({ method: "POST", body: JSON.stringify({ worker_id: "reviewer" }) }),
      );
    });

    fireEvent.change(input, { target: { value: "@r" } });
    expect(await screen.findByText("@reviewer")).toBeInTheDocument();
  });

  it("collapses available workers by default and expands them on demand", async () => {
    responses["/conversations/conv_1/workers"] = { workers: [] };
    renderApp();
    expect(await screen.findByText("No workers added to this conversation.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Conversation workers/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: /Available workers/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText("Add reviewer to conversation")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Available workers/ }));
    expect(screen.getByRole("button", { name: /Available workers/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByLabelText("Add reviewer to conversation")).toBeInTheDocument();
  });

  it("shows backend send errors inline", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "@missing inspect this" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(await screen.findByText("worker_not_found: worker not found: missing")).toBeInTheDocument();
    expect(input).toHaveValue("@missing inspect this");
  });

  it("shows sent messages immediately, reconciles SSE user messages, and labels empty running replies", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "instant message" } });
    fireEvent.click(screen.getByLabelText("Send"));

    expect(await screen.findByText("instant message")).toBeInTheDocument();
    expect(await screen.findByText("sending")).toBeInTheDocument();
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));

    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("message_created", {
          data: JSON.stringify({
            id: "evt_user_sent",
            type: "message_created",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:03Z",
            payload: {
              message_id: "msg_real_sent",
              conversation_id: "conv_1",
              sender_type: "user",
              sender_name: "You",
              target_type: "orchestrator",
              target_id: "orchestrator",
              original_text: "instant message",
              display_text: "instant message",
              status: "completed",
              core_run_id: "run_sent",
              created_at: "2026-06-17T00:00:03Z",
              updated_at: "2026-06-17T00:00:03Z",
            },
          }),
        }),
      );
      eventSources[0].dispatchEvent(
        new MessageEvent("message_created", {
          data: JSON.stringify({
            id: "evt_empty_reply",
            type: "message_created",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:04Z",
            payload: {
              message_id: "msg_empty_reply",
              conversation_id: "conv_1",
              sender_type: "orchestrator",
              sender_name: "Orchestrator",
              target_type: "none",
              original_text: "",
              display_text: "",
              status: "running",
              core_run_id: "run_sent",
              created_at: "2026-06-17T00:00:04Z",
              updated_at: "2026-06-17T00:00:04Z",
            },
          }),
        }),
      );
    });

    await waitFor(() => expect(screen.getAllByText("instant message")).toHaveLength(1));
    expect(await screen.findByText("Generating...")).toBeInTheDocument();
  });

  it("validates empty mention body locally", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "@reviewer" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(await screen.findByText("Message text is required after a worker mention.")).toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8765/conversations/conv_1/messages",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("shows slash command candidates and help output", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "/" } });
    expect(await screen.findByText("/help")).toBeInTheDocument();
    expect(await screen.findByText("/plan <goal>")).toBeInTheDocument();
    expect(await screen.findByText("/brainstorming <message>")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "/help" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(await screen.findByText(/Slash commands/)).toBeInTheDocument();
    expect(await screen.findByText(/\/new - Create a new conversation/)).toBeInTheDocument();
  });

  it("runs slash new and delete conversation commands", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "/new" } });
    fireEvent.click(screen.getByLabelText("Send"));
    expect(await screen.findByText("New conversation")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/conversations", expect.objectContaining({ method: "POST" }));

    fireEvent.change(input, { target: { value: "/delete" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/conversations/conv_new", expect.objectContaining({ method: "DELETE" }));
    });
  });

  it("runs slash plan by sending a plan request message", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "/plan build auth flow" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/messages",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            text: "Create a plan for: build auth flow. Use agent.plan_template, then write the plan Markdown to the suggested project plan directory.",
          }),
        }),
      );
    });
  });

  it("loads a slash skill before sending the skill message", async () => {
    renderApp();
    const input = await screen.findByPlaceholderText("Message, @worker, or /help");
    fireEvent.change(input, { target: { value: "/brainstorming design agent hub" } });
    fireEvent.click(screen.getByLabelText("Send"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/skills/brainstorming/load",
        expect.objectContaining({ method: "POST", body: JSON.stringify({ name: "brainstorming" }) }),
      );
    });
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/conversations/conv_1/messages",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ text: "design agent hub" }) }),
    );
  });

  it("handles permission cards from SSE events", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));
    const event = new MessageEvent("permission_requested", {
      data: JSON.stringify({
        id: "evt_1",
        type: "permission_requested",
        conversation_id: "conv_1",
        created_at: "2026-06-17T00:00:00Z",
        payload: {
          permission_request_id: "perm_1",
          tool_name: "code.run_command",
          permission: "write",
          target_scope: "project",
          args_summary: "ls",
        },
      }),
    });
    act(() => {
      eventSources[0].dispatchEvent(event);
    });
    expect(await screen.findByText("code.run_command")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Allow Once"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/permissions/perm_1/decision",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  it("posts allow-session and deny permission decisions", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));

    function dispatchPermission(id: string) {
      act(() => {
        eventSources[0].dispatchEvent(
          new MessageEvent("permission_requested", {
            data: JSON.stringify({
              id: `evt_${id}`,
              type: "permission_requested",
              conversation_id: "conv_1",
              created_at: "2026-06-17T00:00:00Z",
              payload: {
                permission_request_id: id,
                tool_name: "code.write_file",
                permission: "write",
                target_scope: "project",
                args_summary: "write",
              },
            }),
          }),
        );
      });
    }

    dispatchPermission("perm_session");
    fireEvent.click(await screen.findByText("Allow Session"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/permissions/perm_session/decision",
        expect.objectContaining({ body: JSON.stringify({ decision: "allow_for_session" }) }),
      );
    });
    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("permission_resolved", {
          data: JSON.stringify({
            id: "evt_perm_session_resolved",
            type: "permission_resolved",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:01Z",
            payload: { permission_request_id: "perm_session", status: "allowed", decision: "allow_for_session" },
          }),
        }),
      );
    });

    dispatchPermission("perm_deny");
    fireEvent.click(await screen.findByText("Deny"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/permissions/perm_deny/decision",
        expect.objectContaining({ body: JSON.stringify({ decision: "deny" }) }),
      );
    });
  });

  it("handles worker lifecycle SSE events", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));
    const workerMessage = {
      message_id: "msg_worker_1",
      conversation_id: "conv_1",
      sender_type: "worker",
      sender_name: "Reviewer",
      original_text: "",
      display_text: "review queued",
      status: "queued",
      core_run_id: "run_1",
      worker_id: "reviewer",
      queue_id: "queue_1",
      created_at: "2026-06-17T00:00:01Z",
      updated_at: "2026-06-17T00:00:01Z",
    };
    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("worker_queued", {
          data: JSON.stringify({
            id: "evt_worker_queued",
            type: "worker_queued",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:00Z",
            payload: { message: workerMessage },
          }),
        }),
      );
    });
    expect(await screen.findByText("review queued")).toBeInTheDocument();

    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("worker_completed", {
          data: JSON.stringify({
            id: "evt_worker_completed",
            type: "worker_completed",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:02Z",
            payload: { message: { ...workerMessage, display_text: "review done", status: "completed" } },
          }),
        }),
      );
    });
    expect(await screen.findByText("review done")).toBeInTheDocument();
  });

  it("opens an inline worker reply box and sends to that worker", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("worker_completed", {
          data: JSON.stringify({
            id: "evt_worker_completed_reply",
            type: "worker_completed",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:02Z",
            payload: {
              message: {
                message_id: "msg_worker_reply",
                conversation_id: "conv_1",
                sender_type: "worker",
                sender_name: "Reviewer",
                original_text: "",
                display_text: "review done",
                status: "completed",
                core_run_id: "run_1",
                worker_id: "reviewer",
                created_at: "2026-06-17T00:00:01Z",
                updated_at: "2026-06-17T00:00:02Z",
              },
            },
          }),
        }),
      );
    });
    fireEvent.click(await screen.findByTitle("Reply to Reviewer"));
    expect(await screen.findByText("Reply to Reviewer")).toBeInTheDocument();
    const replyInput = screen.getByPlaceholderText("Message Reviewer");
    await waitFor(() => expect(replyInput).toHaveFocus());
    expect(screen.getByPlaceholderText("Message, @worker, or /help")).toHaveValue("");

    fireEvent.change(replyInput, { target: { value: "please continue" } });
    fireEvent.click(screen.getByLabelText("Send reply to Reviewer"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/messages",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ text: "@reviewer please continue" }),
        }),
      );
    });
    expect(screen.queryByText("Reply to Reviewer")).not.toBeInTheDocument();
  });

  it("restarts the inline reply box when the same reply button is clicked again", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    act(() => {
      eventSources[0].dispatchEvent(
        new MessageEvent("worker_completed", {
          data: JSON.stringify({
            id: "evt_worker_completed_repeat_reply",
            type: "worker_completed",
            conversation_id: "conv_1",
            created_at: "2026-06-17T00:00:02Z",
            payload: {
              message: {
                message_id: "msg_worker_repeat_reply",
                conversation_id: "conv_1",
                sender_type: "worker",
                sender_name: "Reviewer",
                original_text: "",
                display_text: "review done",
                status: "completed",
                core_run_id: "run_1",
                worker_id: "reviewer",
                created_at: "2026-06-17T00:00:01Z",
                updated_at: "2026-06-17T00:00:02Z",
              },
            },
          }),
        }),
      );
    });
    fireEvent.click(await screen.findByTitle("Reply to Reviewer"));
    const replyInput = screen.getByPlaceholderText("Message Reviewer");
    fireEvent.change(replyInput, { target: { value: "unsent text" } });

    fireEvent.click(screen.getByTitle("Reply to Reviewer"));

    expect(await screen.findByText("Reply to Reviewer")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Message Reviewer")).toHaveValue("");
  });

  it("shows worker target on user messages", async () => {
    renderApp();
    expect(await screen.findByText("You -> Reviewer")).toBeInTheDocument();
    expect(await screen.findByText("Orchestrator -> Reviewer")).toBeInTheDocument();
  });

  it("opens branch picker with user nodes", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    fireEvent.click(screen.getByText("Branch"));
    expect(await screen.findByText("Branch from user message")).toBeInTheDocument();
    expect(await screen.findByText("node_user_1")).toBeInTheDocument();
  });

  it("selects branch nodes with keyboard arrows", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    fireEvent.click(screen.getByText("Branch"));
    const picker = await screen.findByRole("dialog", { name: "Branch from user message" });
    expect(await screen.findByRole("button", { name: /node_user_1/i })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(picker, { key: "ArrowDown" });
    expect(await screen.findByRole("button", { name: /node_user_2/i })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(picker, { key: "Enter" });
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/branch",
        expect.objectContaining({ body: JSON.stringify({ core_node_id: "node_user_2" }) }),
      );
    });
  });

  it("branches and forks from selected user nodes", async () => {
    renderApp();
    await screen.findByText("Test conversation");
    fireEvent.click(screen.getByText("Branch"));
    fireEvent.click(await screen.findByRole("button", { name: /node_user_1/i }));
    fireEvent.click(screen.getAllByRole("button", { name: "Branch" }).at(-1)!);
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/branch",
        expect.objectContaining({ method: "POST" }),
      );
    });

    fireEvent.click(screen.getByText("Fork"));
    fireEvent.click(await screen.findByRole("button", { name: /node_user_1/i }));
    fireEvent.click(screen.getAllByRole("button", { name: "Fork" }).at(-1)!);
    expect(await screen.findByText("Fork from node_user")).toBeInTheDocument();
    expect(screen.getByText("Fork from node_user").closest(".conversation-row")).toHaveClass("active");
  });

  it("edits worker JSON", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByLabelText("Edit reviewer"));
    const dialog = await screen.findByRole("dialog", { name: "Edit reviewer" });
    expect(screen.getByText("reviewer · idle · queue 0")).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("button", { name: "JSON" }));
    const editor = await within(dialog).findByLabelText("Worker JSON");
    expect((editor as HTMLTextAreaElement).value).toContain('"worker_id": "reviewer"');
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Edit reviewer" })).not.toBeInTheDocument());
  });

  it("validates worker editor safe ids and tool list", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByRole("button", { name: "New Internal" }));
    const dialog = await screen.findByRole("dialog", { name: "Create Worker" });
    fireEvent.click(within(dialog).getByRole("button", { name: "JSON" }));
    const editor = await within(dialog).findByLabelText("Worker JSON");
    fireEvent.change(editor, {
      target: {
        value: JSON.stringify({
          worker_id: "bad worker",
          system_prompt: "Prompt.",
          allowed_tools: ["code.read_file"],
        }),
      },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Worker" }));
    expect(await screen.findByText("worker_id contains unsafe characters.")).toBeInTheDocument();

    fireEvent.change(editor, {
      target: {
        value: JSON.stringify({
          worker_id: "safe_worker",
          system_prompt: "Prompt.",
          allowed_tools: "code.read_file",
        }),
      },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Worker" }));
    expect(await screen.findByText("allowed_tools must be a list of tool names.")).toBeInTheDocument();
  });

  it("accepts inline agent worker JSON", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByRole("button", { name: "New Internal" }));
    const dialog = await screen.findByRole("dialog", { name: "Create Worker" });
    fireEvent.click(within(dialog).getByRole("button", { name: "JSON" }));
    const editor = await within(dialog).findByLabelText("Worker JSON");
    const payload = {
      worker_id: "inline_reviewer",
      worker_pool_id: "default",
      enabled: true,
      agent: {
        agent_definition_id: "code_reviewer",
        name: "Code Reviewer",
        description: "Reviews code changes.",
        model: {
          provider: "openai",
          name: "qwen2.5:7b",
          base_url: "http://127.0.0.1:11434/v1",
          api_key: "ollama",
          temperature: 0.2,
        },
        system_prompt: "You are a senior code reviewer.",
        suggested_tools: ["code.read_file"],
        tags: ["review"],
      },
      allowed_tools: ["code.read_file"],
    };
    fireEvent.change(editor, { target: { value: JSON.stringify(payload) } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Worker" }));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/workers",
        expect.objectContaining({ method: "POST", body: JSON.stringify(payload) }),
      );
    });
  });

  it("submits model fields from worker form", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByRole("button", { name: "New Internal" }));
    const dialog = await screen.findByRole("dialog", { name: "Create Worker" });
    expect(within(dialog).queryByLabelText("Executor Type")).not.toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText("Worker ID"), { target: { value: "model_worker" } });
    fireEvent.change(within(dialog).getByLabelText("System Prompt"), { target: { value: "Model worker prompt." } });
    fireEvent.change(within(dialog).getByLabelText("Provider"), { target: { value: "openai" } });
    fireEvent.change(within(dialog).getByLabelText("Model Name"), { target: { value: "qwen2.5:7b" } });
    fireEvent.change(within(dialog).getByLabelText("Base URL"), { target: { value: "http://127.0.0.1:11434/v1" } });
    fireEvent.change(within(dialog).getByLabelText("API Key"), { target: { value: "ollama" } });
    fireEvent.change(within(dialog).getByLabelText("Temperature"), { target: { value: "0.2" } });
    fireEvent.change(within(dialog).getByLabelText("Max Output Tokens"), { target: { value: "4096" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Worker" }));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/workers",
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining('"provider":"openai"'),
        }),
      );
    });
    const workerCall = vi.mocked(fetch).mock.calls.find(([input, init]) => String(input) === "http://127.0.0.1:8765/workers" && init?.method === "POST");
    expect(JSON.parse(String(workerCall?.[1]?.body))).toMatchObject({
      worker_id: "model_worker",
      system_prompt: "Model worker prompt.",
      model: {
        provider: "openai",
        name: "qwen2.5:7b",
        base_url: "http://127.0.0.1:11434/v1",
        api_key: "ollama",
        temperature: 0.2,
        max_output_tokens: 4096,
      },
    });
  });

  it("submits generic external executor fields from worker form", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByRole("button", { name: "New External" }));
    const dialog = await screen.findByRole("dialog", { name: "Create Worker" });
    expect(within(dialog).queryByLabelText("Provider")).not.toBeInTheDocument();
    expect(within(dialog).getByLabelText("Supported external workers")).toHaveTextContent("OpenCode");
    expect(within(dialog).getByLabelText("Supported external workers")).toHaveTextContent("Codex");
    expect(within(dialog).getByText("Other external workers need an adapter first.")).toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText("Worker ID"), { target: { value: "external_worker" } });
    fireEvent.change(within(dialog).getByLabelText("System Prompt"), { target: { value: "External worker prompt." } });
    fireEvent.change(within(dialog).getByLabelText("Executor Type"), { target: { value: "custom_executor" } });
    fireEvent.change(within(dialog).getByLabelText("Executor Config JSON"), {
      target: { value: JSON.stringify({ command: ["custom-agent", "acp"], cwd: "/tmp/project" }, null, 2) },
    });
    fireEvent.change(within(dialog).getByLabelText("Allowed Tools Override"), { target: { value: "external.tool\nopencode.acp" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create Worker" }));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/workers",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const workerCall = vi.mocked(fetch).mock.calls.find(([input, init]) => String(input) === "http://127.0.0.1:8765/workers" && init?.method === "POST");
    expect(JSON.parse(String(workerCall?.[1]?.body))).toMatchObject({
      worker_id: "external_worker",
      system_prompt: "External worker prompt.",
      allowed_tools: ["external.tool", "opencode.acp"],
      metadata: {
        worker_executor: {
          type: "custom_executor",
          config: { command: ["custom-agent", "acp"], cwd: "/tmp/project" },
        },
      },
    });
  });

  it("fills the codex external executor type from the supported worker option", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    fireEvent.click(screen.getByRole("button", { name: "New External" }));
    const dialog = await screen.findByRole("dialog", { name: "Create Worker" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Use Codex external worker" }));
    expect(within(dialog).getByLabelText("Executor Type")).toHaveValue("codex_pty");
  });

  it("loads existing external executor worker config into the form", async () => {
    responses["/workers"] = { workers: [mockExternalWorker] };
    renderApp();
    fireEvent.click(await screen.findByRole("button", { name: /Available workers/ }));
    await screen.findByText("OpenCode Worker");
    fireEvent.click(screen.getByLabelText("Edit opencode_worker"));
    const dialog = await screen.findByRole("dialog", { name: "Edit opencode_worker" });
    expect(within(dialog).queryByLabelText("Provider")).not.toBeInTheDocument();
    expect(within(dialog).getByLabelText("Executor Type")).toHaveValue("opencode");
    expect((within(dialog).getByLabelText("Executor Config JSON") as HTMLTextAreaElement).value).toContain('"binary": "opencode"');
    expect(within(dialog).getByLabelText("Allowed Tools Override")).toHaveValue("opencode.acp");
  });

  it("calls worker enable disable and delete APIs", async () => {
    renderApp();
    await screen.findByText("Reviewer");
    responses["/workers"] = { workers: [{ ...mockWorker, enabled: false }] };
    fireEvent.click(screen.getByLabelText("Disable reviewer"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/workers/reviewer/disable", expect.objectContaining({ method: "POST" }));
    });

    responses["/workers"] = { workers: [mockWorker] };
    fireEvent.click(screen.getByLabelText("Enable reviewer"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/workers/reviewer/enable", expect.objectContaining({ method: "POST" }));
    });

    responses["/workers"] = { workers: [mockWorker] };
    fireEvent.click(screen.getByLabelText("Delete reviewer"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:8765/workers/reviewer", expect.objectContaining({ method: "DELETE" }));
    });
  });

  it("cancels running messages and queued worker items", async () => {
    renderApp();
    await screen.findByText("working");
    fireEvent.click(screen.getByTitle("Cancel"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/conversations/conv_1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });

    fireEvent.click(screen.getByLabelText("View reviewer"));
    expect(await screen.findByText("task_1")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Cancel"));
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "http://127.0.0.1:8765/workers/reviewer/queue/worker_queue_1/cancel",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  it("shows SSE disconnected state", async () => {
    renderApp();
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));
    act(() => {
      eventSources[0].onerror?.();
    });
    expect(await screen.findByText("SSE disconnected")).toBeInTheDocument();
  });
});

function jsonResponse(data: unknown, init?: { ok?: boolean; status?: number; statusText?: string }) {
  return Promise.resolve({
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    statusText: init?.statusText ?? "OK",
    json: async () => data,
  } as Response);
}

function workersResponse(): WorkerView[] {
  return ((responses["/workers"] as { workers?: WorkerView[] } | undefined)?.workers ?? []) as WorkerView[];
}

function setWorkerResponses(workers: WorkerView[]) {
  responses["/workers"] = { workers };
  responses["/conversations/conv_1/workers"] = {
    workers: workers.filter((worker) => worker.worker_id === "reviewer" && !worker.deleted_at),
  };
}
