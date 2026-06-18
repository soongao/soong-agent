export type HealthStatus = {
  ok: boolean;
  status: string;
  config_path: string;
  provider?: string;
  model?: string;
  base_url?: string;
  core_started: boolean;
  hub_db_path: string;
  project_dir: string;
  context?: {
    auto_instruction_paths: string[];
    skill_count: number;
    skills: { name: string; description?: string }[];
  };
  warnings: string[];
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export type Conversation = {
  conversation_id: string;
  core_session_id: string;
  title: string;
  status: string;
  active_core_node_id?: string | null;
  last_message_preview: string;
  created_at: string;
  updated_at: string;
};

export type Message = {
  message_id: string;
  conversation_id: string;
  sender_type: "user" | "orchestrator" | "worker" | "system" | string;
  sender_name: string;
  target_type?: string | null;
  target_id?: string | null;
  original_text: string;
  display_text: string;
  status: string;
  core_run_id?: string | null;
  core_node_id?: string | null;
  queue_id?: string | null;
  worker_id?: string | null;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkerView = {
  worker_id: string;
  worker_pool_id: string;
  agent_definition_id: string;
  name: string;
  description: string;
  system_prompt?: string | null;
  model_profile?: string | null;
  model?: Record<string, unknown> | null;
  allowed_tools?: string[] | null;
  enabled: boolean;
  deleted_at?: string | null;
  source: string;
  status: string;
  queue_length: number;
  current_task_id?: string | null;
  current_run_id?: string | null;
  current_step_id?: string | null;
  metadata?: Record<string, unknown>;
};

export type WorkerInlineAgentConfig = {
  id?: string;
  agent_definition_id?: string;
  name?: string;
  description?: string;
  body?: string;
  system_prompt?: string;
  model_profile?: string;
  model?: Record<string, unknown> | null;
  suggested_tools?: string[];
  tags?: string[];
  overrides?: string;
  enabled?: boolean;
  metadata?: Record<string, unknown>;
};

export type WorkerConfigPayload = {
  worker_id: string;
  worker_pool_id?: string;
  agent_definition_id?: string;
  agent?: WorkerInlineAgentConfig;
  name?: string;
  description?: string;
  system_prompt?: string;
  model_profile?: string;
  model?: Record<string, unknown> | null;
  allowed_tools?: string[] | null;
  enabled?: boolean;
  metadata?: Record<string, unknown>;
};

export type HubEvent = {
  id: string;
  type: string;
  conversation_id?: string | null;
  payload: Record<string, unknown>;
  created_at: string;
};

export type BranchableNode = {
  core_node_id: string;
  preview: string;
  created_at: string;
  active: boolean;
};

export type PermissionRequest = {
  permission_request_id: string;
  tool_name: string;
  permission: string;
  target_scope?: string | null;
  args_summary: string;
  suggested_decision?: string | null;
};

export type ToolView = {
  name: string;
  description: string;
  permission: string;
  tags: string[];
  enabled: boolean;
};

export type WorkerQueueItem = {
  queue_id: string;
  worker_id: string;
  worker_agent_id: string;
  session_id: string;
  parent_run_id: string;
  parent_agent_id: string;
  task_id: string;
  status: string;
  position?: number | null;
  created_at: string;
  updated_at: string;
  cancelled: boolean;
};

declare global {
  interface ImportMeta {
    env: {
      VITE_AGENTHUB_BACKEND_URL?: string;
    };
  }

  interface Window {
    agentHub?: {
      backendBaseUrl: string;
    };
  }
}
