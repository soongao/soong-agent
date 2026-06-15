from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    SCHEMA_ERROR = "schema_error"
    CONFIG_ERROR = "config_error"
    PERMISSION_DENIED = "permission_denied"
    TOOL_NOT_AVAILABLE = "tool_not_available"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"

    PROVIDER_ERROR = "provider_error"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    PROVIDER_AUTH_FAILED = "provider_auth_failed"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"

    STORAGE_ERROR = "storage_error"
    MIGRATION_FAILED = "migration_failed"
    SESSION_ACTIVE = "session_active"
    PATH_CONFLICT = "path_conflict"
    FILE_NOT_FOUND = "file_not_found"
    TEXT_NOT_FOUND = "text_not_found"
    AMBIGUOUS_EDIT = "ambiguous_edit"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    PATCH_PATH_MISMATCH = "patch_path_mismatch"
    WRITE_OUTSIDE_ALLOWED_ROOTS = "write_outside_allowed_roots"

    INVALID_AGENT_DEFINITION = "invalid_agent_definition"
    DUPLICATE_AGENT_DEFINITION = "duplicate_agent_definition"
    INVALID_AGENT_OVERRIDE = "invalid_agent_override"
    CHILD_AGENT_LIMIT_EXCEEDED = "child_agent_limit_exceeded"
    WORKER_BUSY = "worker_busy"
    WORKER_NOT_AVAILABLE = "worker_not_available"
    WORKER_POOL_BUSY = "worker_pool_busy"

    TASK_NOT_FOUND = "task_not_found"
    TASK_TERMINAL = "task_terminal"
    TASK_NOT_DISPATCHABLE = "task_not_dispatchable"
    DEPENDENCY_CYCLE = "dependency_cycle"
    STEP_NOT_FOUND = "step_not_found"
    STEP_NOT_READY = "step_not_ready"
    STEP_ALREADY_CLAIMED = "step_already_claimed"
    STEP_ALREADY_CLAIMED_BY_RUN = "step_already_claimed_by_run"
    STEP_HAS_DEPENDENTS = "step_has_dependents"
    NO_STEP_CLAIMED = "no_step_claimed"
    TASK_WAL_UNAVAILABLE = "task_wal_unavailable"

    MEMORY_RECALL_FAILED = "memory_recall_failed"
    MEMORY_WRITE_FAILED = "memory_write_failed"
    SKILL_NOT_FOUND = "skill_not_found"
    SKILL_LOAD_FAILED = "skill_load_failed"

