from __future__ import annotations

from agent_core.tools.builtin_code.commands import run_command
from agent_core.tools.builtin_code.files import edit_file, list_dir, read_file, search, write_file
from agent_core.tools.registry import ToolRegistry
from agent_core.types.tools import ToolDefinition


def register_builtin_code_tools(registry: ToolRegistry) -> None:
    registry.register_tool(
        ToolDefinition(
            name="code.read_file",
            description="Read a text file by line range.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines": {"type": "integer", "default": 200},
                },
                "required": ["path"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        read_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.list_dir",
            description="List directory entries.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["path"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        list_dir,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.search",
            description="Search text in files using ripgrep.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": ["string", "null"]},
                    "glob": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["query"],
            },
            permission="readonly",
            tags={"code", "filesystem", "readonly"},
        ),
        search,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.write_file",
            description="Write a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "create_dirs": {"type": "boolean", "default": True},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
            permission="write",
            tags={"code", "filesystem", "write"},
        ),
        write_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.edit_file",
            description="Edit a file by exact replacement or a single-file unified diff.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                                "replace_all": {"type": "boolean", "default": False},
                            },
                            "required": ["old", "new"],
                        },
                    },
                    "unified_diff": {"type": ["string", "null"]},
                    "create_if_missing": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
            permission="write",
            tags={"code", "filesystem", "write"},
        ),
        edit_file,
    )
    registry.register_tool(
        ToolDefinition(
            name="code.run_command",
            description="Run a command using argv list, without a shell string.",
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": ["string", "null"]},
                    "timeout_ms": {"type": ["integer", "null"]},
                },
                "required": ["argv"],
            },
            permission="write",
            tags={"code", "dangerous"},
        ),
        run_command,
    )
