from __future__ import annotations


SLASH_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("help", "/help", "show slash command help"),
    ("clear", "/clear", "clear the transcript"),
    ("new", "/new [session_id]", "start a fresh session"),
    ("mode", "/mode [normal|orchestrator]", "show or set run mode"),
    ("plan", "/plan <goal>", "create and write a plan for a goal"),
    ("session", "/session", "show current session details"),
    ("sessions", "/sessions [n]", "list recent sessions"),
    ("use", "/use <session_id>", "switch current session"),
    ("active", "/active", "show current active node"),
    ("session-nodes", "/session-nodes [n]", "list nodes in current session"),
    ("nodes", "/nodes [n]", "list nodes in current session"),
    ("branch", "/branch [node_id]", "list nodes or switch active node"),
    ("fork-session", "/fork-session [node_id]", "fork current path into a new session"),
    ("config", "/config", "show config and home paths"),
    ("skills", "/skills", "list available skills"),
    ("history", "/history [n]", "show recent prompts"),
    ("autoscroll", "/autoscroll", "toggle transcript autoscroll"),
    ("cancel", "/cancel", "cancel the active run"),
    ("exit", "/exit", "quit the TUI"),
    ("quit", "/quit", "quit the TUI"),
)


def parse_slash_command(message: str) -> tuple[str, str]:
    command, _, argument = message[1:].partition(" ")
    return command.lower(), argument.strip()


def parse_positive_int(text: str) -> int | None:
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def matching_slash_commands(text: str) -> list[tuple[str, str, str]]:
    command, _, _argument = text[1:].partition(" ")
    prefix = command.lower()
    return [entry for entry in SLASH_COMMANDS if entry[0].startswith(prefix)]


def slash_help_text() -> str:
    rows = ["slash commands"]
    rows.extend(f"{usage} - {description}" for _name, usage, description in SLASH_COMMANDS)
    rows.append("/<skill_name> <message> - load a skill and run the message")
    return "\n".join(rows)


def plan_request_message(goal: str) -> str:
    return (
        f"Create a plan for: {goal}. "
        "Use agent.plan_template, then write the plan Markdown to the suggested project plan directory."
    )
