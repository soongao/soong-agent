# Permissions

`soong-agent` enforces permissions, hooks, path boundaries, and output redaction around tools. Treat these controls as part of the runtime contract.

## Permission Boundaries

- Write, edit, command, dangerous, network, and sensitive read operations can require approval.
- Readonly operations are normally allowed unless they touch configured sensitive paths such as secrets, private keys, or environment files.
- `allow_once` applies only to the current request. `allow_for_session` applies only to the current session and target scope. `deny` means do not retry the same operation unchanged.
- If no permission callback is available, operations that require confirmation default to denial.

## Hooks

- Pre-tool hooks can observe or deny tool calls. Explicit hook denial takes precedence over permission approval.
- Hook errors and timeouts are recorded as hook summaries. They do not block the tool unless the hook returns an explicit deny.
- Post-tool hooks can observe results, but they must not be treated as proof that a tool succeeded.

## Workspace Hygiene

- Never bypass permission checks by using a different tool to perform the same side effect.
- Do not write outside the project or configured write roots unless the tool result and policy allow it.
- Do not expose secrets in responses, artifacts, event summaries, or command output summaries.
- Avoid destructive operations unless the user explicitly requested them and permission is granted.
