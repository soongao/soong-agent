from __future__ import annotations

from pathlib import Path

from agent_core.errors import AgentCoreError
from agent_core.errors.codes import ErrorCode


def apply_simple_unified_diff(*, path: Path, original: str, diff: str) -> tuple[str, int]:
    lines = diff.splitlines(keepends=True)
    validate_single_file_diff_path(path=path, lines=lines)
    old_lines = original.splitlines(keepends=True)
    result: list[str] = []
    old_index = 0
    applied = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("@@"):
            i += 1
            continue
        try:
            old_spec = line.split(" ")[1]
            start = int(old_spec.split(",")[0].lstrip("-"))
        except Exception as exc:
            raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "invalid hunk header") from exc
        target_index = max(start - 1, 0)
        result.extend(old_lines[old_index:target_index])
        old_index = target_index
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            hunk_line = lines[i]
            marker = hunk_line[:1]
            body = hunk_line[1:]
            if marker == " ":
                if old_index >= len(old_lines) or old_lines[old_index] != body:
                    raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "hunk context mismatch")
                result.append(old_lines[old_index])
                old_index += 1
            elif marker == "-":
                if old_index >= len(old_lines) or old_lines[old_index] != body:
                    raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "hunk removal mismatch")
                old_index += 1
                applied += 1
            elif marker == "+":
                result.append(body)
                applied += 1
            elif hunk_line.startswith("\\ No newline"):
                pass
            i += 1
    if applied == 0:
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "no patch hunks applied")
    result.extend(old_lines[old_index:])
    return "".join(result), applied


def validate_single_file_diff_path(*, path: Path, lines: list[str]) -> None:
    old_headers = [line for line in lines if line.startswith("--- ")]
    new_headers = [line for line in lines if line.startswith("+++ ")]
    if len(old_headers) > 1 or len(new_headers) > 1:
        raise AgentCoreError(ErrorCode.PATCH_PATH_MISMATCH, "unified diff must modify exactly one file")
    if any(line.startswith(("rename from ", "rename to ", "deleted file mode ", "new file mode ", "Binary files ")) for line in lines):
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "rename/delete/binary patches are not supported")
    if not old_headers and not new_headers:
        return
    if len(old_headers) != 1 or len(new_headers) != 1:
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "unified diff must contain matching --- and +++ headers")
    old_path = diff_header_path(old_headers[0])
    new_path = diff_header_path(new_headers[0])
    if old_path == "/dev/null" or new_path == "/dev/null":
        raise AgentCoreError(ErrorCode.PATCH_APPLY_FAILED, "create/delete patches are not supported")
    if not (diff_path_matches_target(path, old_path) and diff_path_matches_target(path, new_path)):
        raise AgentCoreError(ErrorCode.PATCH_PATH_MISMATCH, "unified diff path does not match target path")


def diff_header_path(header: str) -> str:
    value = header[4:].strip()
    if "\t" in value:
        value = value.split("\t", 1)[0]
    if " " in value:
        value = value.split(" ", 1)[0]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def diff_path_matches_target(path: Path, diff_path: str) -> bool:
    if not diff_path:
        return False
    candidate = Path(diff_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve() == path.resolve()
        except OSError:
            return candidate == path
    target_parts = path.parts
    diff_parts = candidate.parts
    return len(diff_parts) <= len(target_parts) and tuple(target_parts[-len(diff_parts) :]) == diff_parts
