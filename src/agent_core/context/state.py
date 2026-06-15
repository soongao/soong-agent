from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class RuntimeContextState:
    instruction_hashes: dict[str, str] = field(default_factory=dict)
    skill_hashes: dict[str, str] = field(default_factory=dict)
    memory_contexts: list[dict] = field(default_factory=list)
    skill_contexts: list[dict] = field(default_factory=list)
    instruction_contexts: list[dict] = field(default_factory=list)

    def mark_instruction(self, path: Path) -> bool:
        digest = file_hash(path)
        key = str(path.resolve())
        already_loaded = self.instruction_hashes.get(key) == digest
        if not already_loaded:
            self.instruction_hashes[key] = digest
            text = path.read_text(encoding="utf-8", errors="replace")
            self.instruction_contexts = [item for item in self.instruction_contexts if item.get("path") != key]
            self.instruction_contexts.append({"path": key, "hash": digest, "body": text})
        return already_loaded

    def mark_skill(self, name: str, path: Path, body: str) -> bool:
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        already_loaded = self.skill_hashes.get(name) == digest
        if not already_loaded:
            self.skill_hashes[name] = digest
            self.skill_contexts.append({"name": name, "path": str(path), "hash": digest, "body": body})
        return already_loaded
