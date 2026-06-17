from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashSuggestion:
    completion: str
    usage: str
    description: str


@dataclass(frozen=True)
class SlashCommandResult:
    handled: bool
    run_message: str | None = None
    keep_overlay: bool = False


@dataclass(frozen=True)
class BranchCandidate:
    node_id: str
    preview: str
    active: bool = False
