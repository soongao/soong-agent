from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactionPayload:
    summary: str
    source_node_ids: list[str]
    stale: bool = False


class RecoveryCompact:
    def compact(self, *, texts: list[str], source_node_ids: list[str], max_summary_chars: int = 4000) -> CompactionPayload:
        combined = "\n".join(texts)
        summary = combined[:max_summary_chars]
        return CompactionPayload(summary=summary, source_node_ids=source_node_ids)
