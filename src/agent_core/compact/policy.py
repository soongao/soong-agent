from __future__ import annotations


def should_compact(*, estimated_tokens: int, threshold: int) -> bool:
    return estimated_tokens >= threshold

