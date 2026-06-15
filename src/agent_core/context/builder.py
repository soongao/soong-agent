from __future__ import annotations

from agent_core.context.composer import build_dynamic_system_blocks, build_static_system_blocks, build_system_blocks
from agent_core.providers.base import ModelMessage, ModelRole
from agent_core.types.runtime import Node
from agent_core.types.content import TextBlock


def build_context_messages(active_path: list[Node]) -> list[ModelMessage]:
    if not active_path:
        return []
    compaction_index = _last_compaction_index(active_path)
    if compaction_index is None:
        return [_node_to_message(node) for node in active_path]

    compaction = active_path[compaction_index]
    first_kept_node_id = compaction.metadata.get("first_kept_node_id")
    tail_start = compaction_index + 1
    if first_kept_node_id:
        for index, node in enumerate(active_path[:compaction_index]):
            if node.node_id == first_kept_node_id:
                tail_start = index + 1
                break
    tail = active_path[tail_start:compaction_index] + active_path[compaction_index + 1 :]
    return [_compaction_message(compaction)] + [_node_to_message(node) for node in tail]


def _last_compaction_index(nodes: list[Node]) -> int | None:
    for index in range(len(nodes) - 1, -1, -1):
        if nodes[index].node_type == "compaction":
            return index
    return None


def _node_to_message(node: Node) -> ModelMessage:
    if node.role == "assistant":
        role = ModelRole.ASSISTANT
    elif node.role == "tool":
        role = ModelRole.TOOL
    else:
        role = ModelRole.USER
    return ModelMessage(role=role, content=node.content, node_type=node.node_type, metadata={"node_id": node.node_id, **node.metadata})


def _compaction_message(node: Node) -> ModelMessage:
    text = "\n".join(getattr(block, "text", "") for block in node.content if getattr(block, "type", None) == "text")
    return ModelMessage(
        role=ModelRole.USER,
        content=[TextBlock(text=f"<compaction node_id=\"{node.node_id}\">\n{text}\n</compaction>")],
        node_type="compaction",
        metadata={"node_id": node.node_id, **node.metadata},
    )


__all__ = [
    "build_context_messages",
    "build_dynamic_system_blocks",
    "build_static_system_blocks",
    "build_system_blocks",
]
