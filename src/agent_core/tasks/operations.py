from __future__ import annotations


def validate_no_dependency_cycle(edges: list[tuple[str, str]]) -> bool:
    graph: dict[str, list[str]] = {}
    for source, target in edges:
        graph.setdefault(source, []).append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for child in graph.get(node, []):
            if not dfs(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(dfs(node) for node in graph)

