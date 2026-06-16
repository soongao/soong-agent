from __future__ import annotations

from pathlib import Path


def build_skill_catalog(home_dir: Path) -> list[dict[str, str]]:
    skills_dir = home_dir / "skills"
    if not skills_dir.exists():
        return []
    catalog: list[dict[str, str]] = []
    for path in _skill_files(skills_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = _frontmatter(text)
        catalog.append(
            {
                "path": str(path),
                "name": metadata.get("name") or path.stem,
                "description": metadata.get("description") or "",
            }
        )
    return catalog


def _skill_files(skills_dir: Path) -> list[Path]:
    paths = [path for path in skills_dir.glob("*.md") if path.is_file()]
    paths.extend(path for path in skills_dir.rglob("SKILL.md") if path.is_file())
    return sorted({path.resolve(): path for path in paths}.values(), key=lambda path: path.relative_to(skills_dir).as_posix())


def find_skill_by_name(home_dir: Path, name: str) -> dict[str, str] | None:
    matches = [entry for entry in build_skill_catalog(home_dir) if entry["name"] == name]
    if len(matches) > 1:
        return {"error": "duplicate", "name": name}
    return matches[0] if matches else None


def read_skill_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def skill_context_text(*, name: str, body: str) -> str:
    return f'<skill name="{name}">\n{body}\n</skill>'


def skill_catalog_text(entries: list[dict[str, str]]) -> str:
    lines = ["# Skill Catalog", ""]
    if not entries:
        lines.append("No user skills discovered.")
        return "\n".join(lines)
    lines.append("Available user skills. Load a skill body only when relevant by calling internal.load_skill(name).")
    lines.append("")
    for entry in entries:
        lines.append(f"- name: {entry['name']}")
        lines.append(f"  description: {entry.get('description') or ''}")
        lines.append(f"  path: {entry['path']}")
    return "\n".join(lines)


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata
