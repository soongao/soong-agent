from __future__ import annotations


class SkillCommandsMixin:
    async def _show_skills(self) -> None:
        await self._ensure_runtime()
        assert self.runtime is not None
        skills = await self.runtime.list_skills()
        if not skills:
            await self._write_message("system", "no skills found")
            return
        rows = ["available skills"]
        rows.extend(f"{skill.name} - {skill.description or skill.path}" for skill in skills)
        rows.append("Use /<skill_name> <message> to load a skill and run the message.")
        await self._write_message("system", "\n".join(rows))

    async def _load_skill_for_run(self, name: str) -> bool:
        if self._has_active_run():
            await self._write_message("warning", "cannot load a skill while a run is active; use /cancel first")
            return False
        name = name.strip()
        if not name:
            return False
        await self._ensure_runtime()
        assert self.runtime is not None
        result = await self.runtime.load_skill(self.session_id, name, mode=self.mode)
        if result.error is not None:
            if str(result.error.code) == "skill_not_found":
                return False
            await self._write_message("error", result.error.message)
            return False
        return True

    def _skill_name_exists(self, name: str) -> bool:
        return any(skill.get("name") == name for skill in self._cached_skills_for_suggestions())
