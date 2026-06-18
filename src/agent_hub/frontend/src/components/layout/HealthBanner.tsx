import { CircleAlert, CircleCheck } from "lucide-react";
import { useAppState } from "../../state/store";

export function HealthBanner() {
  const { health, eventConnection, backendBaseUrl } = useAppState();
  const ok = health?.ok;
  const connectionText =
    eventConnection === "error" ? "SSE disconnected" : eventConnection === "connecting" ? "SSE connecting" : eventConnection;
  const instructionCount = health?.context?.auto_instruction_paths.length ?? 0;
  const skillCount = health?.context?.skill_count ?? 0;
  const contextTitle = [
    ...(health?.context?.auto_instruction_paths ?? []).map((path) => `instruction: ${path}`),
    ...(health?.context?.skills ?? []).map((skill) => `skill: ${skill.name}${skill.description ? ` - ${skill.description}` : ""}`),
  ].join("\n");
  return (
    <header className={`health-banner ${eventConnection === "error" ? "connection-error" : ""}`}>
      <div className="health-title">
        {ok ? <CircleCheck size={16} /> : <CircleAlert size={16} />}
        <span>Agent Hub</span>
      </div>
      <div className="health-meta">
        <span title="Backend URL">{backendBaseUrl || "backend loading"}</span>
        <span title={health?.error?.message}>{health?.error ? `${health.error.code}: ${health.error.message}` : health?.model ?? "loading"}</span>
        <span title={contextTitle || "No auto instructions or skills discovered"}>
          CLAUDE.md {instructionCount} · skills {skillCount}
        </span>
        <span>{connectionText}</span>
      </div>
    </header>
  );
}
