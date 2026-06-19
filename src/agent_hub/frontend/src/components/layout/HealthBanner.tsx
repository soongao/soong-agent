import { CircleAlert, CircleCheck } from "lucide-react";
import { useAppState } from "../../state/store";

export function HealthBanner() {
  const { health, eventConnection, backendBaseUrl } = useAppState();
  const ok = health?.ok;
  const runtimeStatus = ok ? "runtime-ready" : health?.error ? "runtime-error" : "runtime-loading";
  const connectionText =
    eventConnection === "error" ? "SSE disconnected" : eventConnection === "connecting" ? "SSE connecting" : eventConnection;
  const instructionCount = health?.context?.auto_instruction_paths.length ?? 0;
  const skillCount = health?.context?.skill_count ?? 0;
  const contextTitle = [
    ...(health?.context?.auto_instruction_paths ?? []).map((path) => `instruction: ${path}`),
    ...(health?.context?.skills ?? []).map((skill) => `skill: ${skill.name}${skill.description ? ` - ${skill.description}` : ""}`),
  ].join("\n");
  return (
    <header className={`health-banner ${runtimeStatus} ${eventConnection === "error" ? "connection-error" : ""}`}>
      <div className="health-title">
        {ok ? <CircleCheck size={16} /> : <CircleAlert size={16} />}
        <span>Agent Hub</span>
        <small>local orchestration console</small>
      </div>
      <div className="health-meta">
        <span className="health-chip mono" title="Backend URL">{backendBaseUrl || "backend loading"}</span>
        <span className="health-chip" title={health?.error?.message}>{health?.error ? `${health.error.code}: ${health.error.message}` : health?.model ?? "loading"}</span>
        <span className="health-chip" title={health?.project_dir || "workspace loading"}>{health?.project_dir ?? "workspace loading"}</span>
        <span className="health-chip" title={contextTitle || "No auto instructions or skills discovered"}>
          CLAUDE.md {instructionCount} · skills {skillCount}
        </span>
        <span className="health-chip connection-state">{connectionText}</span>
      </div>
    </header>
  );
}
