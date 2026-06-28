from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_core.config.loader import load_runtime_config, resolve_model_config
from agent_core.providers.base import ModelEvent, ModelRequest
from agent_core.providers.registry import default_provider_registry
from agent_core.types.tools import ToolCall

from evals.bfcl.bfcl_format import (
    case_id,
    case_messages,
    case_tools,
    debug_record,
    load_jsonl,
    prediction_record,
)


async def run_case(
    *,
    case: dict[str, Any],
    index: int,
    provider: Any,
    model_name: str,
    max_output_tokens: int | None,
    temperature: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calls: list[ToolCall] = []
    text_parts: list[str] = []
    event_dump: list[dict[str, Any]] = []
    error: str | None = None
    try:
        request = ModelRequest(
            model=model_name,
            messages=case_messages(case),
            tools=case_tools(case),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        async for event in provider.stream(request):
            event_dump.append(_event_summary(event))
            if event.event_type == "model_text_delta" and event.text_delta:
                text_parts.append(event.text_delta)
            elif event.event_type == "model_completed":
                calls = event.tool_calls
                if event.content:
                    text_parts.extend(getattr(block, "text", "") for block in event.content if getattr(block, "type", None) == "text")
            elif event.event_type == "model_failed":
                error = event.error.message if event.error else "model_failed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    assistant_text = "".join(text_parts)
    return (
        prediction_record(case=case, index=index, calls=calls, assistant_text=assistant_text, error=error),
        debug_record(case=case, index=index, calls=calls, assistant_text=assistant_text, events=event_dump, error=error),
    )


async def run(args: argparse.Namespace) -> int:
    config, _paths = load_runtime_config(
        home_dir=args.home_dir,
        config_path=args.config,
        project_dir=args.project_dir,
    )
    model_config = resolve_model_config(config, args.model_profile)
    provider = default_provider_registry().create(model_config.provider, model_config)
    model_name = args.model or model_config.name
    max_output_tokens = args.max_output_tokens if args.max_output_tokens is not None else model_config.max_output_tokens
    temperature = args.temperature if args.temperature is not None else model_config.temperature

    cases = load_jsonl(Path(args.input))
    if args.limit is not None:
        cases = cases[: args.limit]

    output_path = Path(args.output)
    debug_path = Path(args.debug_output) if args.debug_output else output_path.with_suffix(output_path.suffix + ".debug")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as out, debug_path.open("w", encoding="utf-8") as debug:
            for index, case in enumerate(cases):
                prediction, details = await run_case(
                    case=case,
                    index=index,
                    provider=provider,
                    model_name=model_name,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                out.flush()
                debug.write(json.dumps(details, ensure_ascii=False) + "\n")
                debug.flush()
                status = "error" if prediction.get("error") else "ok"
                print(f"{index + 1}/{len(cases)} {case_id(case, index)} {status}")
    finally:
        await provider.close()

    print(f"BFCL predictions: {output_path}")
    print(f"Debug output: {debug_path}")
    return 0


def _event_summary(event: ModelEvent) -> dict[str, Any]:
    data: dict[str, Any] = {"event_type": event.event_type}
    if event.text_delta:
        data["text_delta"] = event.text_delta
    if event.tool_call_delta:
        data["tool_call_delta"] = event.tool_call_delta
    if event.tool_calls:
        data["tool_calls"] = [call.model_dump(mode="json") for call in event.tool_calls]
    if event.error:
        data["error"] = event.error.model_dump(mode="json")
    if event.metadata:
        data["metadata"] = event.metadata
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BFCL function-calling cases through the soong-agent provider adapter.")
    parser.add_argument("--input", required=True, help="BFCL JSONL input file.")
    parser.add_argument("--output", default="evals/bfcl/results/soong_agent_bfcl_result.jsonl", help="Prediction JSONL output.")
    parser.add_argument("--debug-output", default=None, help="Optional debug JSONL output.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--project-dir", default=".", help="Project dir used for loading the default soong-agent config.")
    parser.add_argument("--home-dir", default=None, help="Override SOONG_AGENT_HOME.")
    parser.add_argument("--config", default=None, help="Override config.toml path.")
    parser.add_argument("--model-profile", default=None, help="Optional model override profile from config.toml.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Override temperature for BFCL runs.")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Override max output tokens.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
