from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _register_model(model_alias: str) -> None:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from evals.bfcl.soong_bfcl_handler import SoongAgentBFCLHandler

    MODEL_CONFIG_MAPPING[model_alias] = ModelConfig(
        model_name=model_alias,
        display_name=f"{model_alias} via soong-agent",
        url="local",
        org="local",
        license="local",
        model_handler=SoongAgentBFCLHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )


def generate(args: argparse.Namespace) -> None:
    from bfcl_eval._llm_response_generation import main as generation_main

    _write_run_id_file(args)
    os.environ["SOONG_BFCL_PROJECT_DIR"] = str(Path(args.project_dir).resolve())
    if args.home_dir:
        os.environ["SOONG_BFCL_HOME_DIR"] = args.home_dir
    if args.config:
        os.environ["SOONG_BFCL_CONFIG"] = args.config
    if args.model_profile:
        os.environ["SOONG_BFCL_MODEL_PROFILE"] = args.model_profile
    if args.model:
        os.environ["SOONG_BFCL_MODEL"] = args.model
    if args.max_output_tokens is not None:
        os.environ["SOONG_BFCL_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)

    _register_model(args.model_alias)
    generation_main(
        SimpleNamespace(
            model=[args.model_alias],
            test_category=args.test_category,
            temperature=args.temperature,
            include_input_log=args.include_input_log,
            exclude_state_log=args.exclude_state_log,
            num_gpus=1,
            num_threads=args.num_threads,
            gpu_memory_utilization=0.9,
            backend="sglang",
            skip_server_setup=True,
            local_model_path=None,
            result_dir=args.result_dir,
            allow_overwrite=args.allow_overwrite,
            run_ids=args.run_ids,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    _score_dir(args).mkdir(parents=True, exist_ok=True)
    _register_model(args.model_alias)
    evaluation_main(
        [args.model_alias],
        args.test_category,
        args.result_dir,
        args.score_dir,
        partial_eval=args.partial_eval,
    )


def run_all(args: argparse.Namespace) -> None:
    generate(args)
    evaluate(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official BFCL generation/evaluation through the soong-agent provider path.")
    parser.add_argument("--bfcl-root", default="/Users/bytedance/proj/temp_gorilla/berkeley-function-call-leaderboard")
    parser.add_argument("--model-alias", default="soong-agent-ollama-FC")
    parser.add_argument("--test-category", nargs="+", default=["all_scoring"])
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--score-dir", default="score")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--partial-eval", action="store_true", default=False)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--run-id-category", default=None)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--project-dir", default=str(ROOT))
    parser.add_argument("--home-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("command", choices=["generate", "evaluate", "run"], nargs="?", default="run")
    args = parser.parse_args(argv)

    bfcl_root = Path(args.bfcl_root).resolve()
    if not bfcl_root.exists():
        raise SystemExit(f"BFCL root does not exist: {bfcl_root}")
    os.environ["BFCL_PROJECT_ROOT"] = str(bfcl_root)
    if str(bfcl_root) not in sys.path:
        sys.path.insert(0, str(bfcl_root))
    return args


def _score_dir(args: argparse.Namespace) -> Path:
    score_dir = Path(args.score_dir)
    if score_dir.is_absolute():
        return score_dir
    return Path(args.bfcl_root).resolve() / score_dir


def _write_run_id_file(args: argparse.Namespace) -> None:
    if not args.run_id:
        return
    if not args.run_id_category:
        raise SystemExit("--run-id-category is required when --run-id is provided")
    run_id_path = Path(args.bfcl_root).resolve() / "test_case_ids_to_generate.json"
    run_id_path.write_text(
        json.dumps({args.run_id_category: args.run_id}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.run_ids = True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
