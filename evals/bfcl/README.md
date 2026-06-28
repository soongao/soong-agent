# BFCL Function Calling Evaluation

This harness runs BFCL-style function-calling cases through the soong-agent
provider adapter only.

Scope:

- Loads the default soong-agent config, so the default path uses local Ollama
  from `~/.soong-agent/config.toml`.
- Sends BFCL tools as raw `ModelRequest.tools`.
- Does not register or execute tools.
- Does not use `AgentRuntime`, memory, skills, hooks, permissions, SQLite, or
  Task DAG.
- Does not filter cases by the core tool schema subset.

Run a JSONL input file:

```bash
PYTHONPATH=src python3 evals/bfcl/run_agent_bfcl.py \
  --input path/to/BFCL_v4_simple_python.json \
  --output evals/bfcl/results/soong_agent_bfcl_result.jsonl
```

For a quick local check:

```bash
PYTHONPATH=src python3 evals/bfcl/run_agent_bfcl.py \
  --input evals/bfcl/sample_cases.jsonl \
  --limit 1
```

The prediction JSONL uses the common BFCL result shape:

```json
{"id":"...","result":"function_name(arg=value)","inference_log":[...]}
```

The sibling `.debug` JSONL records parsed tool calls, text, provider events, and
request errors for diagnosis. Use the official BFCL evaluator for final scoring.
