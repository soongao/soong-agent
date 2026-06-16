# Soong Agent

`soong-agent` is a Python SDK and minimal CLI for the Agent Core runtime described in `doc/`.

The first implementation target is defined by:

- `doc/21-codegen-contract.md`
- `doc/22-codegen-plan.md`

The import package is `agent_core`, and the CLI command is `soong-agent`.

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Create a default Ollama config:

```bash
mkdir -p ~/.soong-agent
cp src/agent_core/assets/templates/config_default.toml ~/.soong-agent/config.toml
```

Then run the interactive CLI:

```bash
soong-agent chat --path .
```

The default config uses local Ollama at `http://127.0.0.1:11434` with model `gemma4`, so make sure Ollama is running and the model is available.
