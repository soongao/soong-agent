# Agent Core

This project provides a Python SDK and `agentcli` command for the Agent Core runtime described in `doc/`.

The first implementation target is defined by:

- `doc/21-codegen-contract.md`
- `doc/22-codegen-plan.md`

The import package is `agent_core`, and the CLI command is `agentcli`.

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

Then run the TUI chat:

```bash
agentcli chat --path .
```

For a plain stdin/stdout loop, use:

```bash
agentcli chat --path . --plain
```

The default config uses local Ollama at `http://127.0.0.1:11434` with model `gemma4`, so make sure Ollama is running and the model is available.

## Agent Hub

Agent Hub is the local Electron/React desktop layer for `agent_core`.

Run only the backend:

```bash
./agenthub --backend
```

Run the desktop app in development mode:

```bash
cd src/agent_hub/frontend
npm install
cd ../../..
./agenthub
```

If Electron cannot download its binary on your network, install with a mirror:

```bash
cd src/agent_hub/frontend
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ npm install
```

The backend starts with `PYTHONPATH=src python3 -m agent_hub.backend`, creates `~/.soong-agent/config.toml` only when it is missing, and uses the directory where you run `./agenthub` as the project directory.
