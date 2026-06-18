import type { BrowserWindow as BrowserWindowType } from "electron";
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";
import { createRequire } from "node:module";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { app, BrowserWindow, shell } = require("electron") as typeof import("electron");

let backend: ChildProcessWithoutNullStreams | null = null;
let mainWindow: BrowserWindowType | null = null;
let resolvedBackendBaseUrl: string | null = null;
const currentDir = path.dirname(fileURLToPath(import.meta.url));

function backendBaseUrl(): string {
  if (resolvedBackendBaseUrl) return resolvedBackendBaseUrl;
  if (process.env.AGENTHUB_BACKEND_URL) return process.env.AGENTHUB_BACKEND_URL;
  const port = process.env.AGENTHUB_BACKEND_PORT ?? "8765";
  return `http://127.0.0.1:${port}`;
}

async function startBackend(): Promise<void> {
  if (process.env.AGENTHUB_BACKEND_URL) {
    resolvedBackendBaseUrl = process.env.AGENTHUB_BACKEND_URL;
    console.log(`[agenthub-electron] using external backend ${resolvedBackendBaseUrl}`);
    return;
  }
  const repoRoot = process.env.AGENTHUB_REPO_ROOT ?? process.cwd();
  const projectDir = process.env.AGENTHUB_PROJECT_DIR ?? repoRoot;
  const pythonPath = process.env.AGENTHUB_PYTHONPATH ?? path.join(repoRoot, "src");
  const port = await findAvailablePort(Number(process.env.AGENTHUB_BACKEND_PORT ?? "8765"));
  resolvedBackendBaseUrl = `http://127.0.0.1:${port}`;
  process.env.AGENTHUB_BACKEND_URL = resolvedBackendBaseUrl;
  process.env.VITE_AGENTHUB_BACKEND_URL = resolvedBackendBaseUrl;
  console.log(
    `[agenthub-electron] starting backend url=${resolvedBackendBaseUrl} repo_root=${repoRoot} project_dir=${projectDir} pythonpath=${pythonPath}`,
  );
  backend = spawn("python3", ["-m", "agent_hub.backend", "--host", "127.0.0.1", "--port", port], {
    cwd: projectDir,
    env: { ...process.env, PYTHONPATH: pythonPath },
  });
  backend.stdout.on("data", (data) => console.log(`[agenthub-backend] ${data}`));
  backend.stderr.on("data", (data) => console.error(`[agenthub-backend] ${data}`));
  backend.once("exit", (code, signal) => {
    console.log(`[agenthub-electron] backend exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    backend = null;
  });
}

async function waitForBackend(): Promise<void> {
  const base = backendBaseUrl();
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`${base}/health`);
      if (response.ok) {
        console.log(`[agenthub-electron] backend ready url=${base}`);
        return;
      }
      console.warn(`[agenthub-electron] backend health status=${response.status} url=${base}`);
    } catch {
      // keep polling
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Agent Hub backend did not become ready at ${base}`);
}

async function createWindow(): Promise<void> {
  await startBackend();
  await waitForBackend();
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    webPreferences: {
      preload: path.join(currentDir, "preload.cjs"),
      additionalArguments: [`--agenthub-backend-url=${backendBaseUrl()}`],
    },
  });
  mainWindow.on("closed", () => {
    console.log("[agenthub-electron] main window closed");
    mainWindow = null;
  });
  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    console.log(`[agenthub-renderer:${level}] ${message} (${sourceId}:${line})`);
  });
  mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
    console.error(`[agenthub-electron] renderer load failed code=${errorCode} reason=${errorDescription} url=${validatedURL}`);
  });
  mainWindow.webContents.on("did-finish-load", () => {
    console.log(`[agenthub-electron] renderer loaded url=${mainWindow?.webContents.getURL() ?? "unknown"}`);
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  const devUrl = process.env.AGENTHUB_RENDERER_URL ?? "http://127.0.0.1:5173";
  console.log(`[agenthub-electron] creating window renderer_url=${devUrl} backend_url=${backendBaseUrl()}`);
  await mainWindow.loadURL(devUrl);
}

app.whenReady().then(createWindow).catch((error) => {
  console.error(error);
  app.quit();
});

app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});

app.on("before-quit", () => {
  stopBackend();
});

function stopBackend(): void {
  if (!backend) return;
  const child = backend;
  backend = null;
  if (child.exitCode === null && child.signalCode === null) {
    console.log("[agenthub-electron] stopping backend");
    child.kill();
  }
}

async function findAvailablePort(startPort: number): Promise<string> {
  for (let port = startPort; port < startPort + 100; port += 1) {
    if (await canBind(port)) return String(port);
  }
  throw new Error(`No available Agent Hub backend port found from ${startPort} to ${startPort + 99}`);
}

function canBind(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "127.0.0.1");
  });
}
