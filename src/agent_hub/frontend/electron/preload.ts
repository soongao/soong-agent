const { contextBridge } = require("electron") as typeof import("electron");

function backendBaseUrl(): string {
  const argumentPrefix = "--agenthub-backend-url=";
  const argumentValue = process.argv.find((argument) => argument.startsWith(argumentPrefix));
  if (argumentValue) {
    const url = argumentValue.slice(argumentPrefix.length);
    console.log(`[agenthub-preload] backend url from argument: ${url}`);
    return url;
  }
  if (process.env.AGENTHUB_BACKEND_URL) {
    console.log(`[agenthub-preload] backend url from env: ${process.env.AGENTHUB_BACKEND_URL}`);
    return process.env.AGENTHUB_BACKEND_URL;
  }
  const port = process.env.AGENTHUB_BACKEND_PORT ?? "8765";
  const url = `http://127.0.0.1:${port}`;
  console.warn(`[agenthub-preload] backend url from fallback port: ${url}`);
  return url;
}

contextBridge.exposeInMainWorld("agentHub", {
  backendBaseUrl: backendBaseUrl(),
});
