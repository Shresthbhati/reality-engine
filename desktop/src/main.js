"use strict";
/**
 * Reality Engine desktop shell.
 *
 * Starts the packaged API (embedded CPython + uvicorn) and the Next.js standalone UI on free loopback ports, waits for
 * /api/health, then opens the window. Everything writable lives under %APPDATA%/RealityEngine; nothing is written to the
 * install directory. Child process trees are killed on quit, on crash and (for a previous crashed run) on next start.
 */
const { app, BrowserWindow, ipcMain, shell, dialog } = require("electron");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const net = require("net");
const path = require("path");

const HEADLESS = process.env.REALITY_DESKTOP_HEADLESS === "1";
const DATA_DIR = process.env.REALITY_DESKTOP_DATA_DIR || path.join(app.getPath("appData"), "RealityEngine");
const LOG_DIR = path.join(DATA_DIR, "logs");
const RUNTIME_FILE = path.join(DATA_DIR, "runtime.json");
const API_TIMEOUT_MS = 90_000;
const UI_TIMEOUT_MS = 60_000;

// Session data (cache, local storage) also stays out of the install dir.
app.setPath("userData", path.join(DATA_DIR, "electron"));

const resources = app.isPackaged ? process.resourcesPath : path.join(__dirname, "..", "build");
const PY_EXE = path.join(resources, "python", "python.exe");
const UI_SERVER = path.join(resources, "ui", "server.js");

/** @type {{name: string, child: import("child_process").ChildProcess}[]} */
const children = [];
let mainLog = null;
let win = null;
let quitting = false;
let runtime = { mainPid: process.pid, apiPid: null, uiPid: null, apiPort: null, uiPort: null, ready: false };

function ensureDirs() {
  for (const d of [DATA_DIR, LOG_DIR, path.join(DATA_DIR, "data", "worldstore"), path.join(DATA_DIR, "data", "artifacts")]) {
    fs.mkdirSync(d, { recursive: true });
  }
}

function openLog(name) {
  const file = path.join(LOG_DIR, name);
  try {
    if (fs.existsSync(file) && fs.statSync(file).size > 5 * 1024 * 1024) fs.renameSync(file, file + ".1");
  } catch {}
  return fs.createWriteStream(file, { flags: "a" });
}

function log(msg) {
  const line = `${new Date().toISOString()} [main] ${msg}\n`;
  if (mainLog) try { fs.appendFileSync(mainLog, line); } catch {}
  process.stdout.write(line);
}

function writeRuntime() {
  try {
    fs.writeFileSync(RUNTIME_FILE, JSON.stringify(runtime));
  } catch {}
}

function killTree(pid) {
  if (!pid) return;
  spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
}

function imageName(pid) {
  const r = spawnSync("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"], { windowsHide: true, encoding: "utf8" });
  const m = /^"([^"]+)"/.exec((r.stdout || "").trim());
  return m ? m[1].toLowerCase() : null;
}

/** A previous run that was killed hard (power loss, Task Manager on the shell only) leaves its children running. Reap them. */
function reapStaleChildren() {
  let prev;
  let killed = false;
  try {
    prev = JSON.parse(fs.readFileSync(RUNTIME_FILE, "utf8"));
  } catch {
    return false;
  }
  const mainImage = prev.mainPid ? imageName(prev.mainPid) : null;
  log(`stale runtime found: main=${prev.mainPid} (${mainImage || "dead"}) api=${prev.apiPid} ui=${prev.uiPid}`);
  if (mainImage && prev.mainPid !== process.pid) return false; // that instance is alive (lock handles it)
  for (const [pid, want] of [[prev.apiPid, "python.exe"], [prev.uiPid, path.basename(process.execPath).toLowerCase()]]) {
    if (pid && imageName(pid) === want) {
      log(`reaping orphan from previous run: pid ${pid} (${want})`);
      killTree(pid);
      killed = true;
    }
  }
  // Chromium helper processes of the dead shell also survive and keep the single-instance lock alive: reap every
  // remaining child of the dead main pid, as trees.
  const r = spawnSync(
    "powershell",
    ["-NoProfile", "-Command", `Get-CimInstance Win32_Process -Filter "ParentProcessId=${Number(prev.mainPid)}" | ForEach-Object { $_.ProcessId }`],
    { windowsHide: true, encoding: "utf8" },
  );
  for (const line of (r.stdout || "").split(/\s+/)) {
    const pid = parseInt(line, 10);
    if (pid) {
      log(`reaping child of dead shell: pid ${pid}`);
      killTree(pid);
      killed = true;
    }
  }
  try {
    fs.unlinkSync(RUNTIME_FILE);
  } catch {}
  return killed;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function httpStatus(url) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout: 2500 }, (res) => {
      res.resume();
      resolve(res.statusCode);
    });
    req.on("error", () => resolve(0));
    req.on("timeout", () => {
      req.destroy();
      resolve(0);
    });
  });
}

async function waitFor(url, timeoutMs, child, label) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    if (child.exitCode !== null) throw new Error(`${label} exited with code ${child.exitCode} before becoming ready`);
    if ((await httpStatus(url)) === 200) return;
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(`${label} did not answer ${url} within ${Math.round(timeoutMs / 1000)}s`);
}

function startChild(name, cmd, args, opts, logName) {
  const out = openLog(logName);
  const child = spawn(cmd, args, { ...opts, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.pipe(out, { end: false });
  child.stderr.pipe(out, { end: false });
  child.on("error", (e) => log(`${name} spawn error: ${e.message}`));
  child.on("exit", (code, sig) => {
    log(`${name} exited code=${code} signal=${sig}`);
    out.end();
  });
  children.push({ name, child });
  log(`${name} started pid=${child.pid}`);
  return child;
}

function killChildren() {
  for (const { child } of children.splice(0)) killTree(child.pid);
  try {
    fs.unlinkSync(RUNTIME_FILE);
  } catch {}
}

function sqliteUrl(file) {
  return "sqlite+aiosqlite:///" + file.replace(/\\/g, "/");
}

async function startServices() {
  for (const [what, p] of [["Python runtime", PY_EXE], ["UI server", UI_SERVER]]) {
    if (!fs.existsSync(p)) throw new Error(`${what} is missing from the installation: ${p}`);
  }
  const apiPort = await freePort();
  let uiPort = await freePort();
  while (uiPort === apiPort) uiPort = await freePort();
  runtime.apiPort = apiPort;
  runtime.uiPort = uiPort;

  const dataRoot = path.join(DATA_DIR, "data");
  const api = startChild(
    "api",
    PY_EXE,
    // -s: ignore the user's %APPDATA%\Python site-packages (and its .pth files) so the app only ever sees its bundled packages
    ["-s", "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", String(apiPort), "--log-level", "info"],
    {
      cwd: DATA_DIR,
      env: {
        ...process.env,
        DATABASE_URL: sqliteUrl(path.join(dataRoot, "app.db")),
        STORAGE_ROOT: path.join(dataRoot, "artifacts"),
        WORLDSTORE_ROOT: path.join(dataRoot, "worldstore"),
        CORS_ORIGINS: `http://127.0.0.1:${uiPort},http://localhost:${uiPort}`,
        PYTHONUTF8: "1",
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
      },
    },
    "api.log",
  );
  runtime.apiPid = api.pid;
  writeRuntime();

  const ui = startChild(
    "ui",
    process.execPath,
    [UI_SERVER],
    {
      cwd: path.dirname(UI_SERVER),
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: "1",
        NODE_ENV: "production",
        PORT: String(uiPort),
        HOSTNAME: "127.0.0.1",
        REALITY_BACKEND_URL: `http://127.0.0.1:${apiPort}`,
        NEXT_TELEMETRY_DISABLED: "1",
      },
    },
    "ui.log",
  );
  runtime.uiPid = ui.pid;
  writeRuntime();

  await waitFor(`http://127.0.0.1:${apiPort}/api/health`, API_TIMEOUT_MS, api, "Engine API");
  log("API healthy");
  await waitFor(`http://127.0.0.1:${uiPort}/`, UI_TIMEOUT_MS, ui, "Studio UI");
  log("UI healthy");
  runtime.ready = true;
  writeRuntime();
  return { apiPort, uiPort };
}

function showError(err) {
  log(`FATAL: ${err.message}`);
  killChildren();
  if (HEADLESS) {
    app.exit(1);
    return;
  }
  const w = new BrowserWindow({
    width: 640,
    height: 420,
    resizable: false,
    title: "Reality Engine - could not start",
    webPreferences: { preload: path.join(__dirname, "error-preload.js"), contextIsolation: true, sandbox: true },
  });
  w.setMenuBarVisibility(false);
  w.loadFile(path.join(__dirname, "error.html"), { query: { message: err.message, logs: LOG_DIR } });
  win = w;
}

function openMainWindow(apiPort, uiPort) {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    title: "Reality Engine",
    backgroundColor: "#0b0f14",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      additionalArguments: [`--reality-api-base=http://127.0.0.1:${apiPort}`],
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });
  win.setMenuBarVisibility(false);
  const origin = `http://127.0.0.1:${uiPort}`;
  // The UI is the only thing this window may show; external links open in the default browser.
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(origin)) {
      e.preventDefault();
      shell.openExternal(url);
    }
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.loadURL(origin + "/");
}

ipcMain.on("open-logs", () => shell.openPath(LOG_DIR));
ipcMain.on("quit", () => app.quit());

// Reap a crashed previous run BEFORE taking the lock: its orphaned helpers would otherwise hold it and block this start.
try {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  mainLog = path.join(LOG_DIR, "main.log");
  if (reapStaleChildren()) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 2500); // let the OS release the dead run's lock/handles
} catch (e) {
  console.error(e);
}

if (!app.requestSingleInstanceLock()) {
  app.exit(0);
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  const shutdown = () => {
    quitting = true;
    killChildren();
  };
  app.on("before-quit", shutdown);
  app.on("will-quit", shutdown);
  process.on("exit", shutdown);
  process.on("uncaughtException", (e) => {
    log(`uncaughtException: ${e && e.stack}`);
    shutdown();
    app.exit(1);
  });
  app.on("window-all-closed", () => app.quit());

  app.whenReady().then(async () => {
    try {
      ensureDirs();
      log(`Reality Engine ${app.getVersion()} starting; data dir ${DATA_DIR}`);
      const { apiPort, uiPort } = await startServices();
      if (HEADLESS) {
        // No window to close in headless/smoke mode: quit when the quit-flag file appears (graceful path, same as a user quit).
        const flag = path.join(DATA_DIR, "quit.flag");
        setInterval(() => {
          if (fs.existsSync(flag)) {
            try { fs.unlinkSync(flag); } catch {}
            app.quit();
          }
        }, 300);
      } else if (!quitting) {
        openMainWindow(apiPort, uiPort);
      }
    } catch (e) {
      showError(e);
    }
  });
}
