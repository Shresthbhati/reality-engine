/**
 * Detailed Chrome DevTools Protocol Telemetry & Error Audit
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const PROFILE_DIR = 'C:\\Users\\shres\\.temp-chrome-telemetry-profile';
const PORT = 9223;

if (!fs.existsSync(PROFILE_DIR)) {
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
}

const chromeProc = spawn(CHROME_PATH, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${PROFILE_DIR}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-background-networking',
  '--window-size=1280,800',
  '--enable-webgl',
  '--ignore-gpu-blocklist',
  'about:blank'
], { stdio: 'ignore' });

async function waitForCdp(maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const listRes = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      if (listRes.ok) {
        const list = await listRes.json();
        const page = list.find(t => t.type === 'page' && t.webSocketDebuggerUrl) || list[0];
        if (page?.webSocketDebuggerUrl) return page;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 200));
  }
  throw new Error("Unable to connect to Chrome CDP");
}

class CdpClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 1;
    this.pending = new Map();
    this.consoleMessages = [];
    this.networkErrors = [];
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((res, rej) => {
      this.ws.onopen = res;
      this.ws.onerror = rej;
    });

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
      if (msg.method === "Runtime.consoleAPICalled") {
        this.consoleMessages.push(msg.params);
      }
      if (msg.method === "Network.loadingFailed") {
        this.networkErrors.push(msg.params);
      }
    };
  }

  send(method, params = {}) {
    const id = this.id++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    return res.result?.value;
  }

  close() {
    if (this.ws) this.ws.close();
  }
}

async function main() {
  const target = await waitForCdp();
  const client = new CdpClient(target.webSocketDebuggerUrl);
  await client.connect();

  await client.send("Page.enable");
  await client.send("Runtime.enable");
  await client.send("Network.enable");

  // 1. Cold Load
  const t0 = performance.now();
  await client.send("Page.navigate", { url: "http://localhost:3005/worlds/w-1" });
  await new Promise((res) => {
    const onMsg = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.method === "Page.loadEventFired") {
        client.ws.removeEventListener("message", onMsg);
        res();
      }
    };
    client.ws.addEventListener("message", onMsg);
  });
  const coldLoadTime = (performance.now() - t0).toFixed(2);

  // Wait for canvas mount
  await client.evaluate(`new Promise((res) => {
    const check = () => {
      if (document.querySelector('canvas')) res();
      else setTimeout(check, 50);
    };
    check();
  })`);

  // 2. Measure Frame Timing Distribution (p50, p95, p99, min, max)
  const timing = await client.evaluate(`new Promise((res) => {
    const deltas = [];
    let last = performance.now();
    let count = 0;
    function loop(now) {
      deltas.push(now - last);
      last = now;
      count++;
      if (count < 120) {
        requestAnimationFrame(loop);
      } else {
        deltas.sort((a,b) => a - b);
        const p50 = deltas[Math.floor(deltas.length * 0.50)];
        const p95 = deltas[Math.floor(deltas.length * 0.95)];
        const p99 = deltas[Math.floor(deltas.length * 0.99)];
        const min = deltas[0];
        const max = deltas[deltas.length - 1];
        const avg = deltas.reduce((a,b) => a+b, 0) / deltas.length;
        res({ p50: p50.toFixed(2), p95: p95.toFixed(2), p99: p99.toFixed(2), min: min.toFixed(2), max: max.toFixed(2), avg: avg.toFixed(2), fps: (1000/avg).toFixed(1) });
      }
    }
    requestAnimationFrame(loop);
  })`);

  // 3. Warm Load (In-App Navigation to /worlds/w-1 from /activity)
  await client.evaluate(`window.history.pushState({}, '', '/activity'); window.dispatchEvent(new PopStateEvent('popstate'));`);
  await new Promise(r => setTimeout(r, 200));

  const tWarm0 = performance.now();
  await client.evaluate(`window.history.pushState({}, '', '/worlds/w-1'); window.dispatchEvent(new PopStateEvent('popstate'));`);
  await client.evaluate(`new Promise((res) => {
    const check = () => {
      if (document.querySelector('canvas')) res();
      else setTimeout(check, 30);
    };
    check();
  })`);
  const warmLoadTime = (performance.now() - tWarm0).toFixed(2);

  // 4. WebGL Context Warnings / Errors
  const webglErrors = client.consoleMessages.filter(m => 
    m.type === "error" || 
    (m.args || []).some(a => String(a.value).toLowerCase().includes("webgl"))
  );

  console.log("=== BROWSER CDP TELEMETRY RESULTS ===");
  console.log(`Cold Load Time       : ${coldLoadTime} ms`);
  console.log(`Warm Load Time       : ${warmLoadTime} ms`);
  console.log(`Steady-State FPS     : ${timing.fps} FPS`);
  console.log(`Frame Time p50       : ${timing.p50} ms`);
  console.log(`Frame Time p95       : ${timing.p95} ms`);
  console.log(`Frame Time p99       : ${timing.p99} ms`);
  console.log(`Frame Time Min / Max : ${timing.min} ms / ${timing.max} ms`);
  console.log(`Console Errors       : ${client.consoleMessages.filter(m => m.type === "error").length}`);
  console.log(`Network Failures     : ${client.networkErrors.length}`);
  console.log(`WebGL Warnings       : ${webglErrors.length}`);
  console.log("=====================================");

  client.close();
  chromeProc.kill();
  process.exit(0);
}

main().catch(err => {
  console.error("Telemetry failed:", err);
  chromeProc.kill();
  process.exit(1);
});
