/**
 * Script to capture verified screenshots of Reality Studio for documentation.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PROFILE_DIR = path.join(os.tmpdir(), 'temp-chrome-docs-profile');
const PORT = 9225; // use unique port
const TARGET_URL = "http://localhost:3005/worlds/wld_37ed7b447247";
const OUT_DIR = path.resolve("docs/screenshots");

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

if (!fs.existsSync(PROFILE_DIR)) {
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
}

console.log(">>> Launching Google Chrome (Headless with WebGL)...");
const chromeProc = spawn(
  CHROME_PATH,
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${PROFILE_DIR}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--window-size=1600,1000",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "about:blank",
  ],
  { stdio: "ignore" }
);

async function waitForCdp(maxAttempts = 50) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const listRes = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      if (listRes.ok) {
        const list = await listRes.json();
        const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl) || list[0];
        if (page?.webSocketDebuggerUrl) {
          return page;
        }
      }
      const newRes = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(TARGET_URL)}`, {
        method: "PUT",
      });
      if (newRes.ok) {
        const target = await newRes.json();
        if (target?.webSocketDebuggerUrl) {
          return target;
        }
      }
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("Failed to connect to Chrome via CDP");
}

async function main() {
  const target = await waitForCdp();
  console.log(">>> Connected to Chrome CDP target:", target.id);

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let id = 1;
  const pending = new Map();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(msg.error);
      else resolve(msg.result);
    }
  };

  await new Promise((res) => (ws.onopen = res));

  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = id++;
      pending.set(msgId, { resolve, reject });
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1600,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });

  console.log(">>> Navigating to", TARGET_URL);
  await send("Page.navigate", { url: TARGET_URL });
  await new Promise((r) => setTimeout(r, 4000));

  async function capture(filename) {
    const res = await send("Page.captureScreenshot", { format: "png" });
    const buffer = Buffer.from(res.data, "base64");
    const filepath = path.join(OUT_DIR, filename);
    fs.writeFileSync(filepath, buffer);
    console.log(`Saved screenshot: ${filename} (${buffer.length} bytes)`);
  }

  // 1. Initial Workspace
  await capture("01_workspace_initial.png");

  // 2. Open Hierarchy tab
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const tabs = Array.from(document.querySelectorAll('button'));
        const hierarchyTab = tabs.find(t => t.textContent.includes('Hierarchy'));
        if (hierarchyTab) hierarchyTab.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 600));
  await capture("02_hierarchy_storeys.png");

  // 3. Select an entity in hierarchy
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const items = Array.from(document.querySelectorAll('div, button, span'));
        const entityItem = items.find(el => el.textContent.includes('Storey') || el.textContent.includes('Room') || el.textContent.includes('Wall'));
        if (entityItem) entityItem.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 800));
  await capture("03_adaptive_inspector.png");

  // 4. Trigger measurement tool
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const rulerBtn = document.querySelector('button[title*="Measurement"]');
        if (rulerBtn) rulerBtn.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 600));
  await capture("04_measurement_tool.png");

  // Reset measurement
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const rulerBtn = document.querySelector('button[title*="Measurement"]');
        if (rulerBtn) rulerBtn.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 400));

  // 5. Open Spatial Query
  await send("Runtime.evaluate", {
    expression: `window.dispatchEvent(new CustomEvent('open-spatial-query'));`,
  });
  await new Promise((r) => setTimeout(r, 800));
  await capture("05_spatial_query.png");

  // Close Spatial Query
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const closeBtn = document.querySelector('button[title="Close Query Panel"]');
        if (closeBtn) closeBtn.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 400));

  // 6. Open Room Construction Modal
  await send("Runtime.evaluate", {
    expression: `window.dispatchEvent(new CustomEvent('open-room-construction'));`,
  });
  await new Promise((r) => setTimeout(r, 800));
  await capture("06_room_construction.png");

  // Close Room Construction Modal
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const closeBtn = document.querySelector('button[aria-label="Close"], button:has(svg.lucide-x)');
        if (closeBtn) closeBtn.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 400));

  // 7. Open Version Diff Modal
  await send("Runtime.evaluate", {
    expression: `window.dispatchEvent(new CustomEvent('open-version-diff'));`,
  });
  await new Promise((r) => setTimeout(r, 800));
  await capture("07_version_diff.png");

  // Close Version Diff Modal
  await send("Runtime.evaluate", {
    expression: `
      (() => {
        const closeBtn = document.querySelector('button[aria-label="Close"], button:has(svg.lucide-x)');
        if (closeBtn) closeBtn.click();
      })()
    `,
  });
  await new Promise((r) => setTimeout(r, 400));

  // 8. Open Export Panel
  await send("Runtime.evaluate", {
    expression: `window.dispatchEvent(new CustomEvent('open-export-panel'));`,
  });
  await new Promise((r) => setTimeout(r, 800));
  await capture("08_export_panel.png");

  console.log(">>> All 8 screenshots captured successfully!");
  ws.close();
  chromeProc.kill();
  process.exit(0);
}

main().catch((err) => {
  console.error("Screenshot capture error:", err);
  chromeProc.kill();
  process.exit(1);
});
