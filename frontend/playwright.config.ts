import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "@playwright/test";

// Locally no bundled browsers are downloaded: tests drive an installed Chromium-family browser.
// Default Microsoft Edge; override with PLAYWRIGHT_CHANNEL=chrome.
// In CI (CI=1) there is no Edge: use Playwright's bundled chromium (npx playwright install --with-deps chromium).
const channel = process.env.CI ? undefined : process.env.PLAYWRIGHT_CHANNEL || "msedge";

const root = path.resolve(__dirname, "..");
// Everything the API writes lives under test-results/e2e so the suite never touches ./data.
const sandbox = path.join(__dirname, "test-results", "e2e");

fs.mkdirSync(sandbox, { recursive: true });

export const API_URL = "http://localhost:8100";
export const SEED_FILE = path.join(sandbox, "seed.json");

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results/artifacts",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 2, // more parallel Edge launches exhaust desktop resources on Windows (0xC0000142)
  reporter: [["list"], ["html", { open: "never" }]],
  timeout: 45_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: "http://localhost:3000",
    channel,
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", testIgnore: /mobile\.spec\.ts/, use: { viewport: { width: 1440, height: 900 } } },
    {
      name: "mobile",
      testMatch: /mobile\.spec\.ts/,
      use: { viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 3 },
    },
  ],
  webServer: [
    {
      command: "python -m uvicorn apps.api.main:app --port 8100",
      cwd: root,
      url: `${API_URL}/api/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 90_000,
      env: {
        DATABASE_URL: `sqlite+aiosqlite:///${path.join(sandbox, "app.db").replace(/\\/g, "/")}`,
        STORAGE_ROOT: path.join(sandbox, "storage"),
        WORLDSTORE_ROOT: path.join(sandbox, "worldstore"),
      },
    },
    {
      // CI serves the production build (a cold `next dev` compiles each route on first hit: slow/flaky there).
      command: process.env.CI ? "npm run start" : "npm run dev",
      cwd: __dirname,
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
