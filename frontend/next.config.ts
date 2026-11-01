import type { NextConfig } from "next";

// launch_ui.bat sets this in git worktrees, where node_modules is a symlink into the main checkout
// and Turbopack refuses symlinks that point outside its root.
const turbopackRoot = process.env.REALITY_TURBOPACK_ROOT;
// REALITY_DESKTOP=1 is set only by scripts/build_desktop.ps1: emits a self-contained server (.next/standalone) for the Electron app.
const desktop = process.env.REALITY_DESKTOP === "1";

const nextConfig: NextConfig = {
  ...(turbopackRoot ? { turbopack: { root: turbopackRoot } } : {}),
  ...(desktop ? { output: "standalone" as const, outputFileTracingRoot: process.cwd() } : {}),
};

export default nextConfig;
