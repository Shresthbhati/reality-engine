import type { NextConfig } from "next";

// REALITY_DESKTOP=1 is set only by scripts/build_desktop.ps1: emits a self-contained server (.next/standalone) for the Electron app.
const nextConfig: NextConfig =
  process.env.REALITY_DESKTOP === "1" ? { output: "standalone", outputFileTracingRoot: process.cwd() } : {};

export default nextConfig;
