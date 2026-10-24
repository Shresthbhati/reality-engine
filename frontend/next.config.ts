import type { NextConfig } from "next";

// launch_ui.bat sets this in git worktrees, where node_modules is a symlink into the main checkout
// and Turbopack refuses symlinks that point outside its root.
const turbopackRoot = process.env.REALITY_TURBOPACK_ROOT;

const nextConfig: NextConfig = turbopackRoot ? { turbopack: { root: turbopackRoot } } : {};

export default nextConfig;
