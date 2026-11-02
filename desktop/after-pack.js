"use strict";
// electron-builder silently drops any node_modules directory from extraResources (even with an explicit filter), but the
// Next.js standalone server needs its traced node_modules. Copy it into resources/ui after the app is laid out.
const fs = require("fs");
const path = require("path");

exports.default = async function afterPack(context) {
  const from = path.join(__dirname, "build", "ui", "node_modules");
  const to = path.join(context.appOutDir, "resources", "ui", "node_modules");
  if (!fs.existsSync(from)) throw new Error(`staged UI node_modules missing: ${from} (run scripts/build_desktop.ps1)`);
  fs.cpSync(from, to, { recursive: true });
};
