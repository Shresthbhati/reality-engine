"use strict";
// Runs before any page script. With contextIsolation on, the main world is reached through webFrame; the UI's API client
// (frontend/src/lib/api/client.ts) reads window.__REALITY_API_BASE__.
const { webFrame } = require("electron");

const arg = process.argv.find((a) => a.startsWith("--reality-api-base="));
if (arg) webFrame.executeJavaScript(`window.__REALITY_API_BASE__ = ${JSON.stringify(arg.split("=")[1])};`);
