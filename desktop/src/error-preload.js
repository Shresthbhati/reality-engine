"use strict";
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("realityError", {
  openLogs: () => ipcRenderer.send("open-logs"),
  quit: () => ipcRenderer.send("quit"),
});
