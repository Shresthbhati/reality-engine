# Reality Engine desktop (offline, Windows)

Electron shell around the local API and Studio UI. Nothing here changes the dev workflow (`launch_ui.bat` is unchanged).

- `src/main.js` picks two free loopback ports, starts the API (embedded CPython + uvicorn, `apps.api.main:app`) and the Next.js
  standalone server (run with Electron's bundled Node), waits for `/api/health` and the UI root, then opens the window.
  On failure it shows an error window with the log path. Children are killed as trees on quit/crash; a hard-killed previous run's
  orphans are reaped on next start (`runtime.json`). Single-instance lock.
- Data: `%APPDATA%\RealityEngine\` (`data\app.db`, `data\worldstore`, `data\artifacts`, `logs\`, `electron\`). Nothing is written to the
  install dir. Overrides for tests: `REALITY_DESKTOP_DATA_DIR`, `REALITY_DESKTOP_HEADLESS=1` (no window; quit by creating `quit.flag`).
- Browser-to-API calls use the injected `window.__REALITY_API_BASE__` (see `frontend/src/lib/api/client.ts`) because the port is dynamic.
- Optional heavy components are NOT bundled: torch/perception models, COLMAP, GPU libs, IfcOpenShell (IFC export answers 501),
  OpenEXR. `/api/system/verification` reports them honestly as unavailable.
- Version: `pyproject.toml` `[project].version` is the single source; the build stamps it into the installer.

Build: `pwsh scripts/build_desktop.ps1` (needs internet at build time only). Smoke: `pwsh scripts/smoke_desktop.ps1`.
Output: `desktop/dist/Reality Engine-Setup-<version>.exe` and `desktop/dist/win-unpacked/`. Unsigned.
