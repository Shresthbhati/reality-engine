#!/usr/bin/env python3
"""Build the fully self-contained Reality Studio viewer HTML.

Embeds the viewer app, the vendored three.js module, OrbitControls, and
(optionally) one dataset's real artifacts (worldir.json, points.ply,
cameras.json) as base64, producing ONE viewer.html that runs offline in
any modern browser -- no server, no CDN, no network:

    python apps/viewer/build_viewer.py \
        --worldir datasets/room_capture/pipeline_out/worldir.json \
        --points  datasets/room_capture/pipeline_out/points.ply \
        --cameras datasets/room_capture/pipeline_out/cameras.json \
        --out     apps/viewer/viewer.html

Mechanism: a bootstrap script creates blob URLs for the embedded
library/controls/app sources and registers an import map resolving the
bare specifiers 'three' and 'three/addons/controls/OrbitControls.js';
the app module is then loaded with a dynamic import of the app blob URL.
Import maps apply to blob-module imports on all browsers that support
import maps (Chrome/Edge 89+, Firefox 108+, Safari 16.4+). The same
src/main.js runs embedded (file://) and served (http://).

Without --worldir/--points/--cameras the shell embeds only the app +
library; the page then fetches artifacts from ./ or accepts
drag-and-drop files.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def js_literal(src: str) -> str:
    """A JS string literal containing `src`, safe to embed in <script>."""
    return json.dumps(src)  # JSON string escaping is valid JS


def js_safe(src: str) -> str:
    """Guard against </script> prematurely closing the embedding tag."""
    return src.replace("</script>", "<\\/script>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worldir", type=Path, default=None)
    ap.add_argument("--points", type=Path, default=None)
    ap.add_argument("--cameras", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=VIEWER_DIR / "viewer.html")
    args = ap.parse_args()

    three_src = (VIEWER_DIR / "vendor" / "three.module.min.js").read_text(encoding="utf-8")
    controls_src = (VIEWER_DIR / "vendor" / "addons" / "controls" / "OrbitControls.js").read_text(encoding="utf-8")
    app_src = (VIEWER_DIR / "src" / "main.js").read_text(encoding="utf-8")

    embed: dict = {}
    if args.worldir:
        embed["worldir"] = b64(args.worldir)
    if args.points:
        embed["points_ply"] = b64(args.points)
    if args.cameras:
        embed["cameras"] = b64(args.cameras)

    bootstrap = f"""
const __srcThree = {js_literal(js_safe(three_src))};
const __srcControls = {js_literal(js_safe(controls_src))};
const __srcApp = {js_literal(js_safe(app_src))};
function blobOf(src) {{
  return URL.createObjectURL(new Blob([src], {{ type: "text/javascript" }}));
}}
window.__threeUrl = blobOf(__srcThree);
window.__controlsUrl = blobOf(__srcControls);
window.__appUrl = blobOf(__srcApp);
const importMap = document.createElement("script");
importMap.type = "importmap";
importMap.textContent = JSON.stringify({{
  imports: {{
    "three": window.__threeUrl,
    "three/addons/controls/OrbitControls.js": window.__controlsUrl,
  }},
}});
document.head.appendChild(importMap);
window.REALITY_EMBED = {json.dumps(embed) if embed else "null"};
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Reality Engine — WorldIR Viewer</title>
<script>{bootstrap}</script>
<style>
  :root {{
    --bg: #0d1117; --panel: #141a22; --line: #232d3a; --fg: #dbe4ee; --dim: #7b8a9c;
    --accent: #4da3ff; --good: #35d07f; --warn: #ffb84d;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--fg);
    font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", sans-serif; overflow: hidden; }}
  #app {{ display: grid; grid-template-columns: minmax(230px, 300px) 1fr minmax(240px, 340px); height: 100vh; }}
  @media (max-width: 980px) {{
    #app {{ grid-template-columns: 1fr; grid-template-rows: 58vh 1fr 1fr;
      grid-template-areas: "vp" "left" "right"; }}
    #viewport {{ grid-area: vp; }}
    .panel.left {{ grid-area: left; border-right: 0; border-top: 1px solid var(--line); }}
    .panel.right {{ grid-area: right; border-left: 0; }}
  }}
  .panel {{ background: var(--panel); border-right: 1px solid var(--line); display: flex;
    flex-direction: column; min-height: 0; }}
  .panel.right {{ border-right: 0; border-left: 1px solid var(--line); }}
  .panel-head {{ padding: 10px 12px; font-weight: 600; border-bottom: 1px solid var(--line);
    letter-spacing: .02em; font-size: 12px; text-transform: uppercase; color: var(--dim); }}
  .panel-body {{ overflow: auto; padding: 10px 12px; flex: 1; min-height: 0; }}
  #viewport {{ position: relative; min-width: 0; }}
  #viewport canvas {{ display: block; }}
  .toolbar {{ position: absolute; top: 10px; left: 10px; display: flex; gap: 10px;
    background: rgba(13,17,23,.85); border: 1px solid var(--line); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; z-index: 5; align-items: center; }}
  .toolbar label {{ display: flex; gap: 4px; align-items: center; cursor: pointer; color: var(--fg); }}
  .toolbar button {{ background: var(--line); color: var(--fg); border: 0; border-radius: 4px;
    padding: 3px 8px; cursor: pointer; }}
  #loading {{ position: absolute; inset: 0; display: flex; flex-direction: column; gap: 12px;
    align-items: center; justify-content: center; background: var(--bg); z-index: 10; }}
  #loading.hidden {{ display: none; }}
  #loading-bar-track {{ width: 260px; height: 4px; background: var(--line); border-radius: 2px; }}
  #loading-bar {{ height: 100%; width: 0; background: var(--accent); border-radius: 2px;
    transition: width .3s ease; }}
  .grid {{ display: grid; grid-template-columns: 92px 1fr; gap: 4px 8px; margin: 6px 0; }}
  .grid .k {{ color: var(--dim); }}
  .grid .v {{ word-break: break-word; }}
  .v.mono, .mono {{ font-family: ui-monospace, Consolas, monospace; }}
  .v.small, .small {{ font-size: 11px; }}
  .dim {{ color: var(--dim); font-size: 11px; }}
  #world-summary {{ display: grid; grid-template-columns: 64px 1fr; gap: 3px 8px;
    padding-bottom: 8px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }}
  .prov-row {{ display: flex; gap: 6px; align-items: baseline; padding: 3px 0;
    font-size: 11px; color: var(--dim); line-height: 1.4; }}
  .tag {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px;
    font-weight: 600; letter-spacing: .03em; text-transform: uppercase; color: #0d1117; }}
  .tag-scale {{ background: var(--accent); }}
  .tag-recon {{ background: var(--good); }}
  .tag-depth {{ background: var(--warn); }}
  .tag-inferred {{ background: var(--warn); }}
  .tag-reconstructed {{ background: var(--accent); }}
  .tag-observed {{ background: var(--good); }}
  .tag-unknown {{ background: var(--dim); }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }}
  #entity-filter {{ width: 100%; background: #0d1117; color: var(--fg);
    border: 1px solid var(--line); border-radius: 4px; padding: 5px 8px; margin-bottom: 8px; }}
  .entity-row {{ display: flex; gap: 7px; align-items: center; padding: 4px 6px;
    border-radius: 4px; cursor: pointer; }}
  .entity-row:hover {{ background: #1b2430; }}
  .entity-row.selected {{ background: #223047; outline: 1px solid var(--accent); }}
  .entity-row .eid {{ flex: 1; font-size: 11px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }}
  .entity-row .etype {{ color: var(--dim); font-size: 11px; }}
  .entity-row .conf {{ font-size: 11px; color: var(--good); min-width: 34px; text-align: right; }}
  .insp-head {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
  .confbar {{ height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; margin: 8px 0; }}
  .confbar > div {{ height: 100%; background: linear-gradient(90deg, var(--warn), var(--good)); }}
  .sect {{ margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--line);
    color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }}
  .obs {{ background: #10161e; border: 1px solid var(--line); border-radius: 6px;
    padding: 8px 10px; margin: 6px 0; }}
  .obs-head {{ color: var(--accent); font-size: 12px; margin-bottom: 4px; }}
  .empty {{ color: var(--dim); padding: 12px 4px; }}
</style>
</head>
<body>
<div id="app">
  <aside class="panel left">
    <div class="panel-head">World</div>
    <div class="panel-body">
      <div id="world-summary" class="grid"></div>
      <div id="world-provenance"></div>
    </div>
    <div class="panel-head">Entities</div>
    <div class="panel-body" style="flex:1">
      <input id="entity-filter" placeholder="filter by id or type…" />
      <div id="entity-list"></div>
    </div>
  </aside>

  <main id="viewport">
    <div class="toolbar">
      <label><input type="checkbox" id="toggle-points" checked /> points</label>
      <label><input type="checkbox" id="toggle-entities" checked /> entities</label>
      <label><input type="checkbox" id="toggle-cameras" checked /> cameras</label>
      <label><input type="checkbox" id="toggle-oversized" /> oversized</label>
      <button id="btn-frame-points">frame points</button>
    </div>
    <div id="loading">
      <div id="loading-msg">loading…</div>
      <div id="loading-bar-track"><div id="loading-bar"></div></div>
    </div>
  </main>

  <aside class="panel right">
    <div class="panel-head">Inspector</div>
    <div class="panel-body" id="inspector"></div>
  </aside>
</div>
<script type="module">
import(window.__appUrl).catch((err) => {{
  const el = document.getElementById("loading-msg");
  if (el) el.textContent = "viewer failed to start: " + err.message;
}});
</script>
</body>
</html>
"""

    args.out.write_text(html, encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB, embedded artifacts: {sorted(embed.keys()) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
