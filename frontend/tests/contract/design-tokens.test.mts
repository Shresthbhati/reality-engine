/**
 * Design tokens (plan U.3): the colour tokens in src/app/globals.css must be complete in BOTH themes and meet WCAG 2.2 AA.
 *
 * - light overrides are layered on the dark `:root` set, so a token the light block forgets silently keeps its dark value;
 *   every theme-dependent token must therefore be redefined in light (the test lists the ones allowed to be shared);
 * - the two light blocks (attribute + prefers-color-scheme) are hand-duplicated in CSS and must stay identical;
 * - text pairs >= 4.5:1, UI-component boundaries and the focus ring >= 3:1.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../../src/app/globals.css", import.meta.url), "utf8");

/** The body of the first `{...}` after `selector` (no nested braces in these token blocks). */
function block(selectorPattern: RegExp, from = 0): { body: string; end: number } {
  const m = selectorPattern.exec(css.slice(from));
  assert.ok(m, `selector not found: ${selectorPattern}`);
  const open = from + m.index + m[0].length;
  const close = css.indexOf("}", open);
  return { body: css.slice(open, close), end: close };
}

function decls(body: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/gi)) out[m[1]] = m[2].trim();
  return out;
}

const dark = decls(block(/^:root\s*\{/m).body);
const lightAttr = decls(block(/^:root\[data-theme="light"\]\s*\{/m).body);
const mediaStart = css.indexOf("@media (prefers-color-scheme: light)");
const lightMedia = decls(block(/:root:not\(\[data-theme="dark"\]\)\s*\{/, mediaStart).body);
const light = { ...dark, ...lightAttr };

function rgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  assert.match(h, /^[0-9a-f]{6}$/i, `not a 6-digit hex colour: ${hex}`);
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}
function luminance([r, g, b]: [number, number, number]): number {
  const f = (c: number) => ((c /= 255) <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function contrast(a: string, b: string): number {
  const [la, lb] = [luminance(rgb(a)), luminance(rgb(b))];
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

const THEMES: [string, Record<string, string>][] = [["dark", dark], ["light", light]];

test("the two light blocks are identical", () => {
  assert.deepEqual(lightMedia, lightAttr);
});

/** Tokens that may legitimately be the same in both themes (brand-neutral or already theme-safe). */
const SHARED = new Set([
  "--font-sans", "--font-mono",
  "--conf-unknown", // neutral grey
  "--primary", "--primary-hover", "--on-primary", // brand fill and the text on it are identical in both themes
]);

test("every theme-dependent colour token is redefined in the light theme", () => {
  const missing = Object.keys(dark).filter(
    (k) => !(k in lightAttr) && !SHARED.has(k) && !k.startsWith("--font") && !k.startsWith("--radius") && !k.startsWith("--space"),
  );
  assert.deepEqual(missing, [], `light theme keeps the dark value of: ${missing.join(", ")}`);
});

const SURFACES = ["--bg-base", "--bg-sidebar", "--bg-surface", "--bg-elevated"];

test("text tokens meet 4.5:1 on every surface, in both themes", () => {
  const fails: string[] = [];
  for (const [name, t] of THEMES)
    for (const fg of ["--text-primary", "--text-secondary", "--text-tertiary"])
      for (const bg of SURFACES) {
        const c = contrast(t[fg], t[bg]);
        if (c < 4.5) fails.push(`${name}: ${fg} on ${bg} = ${c.toFixed(2)}`);
      }
  assert.deepEqual(fails, []);
});

test("status and accent colours used as text meet 4.5:1 on base and surface, in both themes", () => {
  const fails: string[] = [];
  for (const [name, t] of THEMES)
    for (const fg of ["--accent", "--success", "--warning", "--error", "--info", "--sage", "--purple"])
      for (const bg of ["--bg-base", "--bg-surface"]) {
        const c = contrast(t[fg], t[bg]);
        if (c < 4.5) fails.push(`${name}: ${fg} on ${bg} = ${c.toFixed(2)}`);
      }
  assert.deepEqual(fails, []);
});

test("text on solid fills meets 4.5:1", () => {
  const fails: string[] = [];
  for (const [name, t] of THEMES)
    for (const [fg, bg] of [["--on-primary", "--primary"], ["--on-primary", "--primary-hover"], ["--on-error", "--error"], ["--on-accent", "--accent"]]) {
      const c = contrast(t[fg], t[bg]);
      if (c < 4.5) fails.push(`${name}: ${fg} on ${bg} = ${c.toFixed(2)}`);
    }
  assert.deepEqual(fails, []);
});

test("input boundaries and the focus ring meet 3:1 against the surface they sit on", () => {
  const fails: string[] = [];
  for (const [name, t] of THEMES)
    for (const [fg, bg] of [["--border-strong", "--bg-surface"], ["--border-strong", "--bg-base"], ["--accent", "--bg-base"], ["--accent", "--bg-surface"]]) {
      const c = contrast(t[fg], t[bg]);
      if (c < 3) fails.push(`${name}: ${fg} on ${bg} = ${c.toFixed(2)}`);
    }
  assert.deepEqual(fails, []);
});

test("confidence ramp stays distinguishable from the surface in both themes", () => {
  const fails: string[] = [];
  for (const [name, t] of THEMES)
    for (const fg of ["--conf-high", "--conf-low", "--conf-unknown"]) {
      const c = contrast(t[fg], t["--bg-base"]);
      if (c < 3) fails.push(`${name}: ${fg} on --bg-base = ${c.toFixed(2)}`);
    }
  assert.deepEqual(fails, []);
});

// --- source guard: components must use the tokens, not raw palette colours ------------------------------------------

import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

test("no raw Tailwind palette colours in src (use bg-surface / text-fg / text-success ... from globals.css)", () => {
  const root = new URL("../../src", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
  const raw = /(?<![\w-])(?:[a-z0-9\[\]-]+:)*(?:text|bg|border|ring|divide|outline|shadow|accent|placeholder|fill|stroke|from|to|via)-(?:(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}|white|black)(?:\/\d+)?(?![\w-])/g; // shade required, so our own `text-purple` token is fine
  const allowed = /^(?:[a-z0-9\[\]-]+:)*bg-black\/\d+$/; // canvas scrims are theme-independent on purpose
  const bad: string[] = [];
  for (const f of walk(root).filter((p) => /\.(tsx?|css)$/.test(p) && !p.endsWith("globals.css"))) {
    const text = readFileSync(f, "utf8");
    for (const m of text.matchAll(raw)) if (!allowed.test(m[0])) bad.push(`${f.slice(root.length + 1)}: ${m[0]}`);
  }
  assert.deepEqual(bad, [], `raw palette colours found:\n${bad.slice(0, 20).join("\n")}`);
});
