/**
 * Consumer wording stays free of engine terms.
 *
 * Checks lib/copy.ts and every file under src/components/simple (Simple-mode screens, added from Phase 1) for the
 * engine terms in BANNED_IN_SIMPLE. NAV.sessions / NAV.evidence are Pro-only entries and are excluded.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { BANNED_IN_SIMPLE, NAV, allConsumerStrings, photosPlaced } from "../../src/lib/copy.ts";

/** Whole-word, plural-tolerant, case-insensitive: "menu" does not trip "ENU", "sessions" does trip "session". */
function hits(text: string): string[] {
  return BANNED_IN_SIMPLE.filter((t) => new RegExp(`\b${t.replace(/ /g, "\s+")}s?\b`, "i").test(text));
}

test("consumer copy contains no engine terms", () => {
  const proOnly = new Set<string>([NAV.sessions, NAV.evidence]);
  for (const s of allConsumerStrings()) {
    if (proOnly.has(s)) continue;
    assert.deepEqual(hits(s), [], `"${s}" contains an engine term`);
  }
});

test("photosPlaced pluralises", () => {
  assert.equal(photosPlaced(1, 1), "1 of 1 photo placed");
  assert.equal(photosPlaced(7, 10), "7 of 10 photos placed");
});

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

test("Simple-mode components contain no engine terms in string literals", () => {
  const dir = new URL("../../src/components/simple", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
  if (!existsSync(dir)) return; // no Simple-mode screens yet (Phase 1+)
  for (const f of walk(dir).filter((p) => p.endsWith(".tsx"))) {
    const literals = readFileSync(f, "utf8").match(/(["'`])(?:(?!\1)[^\\n]|\.){3,}\1|>[^<>{}]{3,}</g) ?? [];
    for (const lit of literals) {
      if (/^["'`](?:@|\.|\/|[a-z-]+$)/.test(lit)) continue; // imports, paths, class names
      assert.deepEqual(hits(lit), [], `${f}: ${lit}`);
    }
  }
});
