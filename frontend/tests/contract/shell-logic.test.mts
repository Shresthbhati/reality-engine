/**
 * Shell navigation + footer derivation: no dead links, honest connection state.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";

import {
  MODE_TABS,
  RAIL_ITEMS,
  activeModeId,
  connectionState,
  evidenceLabel,
  isEntryActive,
  relativeTime,
  worldIdFromPath,
} from "../../src/lib/shell/shell-logic.ts";

/** A static href must resolve to a page.tsx under app/(desktop). */
function routeExists(href: string): boolean {
  const dir = join(import.meta.dirname, "../../src/app/(desktop)", href === "/" ? "" : href);
  return existsSync(join(dir, "page.tsx"));
}

test("every nav href points at a real route; entries without one are null (disabled)", () => {
  for (const e of [...MODE_TABS, ...RAIL_ITEMS]) {
    if (e.href !== null) assert.ok(routeExists(e.href), `${e.id} -> ${e.href} has no page.tsx`);
  }
  assert.equal(MODE_TABS.find((t) => t.id === "compare")?.href, null);
  assert.equal(RAIL_ITEMS.find((t) => t.id === "help")?.href, null);
});

test("Home is active only on /, never on a world workspace", () => {
  const home = RAIL_ITEMS.find((e) => e.id === "home")!;
  assert.equal(isEntryActive(home, "/"), true);
  assert.equal(isEntryActive(home, "/worlds/w-1"), false);
});

test("worlds rail + Explore tab are active on a world route; a disabled entry never is", () => {
  const worlds = RAIL_ITEMS.find((e) => e.id === "worlds")!;
  assert.equal(isEntryActive(worlds, "/worlds/w-1"), true);
  assert.equal(activeModeId("/worlds/w-1"), "explore");
  assert.equal(activeModeId("/reports/new"), "export");
  assert.equal(activeModeId("/"), null);
  assert.equal(isEntryActive(MODE_TABS.find((t) => t.id === "compare")!, "/anything"), false);
});

test("prefix matching does not bleed across sibling names", () => {
  const worlds = RAIL_ITEMS.find((e) => e.id === "worlds")!;
  assert.equal(isEntryActive(worlds, "/worldsfoo"), false);
});

test("worldIdFromPath extracts real ids only", () => {
  assert.equal(worldIdFromPath("/worlds/w-1"), "w-1");
  assert.equal(worldIdFromPath("/worlds/w%201"), "w 1");
  assert.equal(worldIdFromPath("/worlds/new"), null);
  assert.equal(worldIdFromPath("/worlds"), null);
  assert.equal(worldIdFromPath("/"), null);
});

test("connection state comes only from a health result", () => {
  assert.equal(connectionState(null), "checking");
  assert.equal(connectionState({ ok: true }), "online");
  assert.equal(connectionState({ ok: false }), "offline");
});

test("evidence label is omitted unless the API reported a count", () => {
  assert.equal(evidenceLabel(null), null);
  assert.equal(evidenceLabel(undefined), null);
  assert.equal(evidenceLabel(Number.NaN), null);
  assert.equal(evidenceLabel(0), "0 evidence items");
  assert.equal(evidenceLabel(1), "1 evidence item");
  assert.equal(evidenceLabel(4527), "4,527 evidence items");
});

test("relativeTime is honest about unparseable and recent values", () => {
  const now = Date.parse("2026-10-18T12:00:00Z");
  assert.equal(relativeTime(null, now), null);
  assert.equal(relativeTime("not a date", now), null);
  assert.equal(relativeTime("2026-10-18T11:59:30Z", now), "just now");
  assert.equal(relativeTime("2026-10-18T10:00:00Z", now), "2 hours ago");
  assert.equal(relativeTime("2026-10-17T12:00:00Z", now), "1 day ago");
});
