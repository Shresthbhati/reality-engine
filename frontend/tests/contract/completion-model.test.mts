/**
 * Complete-interior pure logic: request body, validation, honest labels, diff summaries.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { effectiveEntrance, templateAdjacency, templateRows } from "../../src/lib/procedural/apartment-model.ts";
import {
  addedAllGenerated,
  diffSummary,
  footprintLine,
  proposalPlanSvg,
  provenanceText,
  sortProposals,
  statusBadge,
  toProposalBody,
  unitText,
  validateCompletion,
  type CompletionDiff,
  type CompletionDraft,
  type CompletionProposalRow,
} from "../../src/lib/procedural/completion-model.ts";

function draft(over: Partial<CompletionDraft> = {}): CompletionDraft {
  return { name: "interior", program: "studio", rows: [], entrance: "", adjacency: [], seed: 0, alternatives: 2, ...over };
}

test("template programs send only the template key", () => {
  assert.deepEqual(toProposalBody(draft({ name: " flat " })), { name: "flat", seed: 0, alternatives: 2, template: "studio" });
  assert.equal(toProposalBody(draft({ program: "one_bed" })).template, "one_bed");
});

test("custom program sends rooms, entrance and tuple adjacency, never a footprint", () => {
  const rows = templateRows("studio");
  const body = toProposalBody(draft({
    program: "custom", rows, entrance: effectiveEntrance(rows, ""), adjacency: templateAdjacency(rows), seed: 3,
  }));
  assert.equal(body.template, undefined);
  assert.equal(body.rooms?.length, rows.length);
  assert.equal(body.entrance, "hallway");
  assert.ok(body.adjacency?.every((a) => a.length === 3));
  assert.ok(!("footprint" in body));
  assert.equal(body.seed, 3);
});

test("validation: templates need only name/seed/alternatives; custom needs rooms", () => {
  assert.deepEqual(validateCompletion(draft()), []);
  assert.ok(validateCompletion(draft({ name: " " })).some((m) => /name/i.test(m)));
  assert.ok(validateCompletion(draft({ seed: 1.5 })).some((m) => /Seed/.test(m)));
  assert.ok(validateCompletion(draft({ alternatives: 4 })).some((m) => /1 to 3/.test(m)));
  const none = validateCompletion(draft({ program: "custom" }));
  assert.ok(none.some((m) => /at least one room/.test(m)));
  const rows = templateRows("studio");
  assert.deepEqual(validateCompletion(draft({ program: "custom", rows, entrance: effectiveEntrance(rows, "") })), []);
  rows[0].min_area = 99;
  assert.ok(validateCompletion(draft({ program: "custom", rows, entrance: "hallway" })).some((m) => /larger than max/.test(m)));
});

test("units are stated as returned: model units are not metres", () => {
  assert.match(unitText("model_units"), /not metres/);
  assert.match(unitText("m"), /metric or authored/);
  assert.match(unitText(undefined), /not reported/);
});

test("footprint line states the bounding-box method and the unit", () => {
  const fp = { x_min: 0, z_min: 0, z_max: 5, width: 6, depth: 5.004, floor_y: 0.1, method: "axis_aligned_bbox" };
  const t = footprintLine(fp, "model_units");
  assert.match(t, /^6 x 5 model units, axis-aligned bounding box/);
  assert.match(footprintLine({ ...fp, method: "other" }, "m"), /method other/);
});

const diff = (over: Partial<CompletionDiff> = {}): CompletionDiff => ({
  proposal_id: "cmp_1", status: "proposed", unit: "model_units",
  footprint: { x_min: 0, z_min: 0, z_max: 5, width: 6, depth: 5, floor_y: 0, method: "axis_aligned_bbox" },
  generated_label: "x", added: ["a", "b", "c"], changed: [], removed: [],
  counts: { added: 3, changed: 0, removed: 0 },
  provenance: { existing: { RECONSTRUCTED: 2 }, added: { GENERATED: 3 } }, ...over,
});

test("diff summary and provenance text come from the server counts", () => {
  assert.equal(diffSummary(diff()), "3 added, 0 changed, 0 removed");
  assert.equal(provenanceText({ RECONSTRUCTED: 2, OBSERVED: 1 }), "1 OBSERVED, 2 RECONSTRUCTED");
  assert.equal(provenanceText({}), "none");
});

test("added entities that are not all GENERATED are flagged, not hidden", () => {
  assert.equal(addedAllGenerated(diff()), true);
  assert.equal(addedAllGenerated(diff({ provenance: { existing: {}, added: { GENERATED: 2, OBSERVED: 1 } } })), false);
  assert.equal(addedAllGenerated(diff({ provenance: { existing: {}, added: {} } })), false);
});

test("badges never call a proposal observed or reconstructed", () => {
  for (const s of ["proposed", "accepted", "rejected"]) {
    assert.doesNotMatch(statusBadge(s), /OBSERVED|RECONSTRUCTED/);
  }
  assert.equal(statusBadge("proposed"), "PROPOSED");
  assert.match(statusBadge("accepted"), /GENERATED/);
});

test("proposals: pending first, newest batch first, then alternative order", () => {
  const row = (id: string, status: string, created: string, alt: number) =>
    ({ id, status, created_at: created, alt_index: alt }) as CompletionProposalRow;
  const out = sortProposals([
    row("old0", "proposed", "2026-01-01", 0), row("done", "accepted", "2026-03-01", 0),
    row("new1", "proposed", "2026-02-01T10:00:00.900", 1), row("new0", "proposed", "2026-02-01T10:00:00.100", 0),
  ]).map((r) => r.id);
  assert.deepEqual(out, ["new0", "new1", "old0", "done"]);
});

test("proposalPlanSvg scales rooms and the shell outline together; null without rects", () => {
  const row = {
    id: "p", status: "proposed", alt_index: 0, seed: 0, summary: { footprint: { width: 6, depth: 5 }, unit: "m" },
    plan: {
      rects: { living: [0, 0, 4, 5], bath: [4, 0, 2, 5] },
      kinds: { living: "living", bath: "bathroom" }, footprint: [6, 5], provenance: "GENERATED",
    },
  } as unknown as CompletionProposalRow;
  const svg = proposalPlanSvg(row, 120)!;
  assert.equal(svg.rooms.length, 2);
  assert.equal(svg.width, 120);
  assert.equal(svg.shell.w, 120);
  assert.equal(svg.shell.h, 100);
  assert.equal(svg.rooms[0].w, 80);
  assert.equal(proposalPlanSvg({ ...row, plan: null } as CompletionProposalRow), null);
  assert.equal(proposalPlanSvg({ ...row, plan: { ...row.plan!, rects: {} } } as CompletionProposalRow), null);
});
