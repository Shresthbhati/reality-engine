/**
 * Change-proposal review logic: ordering, default selection, and the acceptance request/blockers.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  confidenceText,
  defaultSelection,
  planAcceptance,
  sortAssessments,
  summaryLine,
} from "../../src/lib/proposals/proposal-model.ts";
import type { ProposalAssessmentDto } from "../../src/lib/api/proposals.ts";

function a(entity_id: string, verdict: ProposalAssessmentDto["verdict"], over: Partial<ProposalAssessmentDto> = {}): ProposalAssessmentDto {
  return {
    entity_id, kind: "modified", verdict, reasons: [], categories: [], new_evidence: [],
    old_confidence: null, new_confidence: null, ...over,
  };
}

const ITEMS = [
  a("window", "supported"),
  a("door", "needs_review"),
  a("wall-b", "conflict"),
  a("wall-a", "needs_review"),
  a("beam", "supported"),
];

test("conflicts sort first, then needs_review, then supported, stable by id", () => {
  assert.deepEqual(sortAssessments(ITEMS).map((x) => x.entity_id), ["wall-b", "door", "wall-a", "beam", "window"]);
  assert.equal(ITEMS[0].entity_id, "window", "input is not mutated");
});

test("default selection is exactly the supported changes (what the server applies with no selection)", () => {
  assert.deepEqual(defaultSelection(ITEMS), ["beam", "window"]);
  assert.deepEqual(defaultSelection([]), []);
});

test("a supported-only selection is sendable and carries no overrides", () => {
  const p = planAcceptance(ITEMS, ["window", "beam"], "");
  assert.deepEqual(p.entity_ids, ["beam", "window"]);
  assert.deepEqual(p.override_conflicts, []);
  assert.deepEqual(p.blockers, []);
});

test("selecting a conflict makes it an override and demands a reason", () => {
  const blocked = planAcceptance(ITEMS, ["window", "wall-b"], "   ");
  assert.deepEqual(blocked.override_conflicts, ["wall-b"]);
  assert.equal(blocked.blockers.length, 1);
  assert.match(blocked.blockers[0], /needs a written reason/);
  const ok = planAcceptance(ITEMS, ["window", "wall-b"], "  site survey  ");
  assert.deepEqual(ok.blockers, []);
  assert.equal(ok.reason, "site survey");
  assert.deepEqual(ok.overridden, ["wall-b"]);
});

test("needs_review changes are applied when named, without being treated as overrides", () => {
  const p = planAcceptance(ITEMS, ["door"], "");
  assert.deepEqual(p.entity_ids, ["door"]);
  assert.deepEqual(p.override_conflicts, []);
  assert.deepEqual(p.blockers, []);
});

test("empty selection is blocked; unknown ids and duplicates are dropped", () => {
  assert.match(planAcceptance(ITEMS, [], "").blockers[0], /at least one change/);
  const p = planAcceptance(ITEMS, ["window", "window", "ghost"], "");
  assert.deepEqual(p.entity_ids, ["window"]);
});

test("summary line and confidence text", () => {
  assert.equal(summaryLine({ supported: 1, needs_review: 2, conflict: 1, changes: 4, diff: {}, categories: {} }),
    "1 conflict · 2 need review · 1 supported");
  assert.equal(summaryLine({ supported: 0, needs_review: 0, conflict: 0, changes: 0, diff: {}, categories: {} }), "No changes");
  assert.equal(confidenceText(a("x", "conflict", { old_confidence: 0.9, new_confidence: 0.4 })), "0.90 → 0.40");
  assert.equal(confidenceText(a("x", "supported", { new_confidence: 0.8 })), "new 0.80");
  assert.equal(confidenceText(a("x", "needs_review", { old_confidence: 0.7 })), "was 0.70");
  assert.equal(confidenceText(a("x", "supported")), null);
});
