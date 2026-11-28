/**
 * Contract tests for Blueprint Task 8.5: User Corrections Model & Payload Contracts.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  buildReclassifyPayload,
  buildSplitPayload,
  buildMergePayload,
  type CorrectionResult,
  type ReclassifyParams,
  type SplitParams,
  type MergeParams,
} from "../../src/lib/api/corrections-model.ts";

describe("User Corrections Model Contract", () => {
  it("builds valid reclassify payload", () => {
    const params: ReclassifyParams = {
      worldId: "wld_test",
      entityId: "ent_01",
      newClass: "furniture/seating/armchair",
      note: "verified armchair",
      parentVersionId: "v_123",
    };
    const payload = buildReclassifyPayload(params);
    assert.equal(payload.entity_id, "ent_01");
    assert.equal(payload.new_class, "furniture/seating/armchair");
    assert.equal(payload.note, "verified armchair");
    assert.equal(payload.parent_version_id, "v_123");
  });

  it("throws when reclassify parameters are incomplete", () => {
    assert.throws(() => buildReclassifyPayload({ worldId: "w", entityId: "", newClass: "chair", parentVersionId: "v1" }));
    assert.throws(() => buildReclassifyPayload({ worldId: "w", entityId: "e1", newClass: "", parentVersionId: "v1" }));
  });

  it("builds valid split payload with default axis and ratio", () => {
    const params: SplitParams = {
      worldId: "wld_test",
      entityId: "ent_02",
      parentVersionId: "v_123",
    };
    const payload = buildSplitPayload(params);
    assert.equal(payload.entity_id, "ent_02");
    assert.equal(payload.split_axis, "x");
    assert.equal(payload.split_ratio, 0.5);
  });

  it("throws when split ratio is out of range", () => {
    assert.throws(() => buildSplitPayload({ worldId: "w", entityId: "e1", splitRatio: 0.01, parentVersionId: "v1" }));
    assert.throws(() => buildSplitPayload({ worldId: "w", entityId: "e1", splitRatio: 0.99, parentVersionId: "v1" }));
  });

  it("builds valid merge payload", () => {
    const params: MergeParams = {
      worldId: "wld_test",
      primaryEntityId: "ent_primary",
      secondaryEntityIds: ["ent_sec_1", "ent_sec_2"],
      newName: "Combined Sofa",
      parentVersionId: "v_123",
    };
    const payload = buildMergePayload(params);
    assert.equal(payload.primary_entity_id, "ent_primary");
    assert.equal(payload.secondary_entity_ids.length, 2);
    assert.equal(payload.new_name, "Combined Sofa");
  });

  it("throws when merge secondary entities list is empty", () => {
    assert.throws(() => buildMergePayload({ worldId: "w", primaryEntityId: "e1", secondaryEntityIds: [], parentVersionId: "v1" }));
  });

  it("refuses to build any correction payload without a base version", () => {
    assert.throws(() => buildReclassifyPayload({ worldId: "w", entityId: "e1", newClass: "x/y", parentVersionId: "" }), /parentVersionId is required/);
    assert.throws(() => buildSplitPayload({ worldId: "w", entityId: "e1", parentVersionId: "" }), /parentVersionId is required/);
    assert.throws(() => buildMergePayload({ worldId: "w", primaryEntityId: "e1", secondaryEntityIds: ["e2"], parentVersionId: "" }), /parentVersionId is required/);
  });

  it("conforms to CorrectionResult contract", () => {
    const mockResult: CorrectionResult = {
      version_id: "v_new_123",
      world_id: "wld_test",
      entity_id: "ent_01",
      changed_fields: ["reclassify"],
      duplicate: false,
      operation: "reclassify",
    };
    assert.equal(mockResult.version_id, "v_new_123");
    assert.equal(mockResult.duplicate, false);
    assert.equal(mockResult.operation, "reclassify");
  });
});
