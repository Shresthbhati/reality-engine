import test from "node:test";
import assert from "node:assert/strict";
import { parsePartsTree, parseAttributes } from "../../src/lib/inspector/parts-attributes-model.ts";
import type { Entity } from "../../src/types/worldir";

test("Parts Tree Parser: builds hierarchy strictly from part_of", () => {
  const entities: Record<string, Entity> = {
    "almirah-1": {
      id: "almirah-1",
      type: "furniture/storage/almirah",
      relationships: [],
    } as unknown as Entity,
    "handle-1": {
      id: "handle-1",
      type: "furniture/storage/handle",
      semantic_labels: ["Right Handle"],
      relationships: [
        { kind: "part_of", target_id: "almirah-1" },
      ],
    } as unknown as Entity,
    "lock-1": {
      id: "lock-1",
      type: "furniture/storage/lock",
      semantic_labels: ["Keyhole Lock"],
      relationships: [
        { kind: "part_of", target_id: "almirah-1" },
      ],
    } as unknown as Entity,
    "keyhole-screw-1": {
      id: "keyhole-screw-1",
      type: "hardware/screw",
      semantic_labels: ["Lock Screw"],
      relationships: [
        { kind: "part_of", target_id: "lock-1" },
      ],
    } as unknown as Entity,
    "cup-1": {
      id: "cup-1",
      type: "misc/cup",
      relationships: [
        { kind: "on", target_id: "almirah-1" }, // NOT a part!
      ],
    } as unknown as Entity,
  };

  const tree = parsePartsTree("almirah-1", entities);
  assert.equal(tree.length, 2, "Expected 2 direct parts (handle and lock)");

  const handleNode = tree.find((n) => n.id === "handle-1");
  assert.ok(handleNode);
  assert.equal(handleNode.label, "Right Handle");
  assert.equal(handleNode.children.length, 0);

  const lockNode = tree.find((n) => n.id === "lock-1");
  assert.ok(lockNode);
  assert.equal(lockNode.label, "Keyhole Lock");
  assert.equal(lockNode.children.length, 1, "Lock should have 1 sub-part (screw)");
  assert.equal(lockNode.children[0].id, "keyhole-screw-1");
});

test("Parts Tree Parser: handles cycles gracefully", () => {
  const entities: Record<string, Entity> = {
    "part-a": {
      id: "part-a",
      type: "component",
      relationships: [{ kind: "part_of", target_id: "part-b" }],
    } as unknown as Entity,
    "part-b": {
      id: "part-b",
      type: "component",
      relationships: [{ kind: "part_of", target_id: "part-a" }],
    } as unknown as Entity,
  };

  const tree = parsePartsTree("part-a", entities);
  assert.ok(Array.isArray(tree));
  // Must terminate and not cause infinite recursion
});

test("Attributes Parser: visual, user_stated, and unknown basis rules", () => {
  const entity: Entity = {
    id: "lock-1",
    type: "opening/door/lock",
    extra_fields: {
      attributes: {
        lock_type: {
          value: "key",
          p: 0.94,
          basis: "visual",
          obs: ["img01#3"],
        },
        handle_type: {
          value: null,
          basis: "visual",
        },
        material: {
          value: "brass",
          basis: "user_stated",
        },
        missing_attr: {
          value: "unknown",
        },
      },
    },
  } as unknown as Entity;

  const rows = parseAttributes(entity);
  assert.equal(rows.length, 4);

  const lockRow = rows.find((r) => r.name === "lock_type");
  assert.ok(lockRow);
  assert.equal(lockRow.value, "key");
  assert.equal(lockRow.displayValue, "key");
  assert.equal(lockRow.basis, "visual");
  assert.equal(lockRow.confidence, 0.94);
  assert.deepEqual(lockRow.obsIds, ["img01#3"]);

  const handleRow = rows.find((r) => r.name === "handle_type");
  assert.ok(handleRow);
  assert.equal(handleRow.value, null);
  assert.equal(handleRow.displayValue, "Unknown — not visible in any photo");
  assert.equal(handleRow.basis, "unknown");

  const matRow = rows.find((r) => r.name === "material");
  assert.ok(matRow);
  assert.equal(matRow.value, "brass");
  assert.equal(matRow.basis, "user_stated");

  const missingRow = rows.find((r) => r.name === "missing_attr");
  assert.ok(missingRow);
  assert.equal(missingRow.value, null);
  assert.equal(missingRow.displayValue, "Unknown — not visible in any photo");
  assert.equal(missingRow.basis, "unknown");
});

test("Attributes Parser: handles empty and null entities safely", () => {
  assert.deepEqual(parseAttributes(null), []);
  assert.deepEqual(parseAttributes(undefined), []);
  assert.deepEqual(parseAttributes({ id: "e1" } as Entity), []);
});
