import test from "node:test";
import assert from "node:assert/strict";
import { HistoryStack } from "../../src/lib/viewport/history-stack.ts";

test("HistoryStack pushes actions and tracks undo/redo capacity", () => {
  const stack = new HistoryStack(10);
  assert.equal(stack.canUndo(), false);
  assert.equal(stack.canRedo(), false);
  assert.equal(stack.getUndoCount(), 0);

  let pos = { x: 0, y: 0, z: 0 };
  stack.push({
    type: "transform",
    description: "Move entity from (0,0,0) to (1,2,3)",
    entityId: "ent-1",
    before: { x: 0, y: 0, z: 0 },
    after: { x: 1, y: 2, z: 3 },
    apply: (state) => { pos = state; },
  });

  assert.equal(stack.canUndo(), true);
  assert.equal(stack.canRedo(), false);
  assert.equal(stack.getUndoCount(), 1);
  assert.equal(stack.peekUndo()?.description, "Move entity from (0,0,0) to (1,2,3)");

  // Undo reverts state to 'before'
  const undone = stack.undo();
  assert.equal(undone?.entityId, "ent-1");
  assert.deepEqual(pos, { x: 0, y: 0, z: 0 });
  assert.equal(stack.canUndo(), false);
  assert.equal(stack.canRedo(), true);
  assert.equal(stack.getRedoCount(), 1);

  // Redo reapplies state to 'after'
  const redone = stack.redo();
  assert.equal(redone?.entityId, "ent-1");
  assert.deepEqual(pos, { x: 1, y: 2, z: 3 });
  assert.equal(stack.canUndo(), true);
  assert.equal(stack.canRedo(), false);
});

test("HistoryStack respects maxDepth ceiling and clears redo on new push", () => {
  const stack = new HistoryStack(3);
  let value = 0;

  for (let i = 1; i <= 5; i++) {
    stack.push({
      type: "geometry",
      description: `Step ${i}`,
      entityId: "ent-2",
      before: i - 1,
      after: i,
      apply: (v) => { value = v; },
    });
  }

  // Max depth is 3, so only steps 3, 4, 5 are kept
  assert.equal(stack.getUndoCount(), 3);
  assert.equal(stack.peekUndo()?.description, "Step 5");

  stack.undo();
  assert.equal(value, 4);
  assert.equal(stack.canRedo(), true);

  // Pushing a new action clears redo stack
  stack.push({
    type: "geometry",
    description: "Step 6 (branch)",
    entityId: "ent-2",
    before: 4,
    after: 10,
    apply: (v) => { value = v; },
  });

  assert.equal(stack.canRedo(), false);
  assert.equal(stack.peekUndo()?.description, "Step 6 (branch)");
});

test("HistoryStack notifies subscribers on changes", () => {
  const stack = new HistoryStack(10);
  let notifications = 0;

  const unsubscribe = stack.subscribe(() => {
    notifications++;
  });

  stack.push({
    type: "classification",
    description: "Change type to furniture",
    entityId: "ent-3",
    before: "debris",
    after: "furniture",
    apply: () => {},
  });
  assert.equal(notifications, 1);

  stack.undo();
  assert.equal(notifications, 2);

  stack.redo();
  assert.equal(notifications, 3);

  unsubscribe();
  stack.undo();
  assert.equal(notifications, 3); // No further calls after unsubscribe
});
