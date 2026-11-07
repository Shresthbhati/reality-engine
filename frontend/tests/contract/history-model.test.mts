/** Studio undo/redo asks the server first (saved edits survive a reload); the in-memory stack is only a fallback for uncommitted changes. */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  currentVersionOf, pickRedoSource, pickUndoSource, redoTitle, undoTitle, type WorldHistory,
} from "../../src/lib/history/history-model.ts";

const h = (over: Partial<WorldHistory> = {}): WorldHistory => ({
  can_undo: false, can_redo: false, undo_depth: 0, redo_depth: 0, head: "v1", undo_label: null, redo_label: null, ...over,
});

test("a saved edit is undone on the server even when the local stack also has an entry", () => {
  // the local entry for a committed edit would only change the screen, never the saved world
  assert.equal(pickUndoSource(h({ can_undo: true, undo_depth: 2 }), 3), "server");
});

test("the local stack is used only when the server has nothing to undo", () => {
  assert.equal(pickUndoSource(h(), 1), "local");
  assert.equal(pickUndoSource(null, 1), "local");     // server unreachable: do not strand an uncommitted change
  assert.equal(pickUndoSource(h(), 0), "none");
});

test("redo follows the same order", () => {
  assert.equal(pickRedoSource(h({ can_redo: true, redo_depth: 1 }), 2), "server");
  assert.equal(pickRedoSource(h(), 2), "local");
  assert.equal(pickRedoSource(null, 0), "none");
});

test("button titles say what will happen, with counts", () => {
  assert.match(undoTitle(h({ can_undo: true, undo_depth: 1 }), null), /1 saved edit can be undone/);
  assert.match(undoTitle(h({ can_undo: true, undo_depth: 3 }), null), /3 saved edits/);
  assert.equal(undoTitle(h(), null), "Nothing to undo");
  assert.match(undoTitle(h(), "Move wall"), /Move wall/);
  assert.match(redoTitle(h({ can_redo: true, redo_depth: 2 }), null), /2 undone edits/);
  assert.equal(redoTitle(null, null), "Nothing to redo");
});

test("after an undo the newest version is not the current one", () => {
  const versions = [{ id: "v3", isCurrent: false }, { id: "v2", isCurrent: true }, { id: "v1", isCurrent: false }];
  assert.equal(currentVersionOf(versions), "v2");
  assert.equal(currentVersionOf([{ id: "only" }]), "only");
  assert.equal(currentVersionOf([]), null);
});
