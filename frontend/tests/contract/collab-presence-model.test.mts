/**
 * Contract tests for Collaborative Telepresence & Spatial Markup Model.
 * Verifies trade hex colors, labels, status mappings, coordinate formatting, and relative timestamps.
 * Run with: npm run test:contract
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  formatMarkupCoord,
  formatRelativeTimestamp,
  getStatusLabel,
  getTradeColorHex,
  getTradeLabel,
  type PeerPresence,
  type SpatialMarkup,
} from "../../src/lib/studio/collab-presence-model.ts";

test("getTradeColorHex returns valid distinct hex colors for all architectural trades", () => {
  const arch = getTradeColorHex("architecture");
  const struct = getTradeColorHex("structure");
  const mep = getTradeColorHex("mep");
  const safety = getTradeColorHex("safety");
  const gen = getTradeColorHex("general");

  assert.equal(arch, "#38bdf8");
  assert.equal(struct, "#f59e0b");
  assert.equal(mep, "#10b981");
  assert.equal(safety, "#ef4444");
  assert.equal(gen, "#a855f7");

  // Colors must be distinct
  const unique = new Set([arch, struct, mep, safety, gen]);
  assert.equal(unique.size, 5);
});

test("getTradeLabel and getStatusLabel return human-readable architectural descriptions", () => {
  assert.equal(getTradeLabel("architecture"), "Architecture");
  assert.equal(getTradeLabel("structure"), "Structure");
  assert.equal(getTradeLabel("mep"), "MEP / Services");
  assert.equal(getTradeLabel("safety"), "Safety & Code");
  assert.equal(getTradeLabel("general"), "General Note");

  assert.equal(getStatusLabel("open"), "Open Issue");
  assert.equal(getStatusLabel("in_review"), "In Review");
  assert.equal(getStatusLabel("resolved"), "Resolved");
});

test("formatMarkupCoord accurately formats 3D positions in meters", () => {
  assert.equal(formatMarkupCoord([0, 0, 0]), "(0.00, 0.00, 0.00) m");
  assert.equal(formatMarkupCoord([2.456, 12.891, 3.2]), "(2.46, 12.89, 3.20) m");
  assert.equal(formatMarkupCoord([-1.5, 4.0, -0.25]), "(-1.50, 4.00, -0.25) m");
});

test("formatRelativeTimestamp handles recent and historical dates", () => {
  const now = new Date().toISOString();
  assert.equal(formatRelativeTimestamp(now), "just now");

  const tenMinsAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();
  assert.equal(formatRelativeTimestamp(tenMinsAgo), "10m ago");

  const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
  assert.equal(formatRelativeTimestamp(twoHoursAgo), "2h ago");

  assert.equal(formatRelativeTimestamp("invalid-date"), "Recently");
});

test("PeerPresence and SpatialMarkup object shapes conform to contract", () => {
  const peer: PeerPresence = {
    client_id: "peer-01",
    user_name: "Lead Architect",
    role: "architect",
    color: "#38bdf8",
    camera_position: [1.0, 2.0, 1.8],
    look_at: [1.0, 5.0, 1.8],
    field_of_view: 60,
    last_heartbeat: Date.now() / 1000,
    is_active: true,
  };
  assert.equal(peer.client_id, "peer-01");
  assert.equal(peer.is_active, true);

  const markup: SpatialMarkup = {
    markup_id: "mk-01",
    world_id: "world-demo",
    markup_type: "pin",
    trade: "mep",
    title: "Duct collision",
    description: "Intersects column W14x90",
    author_name: "Coordinator",
    author_id: "usr-9",
    position: [2.5, 3.0, 2.7],
    normal: [0, 0, 1],
    stroke_points: [],
    camera_viewpoint: { position: [2.5, 1.0, 1.8], fov: 60 },
    color: "#10b981",
    status: "open",
    created_at: new Date().toISOString(),
  };
  assert.equal(markup.trade, "mep");
  assert.equal(markup.status, "open");
  assert.equal(markup.stroke_points.length, 0);
});
