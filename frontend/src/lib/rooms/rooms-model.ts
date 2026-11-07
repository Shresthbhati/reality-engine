/**
 * Rooms review model (milestone M4, decision: the app PROPOSES with a confidence, the user CONFIRMS).
 * Pure and dependency-free so it is testable under `node --test`. The wording never states a proposal as a fact, never calls an
 * element that was not seen this time "removed", and every confirmation says what will change and what will not.
 */

export interface DoorJoinProposal {
  door_a: string;
  door_b: string;
  room_a: string;
  room_b: string;
  width_diff_m: number;
  height_diff_m: number;
  confidence: number;
  state: "proposed";
  note: string;
}

export interface RevisitProposal {
  room_a: string;
  room_b: string;
  confidence: number;
  state: "proposed";
  evidence: { dimensions: number | null; ceiling_height: number | null; openings: number | null; visual?: number | null };
  visual_detail?: { score: number; best_inliers: number; pairs_tested: number };
  note: string;
}

export interface RoomProposals {
  door_joins: DoorJoinProposal[];
  revisits: RevisitProposal[];
}

export type ChangeState = "matched" | "updated" | "uncertain" | "conflicting" | "new";

export interface RoomChange {
  kind: string;
  state: ChangeState;
  old: string | null;
  new: string | null;
  label: string | null;
  size_diff?: number;
}

export interface RoomChangeSet {
  old_room: string;
  new_room: string;
  state: "proposed";
  summary: Partial<Record<ChangeState, number>>;
  changes: RoomChange[];
  note: string;
}

export interface LostConnection {
  old_door: string;
  other_door: string;
  reason: string;
}

/** `boot2` -> "Room 2"; a bare `boot` (a single capture) is Room 1. Anything else is shown as it is. */
export function roomName(prefix: string): string {
  const m = /^boot(\d*)$/.exec(prefix);
  if (!m) return prefix;
  return `Room ${m[1] === "" ? 1 : Number(m[1])}`;
}

export function confidenceBand(c: number): { label: "High" | "Medium" | "Low"; tone: "good" | "warn" | "bad" } {
  if (c >= 0.8) return { label: "High", tone: "good" };
  if (c >= 0.6) return { label: "Medium", tone: "warn" };
  return { label: "Low", tone: "bad" };
}

export function hasProposals(p: RoomProposals | null | undefined): boolean {
  return !!p && (p.door_joins.length > 0 || p.revisits.length > 0);
}

const cm = (m: number) => `${Math.round(m * 100)} cm`;

export function joinSentence(j: DoorJoinProposal): string {
  const diff = j.width_diff_m < 0.005 ? "the same width" : `widths differ by ${cm(j.width_diff_m)}`;
  return `A door in ${roomName(j.room_a)} and a door in ${roomName(j.room_b)} could be the same opening (${diff}).`;
}

export function revisitSentence(r: RevisitProposal): string {
  const photos = r.visual_detail ? ", and their photos share matching features" : "";
  return `${roomName(r.room_b)} could be a second visit to ${roomName(r.room_a)}: its measured size, ceiling height and openings are similar${photos}.`;
}

/** The photo-matching evidence in words, or null when no photos could be compared (never shown as a zero). */
export function visualEvidenceText(r: RevisitProposal): string | null {
  const d = r.visual_detail;
  if (!d) return null;
  return `${d.best_inliers} verified feature matches in the best of ${d.pairs_tested} photo pair${d.pairs_tested === 1 ? "" : "s"}.`;
}

export function joinConfirmation(roomA: string, roomB: string): string {
  return `${roomName(roomB)} will be moved so its door meets ${roomName(roomA)}'s door, with the two rooms on opposite sides. Nothing else changes: no wall or object is re-measured.`;
}

export function revisitConfirmation(oldRoom: string, newRoom: string): string {
  return `${roomName(newRoom)} replaces ${roomName(oldRoom)}. ${roomName(oldRoom)}'s walls, openings and furniture are removed from the apartment; ` +
    `door connections are re-made through the matching door of ${roomName(newRoom)} where there is one. The earlier version stays in the history.`;
}

const STATE_LABEL: Record<ChangeState, string> = {
  matched: "Same as before",
  updated: "Changed a little",
  conflicting: "Different: check this",
  new: "New this visit",
  uncertain: "Not seen this time (kept as unknown)",
};

export function changeStateLabel(s: ChangeState): string {
  return STATE_LABEL[s];
}

/** Worst first: what needs a person's eye comes before what is simply the same. */
export const REVIEW_ORDER: ChangeState[] = ["conflicting", "uncertain", "new", "updated", "matched"];

const HEADLINE_WORD: Record<ChangeState, string> = {
  conflicting: "different", uncertain: "not seen", new: "new", updated: "changed", matched: "same",
};

export function changeHeadline(summary: Partial<Record<ChangeState, number>>): string {
  const parts = REVIEW_ORDER.filter((s) => (summary[s] ?? 0) > 0).map((s) => `${summary[s]} ${HEADLINE_WORD[s]}`);
  return parts.length ? parts.join(", ") : "Nothing to compare";
}

export function groupChanges(changes: RoomChange[]): Array<{ state: ChangeState; items: RoomChange[] }> {
  return REVIEW_ORDER.map((state) => ({ state, items: changes.filter((c) => c.state === state) })).filter((g) => g.items.length > 0);
}

/** `boot1-opening-01` -> "Room 1" (the room a door belongs to). */
export function roomOfEntity(id: string): string {
  return roomName(id.split("-", 1)[0]);
}

export function lostConnectionText(l: LostConnection): string {
  return `The connection to ${roomOfEntity(l.other_door)} could not be kept: ${l.reason}. Join the rooms again from the proposals if this is the same door.`;
}
