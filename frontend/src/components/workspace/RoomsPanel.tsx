"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, DoorOpen, GitCompare, Loader2, RefreshCw, X } from "lucide-react";
import {
  acceptRevisit, changeHeadline, changeStateLabel, confidenceBand, connectRooms, getRoomChangeSet, getRoomProposals, groupChanges,
  hasProposals, joinConfirmation, joinSentence, lostConnectionText, revisitConfirmation, revisitSentence, roomName,
  type DoorJoinProposal, type LostConnection, type RevisitProposal, type RoomChangeSet, type RoomProposals,
} from "@/lib/api/rooms";

type Confirm = { kind: "join"; join: DoorJoinProposal } | { kind: "revisit"; oldRoom: string; newRoom: string };

const TONE: Record<"good" | "warn" | "bad", string> = { good: "var(--conf-high, #4caf62)", warn: "var(--warning, #e0a458)", bad: "var(--conf-low, #e0524a)" };

function Badge({ confidence }: { confidence: number }) {
  const band = confidenceBand(confidence);
  return (
    <span
      title={`Agreement of measured properties: ${Math.round(confidence * 100)}%. This is not a probability.`}
      className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold"
      style={{ color: TONE[band.tone], border: `1px solid ${TONE[band.tone]}` }}
    >
      {band.label} match
    </span>
  );
}

function Bar({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="flex items-center gap-2 text-[10px] text-[var(--text-secondary)]">
      <span className="w-20 shrink-0">{label}</span>
      <div className="h-1.5 flex-1 rounded bg-[var(--surface-sunken)]">
        <div className="h-full rounded" style={{ width: `${Math.round((value ?? 0) * 100)}%`, background: "var(--accent)" }} />
      </div>
      <span className="w-8 text-right tabular-nums">{value === null ? "n/a" : `${Math.round(value * 100)}%`}</span>
    </div>
  );
}

export default function RoomsPanel({ worldId, onClose, onChanged }: { worldId: string; onClose: () => void; onChanged?: () => void }) {
  const [proposals, setProposals] = useState<RoomProposals | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState<Confirm | null>(null);
  const [review, setReview] = useState<{ key: string; oldRoom: string; newRoom: string; data: RoomChangeSet | null; error?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<{ text: string; lost: LostConnection[] } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setProposals(await getRoomProposals(worldId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load room proposals.");
    } finally {
      setLoading(false);
    }
  }, [worldId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function openReview(r: RevisitProposal, oldRoom: string, newRoom: string) {
    const key = `${r.room_a}|${r.room_b}`;
    setConfirm(null);
    setReview({ key, oldRoom, newRoom, data: null });
    try {
      setReview({ key, oldRoom, newRoom, data: await getRoomChangeSet(worldId, oldRoom, newRoom) });
    } catch (e) {
      setReview({ key, oldRoom, newRoom, data: null, error: e instanceof Error ? e.message : "Could not compare the rooms." });
    }
  }

  async function runConfirmed() {
    if (!confirm) return;
    setBusy(true);
    setError(null);
    try {
      if (confirm.kind === "join") {
        await connectRooms(worldId, confirm.join.door_a, confirm.join.door_b);
        setDone({ text: `${roomName(confirm.join.room_b)} was moved to meet ${roomName(confirm.join.room_a)}.`, lost: [] });
      } else {
        const res = await acceptRevisit(worldId, confirm.oldRoom, confirm.newRoom);
        setDone({ text: `${roomName(confirm.newRoom)} now replaces ${roomName(confirm.oldRoom)}.`, lost: res.lost_connections ?? [] });
      }
      setConfirm(null);
      setReview(null);
      onChanged?.();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The change was not applied.");
    } finally {
      setBusy(false);
    }
  }

  const visibleJoins = (proposals?.door_joins ?? []).filter((j) => !dismissed.has(`${j.door_a}|${j.door_b}`));
  const visibleRevisits = (proposals?.revisits ?? []).filter((r) => !dismissed.has(`${r.room_a}|${r.room_b}`));
  const dismiss = (key: string) => setDismissed((d) => new Set(d).add(key));

  return (
    <div
      role="dialog"
      aria-labelledby="rooms-panel-title"
      className="absolute top-4 right-4 z-20 w-[26rem] max-h-[85vh] overflow-y-auto rounded-lg p-4 flex flex-col gap-3 shadow-2xl backdrop-blur-md"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
    >
      <div className="flex items-center justify-between pb-1 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <DoorOpen className="w-4 h-4 text-[var(--accent)]" />
          <h3 id="rooms-panel-title" className="text-sm font-semibold">Rooms: joins and second visits</h3>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" onClick={() => void load()} aria-label="Refresh proposals" className="p-1 rounded hover:bg-[var(--surface-hover)]">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} style={{ color: "var(--text-tertiary)" }} />
          </button>
          <button type="button" onClick={onClose} aria-label="Close rooms panel" className="p-1 rounded hover:bg-[var(--surface-hover)]">
            <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
          </button>
        </div>
      </div>

      <p className="text-[11px] leading-snug text-[var(--text-secondary)]">
        These are proposals from measured sizes. Nothing changes until you confirm it.
      </p>

      {error && (
        <div role="alert" className="flex items-start gap-2 rounded-md p-2 text-xs" style={{ background: "var(--surface-sunken)", border: "1px solid var(--conf-low, #e0524a)" }}>
          <AlertCircle className="w-4 h-4 shrink-0" style={{ color: "var(--conf-low, #e0524a)" }} />
          <span>{error}</span>
        </div>
      )}

      {done && (
        <div role="status" className="rounded-md p-2 text-xs flex flex-col gap-1.5" style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)" }}>
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" style={{ color: "var(--conf-high, #4caf62)" }} />
            <span>{done.text}</span>
          </div>
          {done.lost.map((l) => (
            <div key={l.old_door} className="flex items-start gap-2 text-[var(--text-secondary)]">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--warning, #e0a458)" }} />
              <span>{lostConnectionText(l)}</span>
            </div>
          ))}
        </div>
      )}

      {loading && !proposals && (
        <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]"><Loader2 className="w-4 h-4 animate-spin" /> Looking for matches…</div>
      )}

      {proposals && !hasProposals(proposals) && !loading && (
        <p className="text-xs text-[var(--text-secondary)]">
          No matches to review. When rooms captured in separate sessions look related, they are proposed here.
        </p>
      )}

      {visibleJoins.length > 0 && (
        <section aria-label="Door joins" className="flex flex-col gap-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-tertiary)]">Join rooms through a door</h4>
          {visibleJoins.map((j) => {
            const key = `${j.door_a}|${j.door_b}`;
            const open = confirm?.kind === "join" && confirm.join.door_a === j.door_a && confirm.join.door_b === j.door_b;
            return (
              <div key={key} className="rounded-md p-2 flex flex-col gap-2" style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)" }}>
                <div className="flex items-start justify-between gap-2">
                  <p className="text-xs leading-snug">{joinSentence(j)}</p>
                  <Badge confidence={j.confidence} />
                </div>
                {open ? (
                  <div ref={(el) => el?.scrollIntoView({ block: "nearest" })} className="flex flex-col gap-2">
                    <p className="text-[11px] leading-snug text-[var(--text-secondary)]">{joinConfirmation(j.room_a, j.room_b)}</p>
                    <div className="flex gap-2">
                      <button type="button" disabled={busy} onClick={() => void runConfirmed()} className="rounded px-2 py-1 text-xs font-semibold disabled:opacity-40" style={{ background: "var(--accent)", color: "var(--on-accent, #fff)" }}>
                        {busy ? "Joining…" : "Yes, join them"}
                      </button>
                      <button type="button" disabled={busy} onClick={() => setConfirm(null)} className="rounded px-2 py-1 text-xs border" style={{ borderColor: "var(--border)" }}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <button type="button" onClick={() => { setDone(null); setConfirm({ kind: "join", join: j }); }} className="rounded px-2 py-1 text-xs font-semibold border" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>Review and join…</button>
                    <button type="button" onClick={() => dismiss(key)} className="rounded px-2 py-1 text-xs text-[var(--text-secondary)]">Not the same door</button>
                  </div>
                )}
              </div>
            );
          })}
        </section>
      )}

      {visibleRevisits.length > 0 && (
        <section aria-label="Possible second visits" className="flex flex-col gap-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-tertiary)]">Possible second visits</h4>
          {visibleRevisits.map((r) => {
            const key = `${r.room_a}|${r.room_b}`;
            const rv = review?.key === key ? review : null;
            const swapped = rv ? rv.oldRoom === r.room_b : false;
            const confirming = confirm?.kind === "revisit" && rv && confirm.oldRoom === rv.oldRoom && confirm.newRoom === rv.newRoom;
            return (
              <div key={key} className="rounded-md p-2 flex flex-col gap-2" style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)" }}>
                <div className="flex items-start justify-between gap-2">
                  <p className="text-xs leading-snug">{revisitSentence(r)}</p>
                  <Badge confidence={r.confidence} />
                </div>
                <div className="flex flex-col gap-1">
                  <Bar label="Size" value={r.evidence.dimensions} />
                  <Bar label="Ceiling" value={r.evidence.ceiling_height} />
                  <Bar label="Openings" value={r.evidence.openings} />
                </div>
                {!rv && (
                  <div className="flex gap-2">
                    <button type="button" onClick={() => { setDone(null); void openReview(r, r.room_a, r.room_b); }} className="flex items-center gap-1 rounded px-2 py-1 text-xs font-semibold border" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>
                      <GitCompare className="w-3.5 h-3.5" /> Review changes…
                    </button>
                    <button type="button" onClick={() => dismiss(key)} className="rounded px-2 py-1 text-xs text-[var(--text-secondary)]">Not the same room</button>
                  </div>
                )}
                {rv && (
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between text-[11px] text-[var(--text-secondary)]">
                      <span>Newer visit: <strong className="text-[var(--text-primary)]">{roomName(rv.newRoom)}</strong></span>
                      <button type="button" onClick={() => void openReview(r, swapped ? r.room_a : r.room_b, swapped ? r.room_b : r.room_a)} className="underline">Swap which is newer</button>
                    </div>
                    {!rv.data && !rv.error && <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]"><Loader2 className="w-3.5 h-3.5 animate-spin" /> Comparing…</div>}
                    {rv.error && <p className="text-xs" style={{ color: "var(--conf-low, #e0524a)" }}>{rv.error}</p>}
                    {rv.data && (
                      <div className="flex flex-col gap-1.5">
                        <p className="text-xs font-semibold">{changeHeadline(rv.data.summary)}</p>
                        {groupChanges(rv.data.changes).map((g) => (
                          <details key={g.state} open={g.state === "conflicting" || g.state === "uncertain"}>
                            <summary className="cursor-pointer text-[11px]">{changeStateLabel(g.state)} ({g.items.length})</summary>
                            <ul className="mt-1 ml-3 list-disc text-[11px] text-[var(--text-secondary)]">
                              {g.items.map((c) => (
                                <li key={`${c.old}|${c.new}`}>{c.label ? `${c.kind}, ${c.label}` : c.kind}{c.size_diff ? ` (size differs ${Math.round(c.size_diff * 100)}%)` : ""}</li>
                              ))}
                            </ul>
                          </details>
                        ))}
                        <p className="text-[10px] text-[var(--text-tertiary)]">Sizes and kinds only: positions are not compared.</p>
                      </div>
                    )}
                    {confirming ? (
                      <div ref={(el) => el?.scrollIntoView({ block: "nearest" })} className="flex flex-col gap-2">
                        <p className="text-[11px] leading-snug text-[var(--text-secondary)]">{revisitConfirmation(rv.oldRoom, rv.newRoom)}</p>
                        <div className="flex gap-2">
                          <button type="button" disabled={busy} onClick={() => void runConfirmed()} className="rounded px-2 py-1 text-xs font-semibold disabled:opacity-40" style={{ background: "var(--accent)", color: "var(--on-accent, #fff)" }}>
                            {busy ? "Replacing…" : "Yes, replace the room"}
                          </button>
                          <button type="button" disabled={busy} onClick={() => setConfirm(null)} className="rounded px-2 py-1 text-xs border" style={{ borderColor: "var(--border)" }}>Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <button type="button" disabled={!rv.data} onClick={() => setConfirm({ kind: "revisit", oldRoom: rv.oldRoom, newRoom: rv.newRoom })} className="rounded px-2 py-1 text-xs font-semibold border disabled:opacity-40" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>
                          This is the same room…
                        </button>
                        <button type="button" onClick={() => { setReview(null); setConfirm(null); }} className="rounded px-2 py-1 text-xs text-[var(--text-secondary)]">Back</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </section>
      )}
    </div>
  );
}
