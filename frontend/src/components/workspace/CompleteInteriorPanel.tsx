"use client";

import { useCallback, useState } from "react";
import { createPortal } from "react-dom";
import { Check, Loader2, Sparkles, X } from "lucide-react";
import { isApiError } from "@/lib/api/client";
import {
  acceptCompletionProposal,
  createCompletionProposals,
  getCompletionDiff,
  listCompletionProposals,
  rejectCompletionProposal,
} from "@/lib/api/completion";
import RoomListEditor, { Field, inputCls } from "@/components/workspace/RoomListEditor";
import { effectiveEntrance, type AdjacencyDraft, type RoomRow } from "@/lib/procedural/apartment-model";
import {
  COMPLETION_PROGRAMS,
  addedAllGenerated,
  diffSummary,
  footprintLine,
  programLine,
  provenanceText,
  sortProposals,
  statusBadge,
  toProposalBody,
  unitText,
  validateCompletion,
  type CompletionDiff,
  type CompletionDraft,
  type CompletionProgram,
  type CompletionProposalRow,
} from "@/lib/procedural/completion-model";

function errText(err: unknown, fallback: string): string {
  // Server text verbatim (422 "no usable shell", 409 "shell changed" / "name taken" / "already ..."); no retry suffix.
  return isApiError(err) ? err.message : fallback;
}

/** Toolbar entry + dialog: propose GENERATED interiors inside a reconstructed shell, review the diff, accept or reject.
 * Nothing is committed until Accept. `onChanged` must refetch the WorldIR. */
export default function CompleteInteriorPanel({ worldId, onChanged }: { worldId: string; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("interior");
  const [program, setProgram] = useState<CompletionProgram>("studio");
  const [rows, setRows] = useState<RoomRow[]>([]);
  const [entrance, setEntrance] = useState("");
  const [adjacency, setAdjacency] = useState<AdjacencyDraft[]>([]);
  const [seed, setSeed] = useState(0);
  const [alternatives, setAlternatives] = useState(2);

  const [proposals, setProposals] = useState<CompletionProposalRow[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [diff, setDiff] = useState<CompletionDiff | null>(null);
  const [diffError, setDiffError] = useState<string | null>(null);
  const [proposing, setProposing] = useState(false);
  const [deciding, setDeciding] = useState<"accept" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const draft: CompletionDraft = { name, program, rows, entrance: effectiveEntrance(rows, entrance), adjacency, seed, alternatives };
  const problems = validateCompletion(draft);

  const refresh = useCallback(async () => {
    try {
      setProposals(sortProposals(await listCompletionProposals(worldId)));
    } catch (err) {
      setError(errText(err, "Could not load proposals."));
    }
  }, [worldId]);

  const select = useCallback(async (pid: string) => {
    setSelected(pid);
    setDiff(null);
    setDiffError(null);
    try {
      setDiff(await getCompletionDiff(worldId, pid));
    } catch (err) {
      setDiffError(errText(err, "Could not compute the diff."));
    }
  }, [worldId]);

  async function propose() {
    setProposing(true);
    setError(null);
    setNotice(null);
    try {
      const res = await createCompletionProposals(worldId, toProposalBody(draft));
      await refresh();
      setNotice(`${res.found} of ${res.requested} requested alternative${res.requested === 1 ? "" : "s"} found. ${res.note}`);
      if (res.proposals[0]) await select(res.proposals[0].id);
    } catch (err) {
      setError(errText(err, "Could not create proposals."));
    } finally {
      setProposing(false);
    }
  }

  async function decide(kind: "accept" | "reject", pid: string) {
    setDeciding(kind);
    setError(null);
    setNotice(null);
    try {
      if (kind === "accept") {
        const r = await acceptCompletionProposal(worldId, pid);
        setNotice(`Accepted: ${r.entity_ids_added.length} GENERATED entities committed as a new version.`);
        onChanged();
      } else {
        await rejectCompletionProposal(worldId, pid);
        setNotice("Rejected. The world is unchanged.");
      }
      await refresh();
      await select(pid);
    } catch (err) {
      setError(errText(err, `Could not ${kind} the proposal.`));
      await refresh(); // a 409 usually means the row changed under us
    } finally {
      setDeciding(null);
    }
  }

  const current = proposals?.find((p) => p.id === selected) ?? null;
  const pending = current?.status === "proposed";

  return (
    <>
      <button
        type="button"
        onClick={() => {
          setOpen(true);
          void refresh();
        }}
        title="Propose a GENERATED interior inside the reconstructed shell"
        data-testid="open-complete"
        className="flex items-center gap-1 px-2.5 h-7 rounded text-xs font-medium bg-[var(--bg-elevated)] hover:bg-neutral-800 text-neutral-300 hover:text-white border border-[var(--border)] transition-colors cursor-pointer"
      >
        <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]" />
        <span className="hidden sm:inline">Complete interior</span>
      </button>

      {open && createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="Complete interior">
          <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--bg-base)] p-4 flex flex-col gap-3 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
                <Sparkles className="w-4 h-4 text-[var(--accent)]" /> Complete interior
                <span className="text-[10px] font-normal text-[var(--warning)] border border-[var(--warning)] rounded px-1">GENERATED</span>
              </h3>
              <button type="button" onClick={() => setOpen(false)} aria-label="Close" className="cursor-pointer">
                <X className="w-4 h-4 text-neutral-400" />
              </button>
            </div>

            <p className="text-xs text-[var(--text-secondary)]">
              Fills the reconstructed shell with a room layout that is proposed, not observed. Nothing is added to the world until you accept it,
              and accepted rooms stay labelled GENERATED. The fit uses the shell&apos;s axis-aligned bounding box.
            </p>

            <Field label="Name">
              <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Interior name" className={inputCls} />
            </Field>
            <Field label="Room program">
              <select value={program} onChange={(e) => setProgram(e.target.value as CompletionProgram)} aria-label="Room program" className={inputCls}>
                {COMPLETION_PROGRAMS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
              </select>
            </Field>
            {program === "custom" && (
              <RoomListEditor
                rows={rows}
                setRows={setRows}
                entrance={draft.entrance}
                setEntrance={setEntrance}
                adjacency={adjacency}
                setAdjacency={setAdjacency}
              />
            )}
            <div className="grid grid-cols-2 gap-2">
              <Field label="Seed (same seed = same proposals)">
                <input type="number" step={1} value={Number.isFinite(seed) ? seed : ""} onChange={(e) => setSeed(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Seed" className={inputCls} />
              </Field>
              <Field label="Alternatives (1-3)">
                <input type="number" step={1} min={1} max={3} value={Number.isFinite(alternatives) ? alternatives : ""} onChange={(e) => setAlternatives(e.target.value === "" ? NaN : Number(e.target.value))} aria-label="Alternatives" className={inputCls} />
              </Field>
            </div>

            {problems.length > 0 && (
              <ul className="text-xs text-[var(--warning)] list-disc pl-4" data-testid="complete-problems">
                {problems.map((p) => <li key={p}>{p}</li>)}
              </ul>
            )}

            <button
              type="button"
              onClick={() => void propose()}
              disabled={proposing || problems.length > 0}
              className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold border border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--accent)] disabled:opacity-40 cursor-pointer"
            >
              {proposing && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              {proposing ? "Proposing…" : "Propose"}
            </button>

            {error && <p role="alert" data-testid="complete-error" className="text-xs text-[var(--error)]">{error}</p>}
            {notice && <p role="status" data-testid="complete-notice" className="text-xs text-[var(--text-secondary)]">{notice}</p>}

            {proposals && proposals.length > 0 && (
              <div className="flex flex-col gap-2" data-testid="complete-proposals">
                <p className="text-xs text-[var(--text-secondary)]">Proposals for this world. Select one to review its diff.</p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {proposals.map((p) => {
                    const on = selected === p.id;
                    const fp = p.summary.footprint;
                    const longest = Math.max(fp.width, fp.depth, 0.001);
                    return (
                      <button
                        key={p.id}
                        type="button"
                        data-testid="complete-proposal"
                        aria-pressed={on}
                        onClick={() => void select(p.id)}
                        className={`rounded border p-1.5 text-left flex flex-col gap-1 cursor-pointer ${
                          on ? "border-[var(--accent)] bg-[var(--accent-subtle)]" : "border-[var(--border)] hover:bg-[var(--bg-hover)]"
                        }`}
                      >
                        <svg viewBox="0 0 120 80" className="w-full h-auto" role="img" aria-label={`Footprint of alternative ${p.alt_index + 1}`}>
                          {/* To-scale shell footprint only: the API does not return the proposed room rectangles before acceptance. */}
                          <rect
                            x={4} y={4} width={(fp.width / longest) * 110} height={(fp.depth / longest) * 70}
                            fill="none" stroke="var(--warning)" strokeWidth={1.5} strokeDasharray="4 3"
                          />
                          <text x={60} y={42} textAnchor="middle" fontSize={9} fill="var(--text-secondary)">shell footprint</text>
                        </svg>
                        <span className="text-[11px] font-semibold text-[var(--warning)]" data-testid="complete-badge">{statusBadge(p.status)}</span>
                        <span className="text-[11px] text-[var(--text-primary)]">{p.summary.name} - alternative {p.alt_index + 1}, seed {p.seed}</span>
                        <span className="text-[11px] text-[var(--text-secondary)]">{footprintLine(fp, p.summary.unit)}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
            {proposals && proposals.length === 0 && (
              <p className="text-xs text-neutral-400" data-testid="complete-empty">No proposals yet. Choose a program and Propose.</p>
            )}

            {current && (
              <section className="rounded border border-dashed border-[var(--warning)] p-2.5 flex flex-col gap-2" data-testid="complete-diff" aria-label="Proposal diff">
                <h4 className="text-xs font-semibold text-white flex items-center gap-2">
                  Existing vs proposed
                  <span className="text-[10px] font-normal text-[var(--warning)] border border-[var(--warning)] rounded px-1">{statusBadge(current.status)}</span>
                </h4>
                <p className="text-[11px] text-[var(--text-secondary)]">{current.label}</p>
                <p className="text-[11px] text-[var(--text-secondary)]">
                  Footprint: {footprintLine(current.summary.footprint, current.summary.unit)}. Units: {unitText(current.summary.unit)}.
                </p>
                <p className="text-[11px] text-[var(--text-secondary)]">Program: {programLine(current.summary.program.rooms)}; entrance {current.summary.program.entrance}.</p>
                {diffError && <p role="alert" className="text-xs text-[var(--error)]">{diffError}</p>}
                {!diff && !diffError && <p className="text-xs text-neutral-400 flex items-center gap-2"><Loader2 className="w-3.5 h-3.5 animate-spin" /> Computing diff…</p>}
                {diff && diff.proposal_id === current.id && (
                  <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs text-[var(--text-primary)]">
                    <dt className="text-[var(--text-secondary)]">Change</dt>
                    <dd data-testid="complete-diff-counts">{diffSummary(diff)}</dd>
                    <dt className="text-[var(--text-secondary)]">Existing provenance</dt>
                    <dd data-testid="complete-diff-existing">{provenanceText(diff.provenance.existing)}</dd>
                    <dt className="text-[var(--text-secondary)]">Proposed provenance</dt>
                    <dd data-testid="complete-diff-added">{provenanceText(diff.provenance.added)} (PROPOSED)</dd>
                  </dl>
                )}
                {diff && diff.proposal_id === current.id && !addedAllGenerated(diff) && (
                  <p role="alert" className="text-xs text-[var(--error)]">Warning: not every added entity is GENERATED. Do not accept until this is explained.</p>
                )}
                {diff && diff.proposal_id === current.id && (
                  <details className="text-[11px] text-[var(--text-secondary)]">
                    <summary className="cursor-pointer">Entity ids ({diff.added.length} added)</summary>
                    <p className="break-words">{diff.added.join(", ") || "none"}</p>
                  </details>
                )}
                {diff && <p className="text-[11px] text-[var(--text-secondary)]">{diff.generated_label}</p>}

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => void decide("accept", current.id)}
                    disabled={!pending || deciding !== null}
                    className="flex-1 flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold bg-[var(--primary)] text-[var(--text-primary)] hover:bg-[var(--primary-hover)] disabled:opacity-40 cursor-pointer"
                  >
                    {deciding === "accept" ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                    Accept as GENERATED
                  </button>
                  <button
                    type="button"
                    onClick={() => void decide("reject", current.id)}
                    disabled={!pending || deciding !== null}
                    className="flex-1 h-9 rounded-md text-sm font-medium border border-[var(--border)] text-neutral-300 hover:text-white disabled:opacity-40 cursor-pointer"
                  >
                    Reject
                  </button>
                </div>
                {!pending && <p className="text-[11px] text-[var(--text-tertiary)]">This proposal is already {current.status}; it can no longer be accepted or rejected.</p>}
              </section>
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
