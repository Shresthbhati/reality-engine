"use client";

import { useState } from "react";
import { AlertTriangle, CheckCircle2, FileUp, GitPullRequest, Loader2, ShieldAlert, XCircle } from "lucide-react";
import { isApiError } from "@/lib/api/client";
import { useApi } from "@/lib/api/hooks";
import {
  acceptProposal,
  createProposal,
  getProposal,
  listProposals,
  rejectProposal,
  type ProposalAssessmentDto,
  type ProposalDto,
  type ProposalListItemDto,
  type ProposalVerdict,
} from "@/lib/api/proposals";
import {
  VERDICT_LABEL,
  confidenceText,
  defaultSelection,
  planAcceptance,
  sortAssessments,
  summaryLine,
} from "@/lib/proposals/proposal-model";

const VERDICT_COLOR: Record<ProposalVerdict, { fg: string; bg: string }> = {
  supported: { fg: "var(--success)", bg: "var(--success-subtle)" },
  needs_review: { fg: "var(--warning)", bg: "var(--warning-subtle)" },
  conflict: { fg: "var(--error)", bg: "var(--error-subtle)" },
};

const STATUS_COLOR = { pending: "var(--accent)", accepted: "var(--success)", rejected: "var(--text-secondary)" } as const;

function errorText(err: unknown): string {
  if (isApiError(err)) {
    const text = err.describe();
    const hint = err.status === 409 ? "Reload the list: the world may have moved on since this proposal was made." : "";
    return hint ? `${text}${/[.!?]$/.test(text) ? " " : ". "}${hint}` : text;
  }
  return err instanceof Error ? err.message : String(err);
}

function Pill({ label, fg, bg }: { label: string; fg: string; bg?: string }) {
  return (
    <span
      className="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded shrink-0"
      style={{ color: fg, background: bg ?? "transparent", border: `1px solid ${fg}` }}
    >
      {label}
    </span>
  );
}

function AssessmentRow({
  a, checked, disabled, onToggle,
}: { a: ProposalAssessmentDto; checked: boolean; disabled: boolean; onToggle: () => void }) {
  const c = VERDICT_COLOR[a.verdict];
  const conf = confidenceText(a);
  const id = `change-${a.entity_id}`;
  return (
    <li
      className="rounded-md p-3 flex gap-3"
      style={{
        background: "var(--bg-surface)",
        border: `1px solid ${a.verdict === "conflict" && checked ? "var(--error)" : "var(--border)"}`,
      }}
    >
      <input
        id={id}
        type="checkbox"
        className="mt-1 shrink-0"
        checked={checked}
        disabled={disabled}
        onChange={onToggle}
        aria-label={`Apply change to ${a.entity_id}`}
      />
      <div className="min-w-0 flex-1 space-y-1.5">
        <label htmlFor={id} className="flex flex-wrap items-center gap-2 cursor-pointer">
          <span className="font-mono text-xs font-semibold" style={{ color: "var(--text-primary)" }}>{a.entity_id}</span>
          <Pill label={a.kind} fg="var(--text-secondary)" />
          <Pill label={VERDICT_LABEL[a.verdict]} fg={c.fg} bg={c.bg} />
          {a.verdict === "conflict" && checked && <Pill label="Override" fg="var(--error)" />}
          {conf && <span className="text-[11px] font-mono" style={{ color: "var(--text-tertiary)" }}>confidence {conf}</span>}
        </label>
        {a.categories.length > 0 && (
          <p className="text-[11px]" style={{ color: "var(--text-tertiary)" }}>Changes: {a.categories.join(", ")}</p>
        )}
        {a.reasons.map((r) => (
          <p key={r} className="text-xs flex gap-1.5" style={{ color: c.fg }}>
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-px" aria-hidden />
            <span>{r}</span>
          </p>
        ))}
        {a.new_evidence.length > 0 && (
          <p className="text-[11px] font-mono" style={{ color: "var(--text-secondary)" }}>
            New evidence: {a.new_evidence.join(", ")}
          </p>
        )}
      </div>
    </li>
  );
}

function ProposalDetail({ worldId, proposal, onChanged }: { worldId: string; proposal: ProposalDto; onChanged: () => void }) {
  const pending = proposal.status === "pending";
  const [selected, setSelected] = useState<Set<string>>(() => new Set(defaultSelection(proposal.assessments)));
  const [reason, setReason] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const [busy, setBusy] = useState<"accept" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [accepted, setAccepted] = useState<string | null>(null);

  const items = sortAssessments(proposal.assessments);
  const plan = planAcceptance(proposal.assessments, selected, reason);
  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  async function doAccept() {
    setBusy("accept");
    setError(null);
    try {
      const res = await acceptProposal(worldId, proposal.proposal_id, {
        entity_ids: plan.entity_ids,
        override_conflicts: plan.override_conflicts,
        reason: plan.reason,
      });
      setAccepted(res.version_id);
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function doReject() {
    setBusy("reject");
    setError(null);
    try {
      await rejectProposal(worldId, proposal.proposal_id, rejectReason);
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  const decision = proposal.decision as { version_id?: string; applied?: string[]; reason?: string; overridden_conflicts?: string[] };
  const v = proposal.candidate_validation;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-mono text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{proposal.proposal_id}</h2>
        <Pill label={proposal.status} fg={STATUS_COLOR[proposal.status]} />
        <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
          against <span className="font-mono">{proposal.base_version_id}</span> · {summaryLine(proposal.summary)}
        </span>
      </div>

      {!v.valid && (
        <div role="alert" className="rounded-md p-3 text-xs space-y-1" style={{ background: "var(--error-subtle)", color: "var(--error)", border: "1px solid var(--error)" }}>
          <p className="font-semibold flex items-center gap-1.5"><ShieldAlert className="w-3.5 h-3.5" aria-hidden />The candidate world fails validation</p>
          {v.errors.slice(0, 5).map((m) => <p key={m} className="font-mono">{m}</p>)}
          <p>Accepting is refused while the merged world is invalid.</p>
        </div>
      )}

      {items.length === 0 ? (
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          The candidate is identical to the base version: there is nothing to review.
        </p>
      ) : (
        <ul className="space-y-2" aria-label="Proposed changes">
          {items.map((a) => (
            <AssessmentRow key={a.entity_id} a={a} checked={selected.has(a.entity_id)} disabled={!pending || busy !== null}
              onToggle={() => toggle(a.entity_id)} />
          ))}
        </ul>
      )}

      {accepted && (
        <p role="status" className="text-sm flex items-center gap-1.5" style={{ color: "var(--success)" }}>
          <CheckCircle2 className="w-4 h-4" aria-hidden />Accepted: new version <span className="font-mono">{accepted}</span> is now current.
        </p>
      )}
      {!pending && !accepted && (
        <div className="rounded-md p-3 text-xs space-y-1" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
          {proposal.status === "accepted" ? (
            <>
              <p>Accepted as version <span className="font-mono">{decision.version_id}</span>; applied {decision.applied?.length ?? 0} change(s).</p>
              {(decision.overridden_conflicts?.length ?? 0) > 0 && <p>Overridden conflicts: {decision.overridden_conflicts?.join(", ")}.</p>}
            </>
          ) : (
            <p>Rejected.</p>
          )}
          {decision.reason && <p>Reason: {decision.reason}</p>}
        </div>
      )}
      {error && <p role="alert" className="text-sm flex gap-1.5" style={{ color: "var(--error)" }}><XCircle className="w-4 h-4 shrink-0" aria-hidden />{error}</p>}

      {pending && (
        <div className="space-y-3 pt-3 border-t" style={{ borderColor: "var(--border)" }}>
          {plan.overridden.length > 0 && (
            <div className="space-y-1">
              <label htmlFor="override-reason" className="text-xs font-semibold" style={{ color: "var(--error)" }}>
                Reason for overriding {plan.overridden.join(", ")}
              </label>
              <textarea id="override-reason" value={reason} onChange={(e) => setReason(e.target.value)} rows={2}
                className="w-full rounded-md p-2 text-xs"
                style={{ background: "var(--bg-base)", color: "var(--text-primary)", border: "1px solid var(--border)" }} />
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3">
            <button type="button" onClick={doAccept} disabled={plan.blockers.length > 0 || busy !== null}
              className="h-9 px-4 rounded-md text-sm font-medium flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ background: "var(--primary)", color: "#fff" }}>
              {busy === "accept" ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden /> : <CheckCircle2 className="w-4 h-4" aria-hidden />}
              Accept {plan.entity_ids.length} change{plan.entity_ids.length === 1 ? "" : "s"}
            </button>
            {plan.blockers.map((b) => <span key={b} role="status" className="text-xs" style={{ color: "var(--warning)" }}>{b}</span>)}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="reject-reason" className="sr-only">Reason for rejecting</label>
            <input id="reject-reason" value={rejectReason} onChange={(e) => setRejectReason(e.target.value)} placeholder="Reason for rejecting"
              className="h-9 px-2 rounded-md text-xs flex-1 min-w-48"
              style={{ background: "var(--bg-base)", color: "var(--text-primary)", border: "1px solid var(--border)" }} />
            <button type="button" onClick={doReject} disabled={rejectReason.trim() === "" || busy !== null}
              className="h-9 px-4 rounded-md text-sm font-medium flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ border: "1px solid var(--error)", color: "var(--error)" }}>
              {busy === "reject" ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden /> : <XCircle className="w-4 h-4" aria-hidden />}
              Reject proposal
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function SubmitCandidate({ worldId, onCreated }: { worldId: string; onCreated: (p: ProposalDto) => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    let doc: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("expected a JSON object");
      doc = parsed as Record<string, unknown>;
    } catch (e) {
      setError(`Not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    setBusy(true);
    try {
      const p = await createProposal(worldId, doc);
      setText("");
      onCreated(p);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="rounded-md" style={{ border: "1px solid var(--border)", background: "var(--bg-surface)" }}>
      <summary className="cursor-pointer text-xs font-medium px-3 py-2 flex items-center gap-1.5" style={{ color: "var(--accent)" }}>
        <FileUp className="w-3.5 h-3.5" aria-hidden />Submit a candidate world
      </summary>
      <div className="p-3 space-y-2">
        <p className="text-[11px]" style={{ color: "var(--text-tertiary)" }}>
          A WorldIR document (for example a new scan&apos;s compiled world). It is compared with the current version; nothing changes until you accept.
        </p>
        <label htmlFor="candidate-file" className="sr-only">Load a WorldIR JSON file</label>
        <input id="candidate-file" type="file" accept="application/json,.json" className="text-xs"
          onChange={async (e) => { const f = e.target.files?.[0]; if (f) setText(await f.text()); }} />
        <label htmlFor="candidate-json" className="sr-only">WorldIR JSON</label>
        <textarea id="candidate-json" value={text} onChange={(e) => setText(e.target.value)} rows={5}
          className="w-full rounded-md p-2 text-xs font-mono"
          style={{ background: "var(--bg-base)", color: "var(--text-primary)", border: "1px solid var(--border)" }} />
        {error && <p role="alert" className="text-xs" style={{ color: "var(--error)" }}>{error}</p>}
        <button type="button" onClick={submit} disabled={busy || text.trim() === ""}
          className="h-8 px-3 rounded-md text-xs font-medium flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
          {busy && <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden />}Create proposal
        </button>
      </div>
    </details>
  );
}

function ListItem({ p, active, onSelect }: { p: ProposalListItemDto; active: boolean; onSelect: () => void }) {
  return (
    <li>
      <button type="button" onClick={onSelect} aria-current={active ? "true" : undefined}
        className="w-full text-left rounded-md p-3 space-y-1 transition-colors"
        style={{ background: active ? "var(--accent-subtle)" : "var(--bg-surface)", border: `1px solid ${active ? "var(--accent-border)" : "var(--border)"}` }}>
        <span className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs font-semibold" style={{ color: "var(--text-primary)" }}>{p.proposal_id}</span>
          <Pill label={p.status} fg={STATUS_COLOR[p.status]} />
        </span>
        <span className="block text-[11px]" style={{ color: "var(--text-secondary)" }}>{summaryLine(p.summary)}</span>
      </button>
    </li>
  );
}

export default function ProposalsPanel({ worldId }: { worldId: string }) {
  const list = useApi(() => listProposals(worldId), [worldId]);
  const [picked, setPicked] = useState<string | null>(null);
  const items = list.data ?? [];
  const activeId = picked ?? items[0]?.proposal_id ?? null;
  const detail = useApi(() => (activeId ? getProposal(worldId, activeId) : Promise.resolve(null)), [worldId, activeId]);

  const refresh = () => { list.refetch(); detail.refetch(); };

  return (
    <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
      <aside className="space-y-3">
        <SubmitCandidate worldId={worldId} onCreated={(p) => { setPicked(p.proposal_id); list.refetch(); }} />
        {list.isLoading && <p role="status" className="text-xs flex items-center gap-1.5" style={{ color: "var(--text-secondary)" }}><Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden />Loading proposals…</p>}
        {list.error && (
          <p role="alert" className="text-xs" style={{ color: "var(--error)" }}>
            {errorText(list.error)} <button type="button" onClick={list.refetch} className="underline">Retry</button>
          </p>
        )}
        {!list.isLoading && !list.error && items.length === 0 && (
          <div className="text-center py-8 space-y-2">
            <GitPullRequest className="w-7 h-7 mx-auto" style={{ color: "var(--text-tertiary)" }} aria-hidden />
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>No change proposals for this world yet.</p>
          </div>
        )}
        <ul className="space-y-2" aria-label="Proposals">
          {items.map((p) => <ListItem key={p.proposal_id} p={p} active={p.proposal_id === activeId} onSelect={() => setPicked(p.proposal_id)} />)}
        </ul>
      </aside>
      <section aria-live="polite">
        {detail.isLoading && activeId && <p role="status" className="text-xs flex items-center gap-1.5" style={{ color: "var(--text-secondary)" }}><Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden />Loading proposal…</p>}
        {detail.error && <p role="alert" className="text-sm" style={{ color: "var(--error)" }}>{errorText(detail.error)}</p>}
        {detail.data && <ProposalDetail key={detail.data.proposal_id + detail.data.status} worldId={worldId} proposal={detail.data} onChanged={refresh} />}
        {!activeId && !list.isLoading && <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Select a proposal to review its changes.</p>}
      </section>
    </div>
  );
}
