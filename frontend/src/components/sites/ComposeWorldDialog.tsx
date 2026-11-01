"use client";

import { useState } from "react";
import Link from "next/link";
import { Plus, X } from "lucide-react";
import { composeSite, isApiError, type ComposeResult, type Placement } from "@/lib/api";
import {
  CONNECTOR_NOTE,
  buildComposePayload,
  composeErrorMessage,
  describeComposeResult,
  newConnectorDraft,
  type ComposeKind,
  type ConnectorDraft,
  type ConnectorKind,
} from "@/lib/sites/compose-model";

interface Props {
  siteId: string;
  siteName: string;
  placements: Placement[];
  onClose: () => void;
}

const field = { background: "var(--bg-base)", borderColor: "var(--border)", color: "var(--text-primary)" } as const;
const KINDS: { value: ConnectorKind; label: string }[] = [
  { value: "driveway", label: "Driveway" },
  { value: "street", label: "Street" },
  { value: "corridor", label: "Corridor" },
];

export default function ComposeWorldDialog({ siteId, siteName, placements, onClose }: Props) {
  const [name, setName] = useState(`${siteName} (composed)`);
  const [kind, setKind] = useState<ComposeKind>("house");
  const [connectors, setConnectors] = useState<ConnectorDraft[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [result, setResult] = useState<ComposeResult | null>(null);

  const ids = placements.map((p) => p.id);
  const check = buildComposePayload(name, kind, connectors, ids);
  const setConn = (i: number, patch: Partial<ConnectorDraft>) =>
    setConnectors((cs) => cs.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  async function submit() {
    setSubmitted(true);
    if (!check.payload) return;
    setBusy(true);
    setFailure(null);
    try {
      setResult(await composeSite(siteId, check.payload));
    } catch (e) {
      setFailure(isApiError(e) ? composeErrorMessage(e.status, e.message) : (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "color-mix(in srgb, var(--bg-base) 70%, transparent)" }}
      onKeyDown={(e) => e.key === "Escape" && !busy && onClose()}
    >
      <div role="dialog" aria-modal="true" aria-labelledby="compose-title" className="w-full max-w-lg max-h-full overflow-y-auto rounded-xl border" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between px-5 h-14 border-b" style={{ borderColor: "var(--border)" }}>
          <h2 id="compose-title" className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Compose world</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>
            <X className="w-4 h-4" />
          </button>
        </div>

        {result ? (
          <div className="p-5 flex flex-col gap-3">
            <p className="text-sm" style={{ color: "var(--text-primary)" }}>Composed a {result.kind}: {describeComposeResult(result)}.</p>
            {result.generated_count > 0 && (
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {result.generated_count} generated {result.generated_count === 1 ? "entity is" : "entities are"} imagined placeholders, marked Generated (not captured).
              </p>
            )}
            {result.warnings.length > 0 && (
              <ul aria-label="Warnings" className="text-xs list-disc pl-4 flex flex-col gap-1" style={{ color: "var(--warning)" }}>
                {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            )}
            <div className="flex justify-end gap-2 mt-2">
              <button type="button" onClick={onClose} className="h-9 px-3 rounded-md text-sm hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>Close</button>
              <Link href={`/worlds/${encodeURIComponent(result.world_id)}`} className="h-9 px-4 rounded-md text-sm font-medium inline-flex items-center" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
                Open composed world
              </Link>
            </div>
          </div>
        ) : (
          <div className="p-5 flex flex-col gap-4">
            <div>
              <label htmlFor="compose-name" className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Name</label>
              <input id="compose-name" value={name} onChange={(e) => setName(e.target.value)} aria-invalid={submitted && !!check.nameError} className="w-full h-9 px-3 rounded-md border text-sm" style={field} />
              {submitted && check.nameError && <p role="alert" className="text-xs mt-1" style={{ color: "var(--error)" }}>{check.nameError}</p>}
            </div>

            <fieldset>
              <legend className="text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Kind</legend>
              <div className="flex gap-4 text-sm" style={{ color: "var(--text-primary)" }}>
                {(["house", "neighbourhood"] as const).map((k) => (
                  <label key={k} className="flex items-center gap-1.5 cursor-pointer">
                    <input type="radio" name="compose-kind" checked={kind === k} onChange={() => setKind(k)} />
                    {k === "house" ? "House" : "Neighbourhood"}
                  </label>
                ))}
              </div>
            </fieldset>

            <section aria-label="Connectors">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-secondary)" }}>Connectors (optional)</h3>
                <button type="button" onClick={() => setConnectors((cs) => [...cs, newConnectorDraft()])} disabled={placements.length < 2} className="flex items-center gap-1 text-xs px-2 h-7 rounded-md disabled:opacity-50" style={{ color: "var(--accent)" }}>
                  <Plus className="w-3 h-3" aria-hidden="true" /> Add connector
                </button>
              </div>
              <p className="text-xs mb-2" style={{ color: "var(--text-tertiary)" }}>{CONNECTOR_NOTE}</p>
              {placements.length < 2 && <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>Place at least two worlds to connect them.</p>}
              {connectors.map((c, i) => (
                <div key={i} className="rounded-md border p-2 mb-2 grid grid-cols-2 gap-2" style={{ borderColor: "var(--border-subtle)" }}>
                  <select aria-label={`Connector ${i + 1} kind`} value={c.kind} onChange={(e) => setConn(i, { kind: e.target.value as ConnectorKind })} className="h-9 px-2 rounded-md border text-sm" style={field}>
                    {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
                  </select>
                  <input aria-label={`Connector ${i + 1} width in metres`} inputMode="decimal" value={c.width} onChange={(e) => setConn(i, { width: e.target.value })} className="h-9 px-3 rounded-md border text-sm" style={field} />
                  <select aria-label={`Connector ${i + 1} from`} value={c.from} onChange={(e) => setConn(i, { from: e.target.value })} className="h-9 px-2 rounded-md border text-sm" style={field}>
                    <option value="">From…</option>
                    {placements.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                  </select>
                  <select aria-label={`Connector ${i + 1} to`} value={c.to} onChange={(e) => setConn(i, { to: e.target.value })} className="h-9 px-2 rounded-md border text-sm" style={field}>
                    <option value="">To…</option>
                    {placements.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                  </select>
                  <div className="col-span-2 flex items-center justify-between">
                    <span role={submitted && check.connectorErrors[i] ? "alert" : undefined} className="text-xs" style={{ color: "var(--error)" }}>{submitted ? check.connectorErrors[i] : null}</span>
                    <button type="button" onClick={() => setConnectors((cs) => cs.filter((_, j) => j !== i))} className="text-xs underline" style={{ color: "var(--text-secondary)" }}>Remove</button>
                  </div>
                </div>
              ))}
            </section>

            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Composing creates a new version of the composed world. If this site was composed before, that world gets a new version; the source worlds are never modified.
            </p>

            {failure && <p role="alert" className="text-sm rounded-md px-3 py-2" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>{failure}</p>}

            <div className="flex justify-end gap-2">
              <button type="button" onClick={onClose} disabled={busy} className="h-9 px-3 rounded-md text-sm hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>Cancel</button>
              <button type="button" onClick={submit} disabled={busy || placements.length === 0} className="h-9 px-4 rounded-md text-sm font-medium disabled:opacity-50" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
                {busy ? "Composing…" : "Compose"}
              </button>
            </div>
            {placements.length === 0 && <p className="text-xs -mt-2" style={{ color: "var(--text-tertiary)" }}>Add at least one world to the site first.</p>}
          </div>
        )}
      </div>
    </div>
  );
}
