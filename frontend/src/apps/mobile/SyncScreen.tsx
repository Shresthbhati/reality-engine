'use client';

/**
 * SyncScreen — session summary + handoff to the engine.
 *
 * The handoff artifact is a real, self-contained session bundle
 * (`re.mobile-session-bundle/v1`) built from the verified pixels in the device
 * vault. The bundle is verified again on ingest:
 *
 *     reality session ingest-mobile <bundle.json> --session-dir <dir>
 *     reality session mobile-tasks  <bundle.json> -o tasks.json
 *
 * A network ingest server does not exist yet. That state is shown honestly —
 * export is real, wireless sync is not, and nothing here pretends otherwise.
 * Contract: docs/mobile/BACKEND_CONTRACT.md
 */
import { useState } from 'react';
import {
  ChevronLeft,
  DownloadCloud,
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  HardDriveDownload,
} from 'lucide-react';
import { summarizeSession, useMobileStore } from './store';
import { buildMobileBundle } from './bundle';
import { blobToBytes, vaultGetMany } from './frameVault';

function humanBytes(b: number): string {
  if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024).toFixed(0)} KB`;
}

interface ExportReport {
  bundleId: string;
  included: number;
  skipped: number;
  payloadBytes: number;
  fileName: string;
}

export function SyncScreen({ onExit }: { onExit: () => void }) {
  const session = useMobileStore((s) => s.sessions.find((x) => x.sessionId === s.activeSessionId) ?? null);
  const tasks = useMobileStore((s) => s.tasks);
  const finishSession = useMobileStore((s) => s.finishSession);
  const markSynced = useMobileStore((s) => s.markSynced);
  const sync = useMobileStore((s) => s.sync);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<ExportReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!session) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 bg-[#0d0e12] text-[#9296a6]">
        <p className="text-sm">No session selected.</p>
        <button type="button" onClick={onExit} className="rounded-xl bg-[#3d8ef7] px-5 py-2 text-sm font-semibold text-[#f0f1f6]">
          Back to sessions
        </button>
      </div>
    );
  }

  const summary = summarizeSession(session);
  const sessionTasks = tasks.filter((t) => t.sessionId === session.sessionId);
  const openTasks = sessionTasks.filter((t) => t.status === 'OPEN');
  const handoffable = summary.total - summary.missingLocalCopy;

  const exportBundle = async () => {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const entries = await vaultGetMany(session.frames.map((f) => f.contentSha256));
      const payloads = new Map<string, Uint8Array>();
      for (const [sha, entry] of entries) payloads.set(sha, await blobToBytes(entry.blob));

      const built = await buildMobileBundle({ session, payloads, tasks });

      const blob = new Blob([built.text], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const fileName = `session-${session.sessionId}.bundle.json`;
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      a.click();
      URL.revokeObjectURL(url);

      markSynced(new Date().toISOString());
      finishSession(session.sessionId);
      setReport({
        bundleId: built.bundleId,
        included: built.includedFrameIds.length,
        skipped: built.skippedFrames.length,
        payloadBytes: built.payloadBytes,
        fileName,
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-[#0d0e12] text-[#f0f1f6]">
      <div className="flex items-center gap-3 border-b border-[#1f222b] px-4 py-3">
        <button type="button" onClick={onExit} aria-label="Back" className="rounded-lg border border-[#2b3040] bg-[#171922] p-2">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">Hand off — {session.name}</div>
          <div className="font-mono text-[10px] text-[#9296a6]">
            {session.status} · {summary.total} frames · {humanBytes(summary.bytes)}
          </div>
        </div>
      </div>

      <div className="space-y-3 p-4">
        <div className="grid grid-cols-2 gap-2">
          <Stat label="frames" value={summary.total} tone="text-[#f0f1f6]" />
          <Stat label="useful" value={summary.useful} tone="text-[#2ecc71]" />
          <Stat label="rejected" value={summary.rejected} tone="text-[#e74c3c]" />
          <Stat label="redundant" value={summary.redundant} tone="text-[#f1c40f]" />
        </div>

        <div className="rounded-2xl border border-[#2b3040] bg-[#171922] p-4">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-[#9296a6]">Evidence completeness</div>
          <p className="text-xs leading-relaxed text-[#f0f1f6]">
            {handoffable} of {summary.total} frame{summary.total === 1 ? '' : 's'} carry verifiable pixels on this device.
          </p>
          {summary.missingLocalCopy > 0 ? (
            <div className="mt-2 flex items-start gap-2 rounded-xl border border-[#e74c3c]/40 bg-[#e74c3c]/10 p-3 text-xs text-[#f0c4c0]">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#e74c3c]" />
              <span>
                {summary.missingLocalCopy} frame{summary.missingLocalCopy === 1 ? '' : 's'} have no local copy and will be
                listed as skipped. Additional evidence would significantly improve this session — recapture those viewpoints.
              </span>
            </div>
          ) : summary.total > 0 ? (
            <div className="mt-2 flex items-start gap-2 rounded-xl border border-[#2ecc71]/40 bg-[#2ecc71]/10 p-3 text-xs text-[#c0f0d4]">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[#2ecc71]" />
              <span>Every captured frame is present and sha256-verifiable. Export to hand off to the engine.</span>
            </div>
          ) : (
            <p className="mt-2 text-xs text-[#9296a6]">Nothing captured yet — there is no evidence to hand off.</p>
          )}
          {summary.headingSpreadDeg != null && (
            <p className="mt-2 font-mono text-[10px] text-[#9296a6]">
              measured viewpoint spread {Math.round(summary.headingSpreadDeg)}° · span{' '}
              {summary.timeSpanSec == null ? 'unknown' : `${Math.round(summary.timeSpanSec)}s`}
            </p>
          )}
        {sessionTasks.length > 0 && (
          <div className="rounded-2xl border border-[#2b3040] bg-[#171922] p-4">
            <div className="mb-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-wide text-[#9296a6]">
              <ClipboardCheck className="h-3.5 w-3.5" /> requested captures
            </div>
            <ul className="space-y-2">
              {sessionTasks.map((task) => (
                <li key={task.taskId} className="rounded-xl bg-[#0d0e12] px-3 py-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className={task.status === 'DONE' ? 'text-[#2ecc71]' : 'text-[#f1c40f]'}>{task.status}</span>
                    <span className="font-mono text-[9px] text-[#9296a6]">{task.taskId}</span>
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-[#f0f1f6]">{task.guidance}</p>
                  <p className="mt-0.5 font-mono text-[9px] text-[#9296a6]">
                    {task.completedByFrameIds.length > 0
                      ? `serviced by ${task.completedByFrameIds.join(', ')}`
                      : 'no usable frame captured for this request yet'}
                  </p>
                </li>
              ))}
            </ul>
            {openTasks.length > 0 && (
              <p className="mt-2 text-[11px] text-[#f1c40f]">
                {openTasks.length} request{openTasks.length === 1 ? '' : 's'} still open — additional evidence would improve this
                session.
              </p>
            )}
          </div>
        )}
</div>

        <div className="rounded-2xl border border-[#2b3040] bg-[#171922] p-4">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-[#9296a6]">Handoff</div>
          {error && (
            <div className="mb-2 flex items-start gap-2 rounded-lg bg-[#e74c3c]/10 px-3 py-2 text-xs text-[#f0c4c0]">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
          {report && (
            <div className="mb-2 rounded-xl border border-[#2ecc71]/40 bg-[#2ecc71]/10 px-3 py-2 text-[11px] text-[#c0f0d4]">
              <div className="font-semibold">
                Wrote {report.fileName} · {report.included} frame(s) · {humanBytes(report.payloadBytes)} of pixels
              </div>
              <div className="mt-1 font-mono text-[9px]">
                {report.bundleId}
                {report.skipped > 0 && ` · ${report.skipped} frame(s) skipped (no local pixels)`}
              </div>
              <div className="mt-1 font-mono text-[9px]">
                desktop: reality session ingest-mobile {report.fileName} --session-dir DIR
              </div>
            </div>
          )}

          <button
            type="button"
            disabled={busy || summary.total === 0}
            onClick={() => void exportBundle()}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#3d8ef7] py-3 text-sm font-bold text-white disabled:opacity-40"
          >
            <DownloadCloud className="h-4 w-4" />
            {busy ? 'Verifying pixels…' : 'Export verified session bundle'}
          </button>

          <div className="mt-3 flex items-start gap-2 text-[11px] text-[#9296a6]">
            <HardDriveDownload className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#f1c40f]" />
            <span>
              {sync.lastSyncAt ? `Last export ${new Date(sync.lastSyncAt).toLocaleString()}. ` : 'No export yet. '}
              Over-the-air ingest is not available in this build — the bundle file is the handoff. Frame identity is verified on
              ingest, so the engine can reject anything that does not match; nothing is silently accepted.
            </span>
          </div>
          {sync.lastSyncError && (
            <div className="mt-2 rounded-lg bg-[#e74c3c]/10 px-3 py-2 text-xs text-[#f0c4c0]">{sync.lastSyncError}</div>
          )}
        </div>

        <div className="rounded-2xl border border-[#2b3040] bg-[#171922] p-4">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-[#9296a6]">Desktop entry point</div>
          <p className="text-[11px] leading-relaxed text-[#9296a6]">
            The bundle lands in a normal Reality Engine session. The engine re-derives quality itself — mobile verdicts travel as
            advisory triage only, and task outcomes are recorded back into the session so the desktop knows which requests were
            serviced.
          </p>
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-[#4a4f60]">
            reality session ingest-mobile bundle.json --session-dir ./site-session
          </p>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-xl bg-[#171922] px-3 py-2">
      <div className={`text-lg font-bold ${tone}`}>{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wide text-[#9296a6]">{label}</div>
    </div>
  );
}
