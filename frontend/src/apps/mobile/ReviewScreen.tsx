'use client';

/**
 * ReviewScreen — evidence inspector for captured frames.
 * Shows per-frame real metrics and reasons. Rejected/redundant frames stay
 * visible (never hidden) because "why was this frame not useful" is part of
 * evidence lineage.
 */
import { useState } from 'react';
import { ChevronLeft, ImageOff, ClipboardCheck } from 'lucide-react';
import { useMobileStore } from './store';
import { QUALITY_THRESHOLDS } from './quality';
import { computeCoverage, coverageSummaryLine } from './coverage';
import type { FrameRecord, FrameVerdict } from './types';

const VERDICT_LABEL: Record<FrameVerdict, string> = {
  USEFUL: 'USEFUL',
  REDUNDANT: 'REDUNDANT',
  REJECTED: 'REJECTED',
  UNASSESSED: 'UNASSESSED',
};

const VERDICT_DOT: Record<FrameVerdict, string> = {
  USEFUL: 'bg-[#2ecc71]',
  REDUNDANT: 'bg-[#f1c40f]',
  REJECTED: 'bg-[#e74c3c]',
  UNASSESSED: 'bg-[#9296a6]',
};

type Filter = 'ALL' | FrameVerdict;

const VERDICT_STYLE: Record<FrameVerdict, string> = {
  USEFUL: 'text-[#2ecc71] border-[#2ecc71]/40 bg-[#2ecc71]/10',
  REDUNDANT: 'text-[#f1c40f] border-[#f1c40f]/40 bg-[#f1c40f]/10',
  REJECTED: 'text-[#e74c3c] border-[#e74c3c]/40 bg-[#e74c3c]/10',
  UNASSESSED: 'text-[#9296a6] border-[#9296a6]/40 bg-[#9296a6]/10',
};

export function ReviewScreen({ onExit }: { onExit: () => void }) {
  const session = useMobileStore((s) => s.sessions.find((x) => x.sessionId === s.activeSessionId) ?? null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('ALL');

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

  const open = session.frames.find((f) => f.frameId === openId) ?? null;
  const missingCount = session.frames.filter((f) => f.localCopy === 'missing').length;
  const visible = filter === 'ALL' ? session.frames : session.frames.filter((f) => f.verdict === filter);
  const coverageModel = computeCoverage(session);
  const coverageLine = coverageSummaryLine(coverageModel);

  return (
    <div className="relative flex h-full flex-col bg-[#0d0e12] text-[#f0f1f6]">
      <div className="flex items-center gap-3 border-b border-[#1f222b] px-4 py-3">
        <button type="button" onClick={onExit} aria-label="Back" className="rounded-lg border border-[#2b3040] bg-[#171922] p-2">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div>
          <div className="text-sm font-semibold">Evidence — {session.name}</div>
          <div className="font-mono text-[10px] text-[#9296a6]">{session.frames.length} frames · sha256 content ids</div>
        </div>
      </div>

      {coverageLine && (
        <div className={`border-b px-4 py-1.5 text-[11px] font-medium ${coverageModel.fullyCovered ? 'border-[#2ecc71]/40 bg-[#2ecc71]/10 text-[#2ecc71]' : 'border-[#f1c40f]/40 bg-[#f1c40f]/10 text-[#f1c40f]'}`}>
          {coverageModel.fullyCovered ? `✓ ${coverageLine}` : coverageLine}
        </div>
      )}

      {session.frames.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
          <p className="text-sm text-[#9296a6]">No evidence captured yet.</p>
          <p className="text-xs text-[#9296a6]">Frames you capture will appear here with their quality verdict.</p>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 overflow-x-auto border-b border-[#1f222b] px-3 py-2">
            {(['ALL', 'USEFUL', 'REDUNDANT', 'REJECTED'] as Filter[]).map((option) => {
              const count =
                option === 'ALL'
                  ? session.frames.length
                  : session.frames.filter((f) => f.verdict === option).length;
              const active = filter === option;
              return (
                <button
                  key={option}
                  type="button"
                  onClick={() => setFilter(option)}
                  className={`shrink-0 rounded-full border px-3 py-1 font-mono text-[10px] ${
                    active
                      ? 'border-[#3d8ef7] bg-[#3d8ef7]/15 text-[#3d8ef7]'
                      : 'border-[#2b3040] bg-[#171922] text-[#9296a6]'
                  }`}
                >
                  {option.toLowerCase()} {count}
                </button>
              );
            })}
            {missingCount > 0 && (
              <span className="shrink-0 rounded-full border border-[#e74c3c]/40 bg-[#e74c3c]/10 px-3 py-1 font-mono text-[10px] text-[#e74c3c]">
                {missingCount} without pixels
              </span>
            )}
          </div>

          <div className="grid min-h-0 flex-1 grid-cols-3 content-start gap-2 overflow-y-auto p-3">
            {visible.map((f) => (
              <button
                key={f.frameId}
                type="button"
                onClick={() => setOpenId(f.frameId)}
                className={`relative overflow-hidden rounded-lg border ${f.verdict === 'REJECTED' ? 'border-[#e74c3c]/50' : 'border-[#2b3040]'} bg-[#171922]`}
              >
                {f.previewUrl ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={f.previewUrl} alt={`${f.verdict} frame ${f.frameId}`} className="aspect-square w-full object-cover" />
                  </>
                ) : (
                  // No pixels on this device: show that, never a broken image.
                  <span className="flex aspect-square w-full flex-col items-center justify-center gap-1 text-[#4a4f60]">
                    <ImageOff className="h-5 w-5" />
                    <span className="font-mono text-[8px]">no local copy</span>
                  </span>
                )}
                <span className={`absolute right-1 top-1 h-2.5 w-2.5 rounded-full ${VERDICT_DOT[f.verdict]}`} />
                {f.taskId && (
                  <span className="absolute left-1 top-1 rounded bg-black/70 p-0.5 text-[#f1c40f]">
                    <ClipboardCheck className="h-3 w-3" />
                  </span>
                )}
              </button>
            ))}
          </div>
        </>
      )}
      <FrameDetail frame={open} onClose={() => setOpenId(null)} />
    </div>
  );
}

function FrameDetail({ frame, onClose }: { frame: FrameRecord | null; onClose: () => void }) {
  if (!frame) return null;
  return (
    <div className="absolute inset-0 z-20 flex bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div
        className="m-auto max-h-[85%] w-[92%] overflow-y-auto rounded-2xl border border-[#2b3040] bg-[#0d0e12] p-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {frame.previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={frame.previewUrl} alt={`frame ${frame.frameId}`} className="max-h-60 w-full rounded-xl object-contain" />
        ) : (
          <div className="flex h-40 w-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-[#2b3040] text-[#4a4f60]">
            <ImageOff className="h-6 w-6" />
            <p className="text-xs">No pixels on this device — this frame cannot be handed off.</p>
          </div>
        )}
        <div className="mt-3 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${VERDICT_STYLE[frame.verdict]}`}>
              {VERDICT_LABEL[frame.verdict]}
            </span>
            <span className="font-mono text-[10px] text-[#9296a6]">{new Date(frame.capturedAt).toLocaleTimeString()}</span>
          </div>
          <div className="font-mono text-[10px] text-[#9296a6]">id {frame.frameId}</div>
          <div className="flex items-center justify-between font-mono text-[10px]">
            <span className="text-[#9296a6]">device storage</span>
            <span className={frame.localCopy === 'vault' ? 'text-[#2ecc71]' : 'text-[#e74c3c]'}>
              {frame.localCopy === 'vault' ? 'local copy present (sha256-addressed)' : 'no local copy — pixels missing'}
            </span>
          </div>
          {frame.taskId && (
            <div className="flex items-center gap-1 font-mono text-[10px] text-[#f1c40f]">
              <ClipboardCheck className="h-3 w-3" /> captured for request {frame.taskId}
            </div>
          )}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[10px]">
            <dt className="text-[#9296a6]">sharpness</dt>
            <dd className={frame.quality.sharpness < QUALITY_THRESHOLDS.sharpnessMin ? 'text-[#e74c3c]' : 'text-[#2ecc71]'}>
              {frame.quality.sharpness.toFixed(0)} / min {QUALITY_THRESHOLDS.sharpnessMin}
            </dd>
            <dt className="text-[#9296a6]">exposure</dt>
            <dd className="text-[#f0f1f6]">{(frame.quality.exposure * 100).toFixed(1)}%</dd>
            <dt className="text-[#9296a6]">clipped</dt>
            <dd className={frame.quality.clippedRatio > QUALITY_THRESHOLDS.clippedMax ? 'text-[#e74c3c]' : 'text-[#f0f1f6]'}>
              {(frame.quality.clippedRatio * 100).toFixed(2)}%
            </dd>
            <dt className="text-[#9296a6]">Δ previous</dt>
            <dd className="text-[#f0f1f6]">
              {frame.quality.diffVsPrevious == null ? 'unknown (first frame)' : `${(frame.quality.diffVsPrevious * 100).toFixed(1)}%`}
            </dd>
            <dt className="text-[#9296a6]">size</dt>
            <dd className="text-[#f0f1f6]">
              {frame.width}×{frame.height} · {(frame.bytes / 1024).toFixed(0)} KB
            </dd>
            {frame.telemetry.geolocation && (
              <>
                <dt className="text-[#9296a6]">gps</dt>
                <dd className="text-[#f0f1f6]">
                  {frame.telemetry.geolocation.latitude.toFixed(5)}, {frame.telemetry.geolocation.longitude.toFixed(5)} ±
                  {Math.round(frame.telemetry.geolocation.accuracyM)}m
                </dd>
              </>
            )}
            {frame.telemetry.headingDeg != null && (
              <>
                <dt className="text-[#9296a6]">heading</dt>
                <dd className="text-[#f0f1f6]">{frame.telemetry.headingDeg.toFixed(1)}°</dd>
              </>
            )}
          </dl>
          <div className="flex flex-wrap gap-1 pt-1">
            {frame.reasons.map((r) => (
              <span key={r} className="rounded bg-[#171922] px-1.5 py-0.5 font-mono text-[9px] text-[#9296a6]">
                {r}
              </span>
            ))}
          </div>
          <div className="pt-1 font-mono text-[9px] text-[#9296a6]">
            provenance: {frame.provenance.origin} · {frame.provenance.device} · {frame.provenance.ingestion}
          </div>
        </div>
        <button type="button" onClick={onClose} className="mt-4 w-full rounded-xl bg-[#3d8ef7] py-2 text-sm font-semibold">
          Close
        </button>
      </div>
    </div>
  );
}

