'use client';

/**
 * CaptureScreen — device camera, one-hand operation, instant honest verdict.
 *
 * The shutter captures a real frame from getUserMedia, computes quality from
 * actual pixels, and shows the outcome immediately. No simulated telemetry:
 * only values the device actually reports (geolocation, compass) appear.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { XCircle, ChevronLeft, ClipboardCheck, AlertTriangle, CircleSlash } from 'lucide-react';
import { useMobileStore } from './store';
import type { FrameVerdict } from './types';
import { computeCoverage, coverageGuidance, coverageSummaryLine } from './coverage';

type CamState = 'idle' | 'starting' | 'live' | 'error';

export function CaptureScreen({ onExit }: { onExit: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [camState, setCamState] = useState<CamState>('idle');
  const [camError, setCamError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [geo, setGeo] = useState<GeolocationPosition | null>(null);
  const [heading, setHeading] = useState<number | null>(null);

  const activeSession = useMobileStore((s) => s.sessions.find((x) => x.sessionId === s.activeSessionId) ?? null);
  const addFrame = useMobileStore((s) => s.addFrame);
  const lastAssessment = useMobileStore((s) => s.lastAssessment);
  const activeTaskId = useMobileStore((s) => s.activeTaskId);
  const activeTask = useMobileStore((s) => s.tasks.find((t) => t.taskId === s.activeTaskId) ?? null);
  const setActiveTask = useMobileStore((s) => s.setActiveTask);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [retryTrigger, setRetryTrigger] = useState(0);

  const startCamera = useCallback(() => {
    setRetryTrigger((c) => c + 1);
  }, []);

  useEffect(() => {
    let active = true;
    navigator.mediaDevices?.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 } },
      audio: false,
    }).then(async (stream) => {
      if (!active) {
        for (const t of stream.getTracks()) t.stop();
        return;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCamState('live');
    }).catch((err) => {
      if (!active) return;
      setCamState('error');
      setCamError(
        (err as Error).name === 'NotAllowedError'
          ? 'Camera permission denied. Grant access in the browser to capture evidence.'
          : `Camera unavailable: ${(err as Error).message}`,
      );
    });

    return () => {
      active = false;
      const v = videoRef.current;
      const stream = v?.srcObject as MediaStream | null;
      if (stream) for (const t of stream.getTracks()) t.stop();
    };
  }, [retryTrigger]);

  // Real device telemetry, only when the device provides it.
  useEffect(() => {
    if (!('geolocation' in navigator)) return;
    const id = navigator.geolocation.watchPosition(
      (p) => setGeo(p),
      () => setGeo(null),
      { enableHighAccuracy: true, timeout: 8000 },
    );
    return () => navigator.geolocation.clearWatch(id);
  }, []);

  useEffect(() => {
    const handler = (e: DeviceOrientationEvent) => {
      const h = (e as DeviceOrientationEvent & { webkitCompassHeading?: number }).webkitCompassHeading;
      if (typeof h === 'number') setHeading(h);
      else if (e.alpha != null) setHeading(360 - e.alpha);
    };
    window.addEventListener('deviceorientationabsolute', handler as EventListener, true);
    return () => window.removeEventListener('deviceorientationabsolute', handler as EventListener, true);
  }, []);


  const shoot = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || camState !== 'live') return;
    setBusy(true);
    setCaptureError(null);
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d')!.drawImage(video, 0, 0);
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('canvas.toBlob returned null'))), 'image/jpeg', 0.92),
      );
      const label = video.srcObject instanceof MediaStream ? video.srcObject.getVideoTracks()[0]?.label ?? 'camera' : 'camera';
      await addFrame({
        blob,
        width: canvas.width,
        height: canvas.height,
        origin: 'device_camera',
        device: label,
        telemetry: {
          geolocation: geo ? { latitude: geo.coords.latitude, longitude: geo.coords.longitude, accuracyM: geo.coords.accuracy } : undefined,
          headingDeg: heading ?? undefined,
        },
      });
    } catch (err) {
      // A capture that failed must be visible: an unhandled rejection here would
      // look exactly like a shutter that did nothing.
      setCaptureError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const frames = activeSession?.frames ?? [];
  const counts = {
    useful: frames.filter((f) => f.verdict === 'USEFUL').length,
    rejected: frames.filter((f) => f.verdict === 'REJECTED').length,
    redundant: frames.filter((f) => f.verdict === 'REDUNDANT').length,
    missing: frames.filter((f) => f.localCopy === 'missing').length,
    total: frames.length,
  };

  const verdictColor: Record<FrameVerdict, string> = {
    USEFUL: 'bg-[#2ecc71]',
    REDUNDANT: 'bg-[#f1c40f]',
    REJECTED: 'bg-[#e74c3c]',
    UNASSESSED: 'bg-[#9296a6]',
  };

  return (
    <div className="flex h-full flex-col bg-[#0d0e12] text-[#f0f1f6]">
      <div className="flex items-center gap-3 border-b border-[#1f222b] px-4 py-3">
        <button type="button" onClick={onExit} aria-label="Back" className="rounded-lg border border-[#2b3040] bg-[#171922] p-2">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{activeSession?.name ?? 'No session'}</div>
          <div className="font-mono text-[10px] text-[#9296a6]">
            {counts.total} captured · {counts.useful} useful · {counts.rejected} rejected · {counts.redundant} redundant
            {counts.missing > 0 && <span className="text-[#e74c3c]"> · {counts.missing} no local copy</span>}
          </div>
        </div>
      </div>

      {activeTask && (
        <div className="flex items-start gap-2 border-b border-[#f1c40f]/40 bg-[#f1c40f]/10 px-4 py-2">
          <ClipboardCheck className="mt-0.5 h-4 w-4 shrink-0 text-[#f1c40f]" />
          <div className="min-w-0 flex-1">
            <div className="text-[11px] font-semibold text-[#f1c40f]">Requested capture</div>
            <div className="text-[11px] leading-snug text-[#f0f1f6]">{activeTask.guidance}</div>
            {activeTask.status === 'DONE' && (
              <div className="mt-0.5 font-mono text-[9px] text-[#2ecc71]">
                serviced by {activeTask.completedByFrameIds.join(', ')}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => setActiveTask(null)}
            aria-label="Stop shooting for this request"
            className="flex shrink-0 items-center gap-1 rounded-lg border border-[#2b3040] bg-[#171922] px-2 py-1 font-mono text-[9px] text-[#9296a6]"
          >
            <CircleSlash className="h-3 w-3" /> stop
          </button>
        </div>
      )}

      {captureError && (
        <div className="flex items-start gap-2 border-b border-[#e74c3c]/40 bg-[#e74c3c]/10 px-4 py-2 text-[11px] text-[#f0c4c0]">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#e74c3c]" />
          <span>{captureError}</span>
        </div>
      )}

      {(() => {
        if (!activeSession) return null;
        const model = computeCoverage(activeSession);
        const gaps = coverageGuidance(model);
        const line = coverageSummaryLine(model);
        if (!line) return null;
        const next = gaps[0];
        return (
          <div className={`border-b px-4 py-2 ${model.fullyCovered ? 'border-[#2ecc71]/40 bg-[#2ecc71]/10' : next ? 'border-[#f1c40f]/40 bg-[#f1c40f]/10' : 'border-[#1f222b] bg-[#171922]'}`}>
            <div className={`text-[11px] font-semibold ${model.fullyCovered ? 'text-[#2ecc71]' : 'text-[#f1c40f]'}`}>
              {model.fullyCovered ? '✓ You don\'t need more footage here' : line}
            </div>
            {next && (
              <div className="mt-0.5 text-[11px] leading-snug text-[#f0f1f6]">Next: {next.guidance}</div>
            )}
            {gaps.length > 1 && (
              <div className="mt-0.5 font-mono text-[9px] text-[#9296a6]">+{gaps.length - 1} more direction{gaps.length - 1 === 1 ? '' : 's'} to cover</div>
            )}
          </div>
        );
      })()}

      <div className="relative min-h-0 flex-1 bg-black">
        <video ref={videoRef} playsInline muted className="h-full w-full object-cover" />
        <canvas ref={canvasRef} className="hidden" />
        {camState !== 'live' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
            {camState === 'error' ? (
              <>
                <XCircle className="h-8 w-8 text-[#e74c3c]" />
                <p className="text-sm text-[#9296a6]">{camError}</p>
                <button type="button" onClick={() => void startCamera()} className="rounded-xl bg-[#3d8ef7] px-5 py-2 text-sm font-semibold">
                  Retry camera
                </button>
              </>
            ) : (
              <p className="font-mono text-xs text-[#9296a6]">Starting camera…</p>
            )}
          </div>
        )}

        {lastAssessment && (
          <div className="absolute inset-x-3 top-3 z-10">
            <div className="flex items-center gap-2 rounded-xl bg-black/80 px-3 py-2 backdrop-blur-md">
              <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${verdictColor[lastAssessment.verdict]}`} />
              <p className="text-xs font-medium leading-snug">{lastAssessment.message}</p>
            </div>
          </div>
        )}

        <div className="absolute right-3 top-3 z-10 space-y-1 text-right font-mono text-[10px]">
          {geo && <div className="rounded-lg bg-black/75 px-2 py-1 backdrop-blur-md">GPS ±{Math.round(geo.coords.accuracy)}m</div>}
          {heading != null && <div className="rounded-lg bg-black/75 px-2 py-1 backdrop-blur-md">HDG {Math.round(heading)}°</div>}
          {!geo && !heading && (
            <div className="rounded-lg bg-black/75 px-2 py-1 text-[#9296a6] backdrop-blur-md">No telemetry reported</div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-center border-t border-[#1f222b] bg-[#0d0e12] px-6 py-5">
        <button
          type="button"
          disabled={busy || camState !== 'live'}
          onClick={() => void shoot()}
          aria-label="Capture frame"
          className="flex h-16 w-16 items-center justify-center rounded-full border-4 border-white/80 p-1 transition-transform active:scale-95 disabled:opacity-40"
        >
          <span className="block h-full w-full rounded-full bg-white" />
        </button>
      </div>
      <div className="pb-4 text-center text-[10px] text-[#9296a6]">
        {busy ? 'Assessing frame…' : 'Every frame is quality-assessed the moment it is captured.'}
      </div>
    </div>
  );
}
