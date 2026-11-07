"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AlertTriangle, Camera, Compass, Image as ImageIcon, Loader2, MapPin, RotateCcw, Smartphone, X } from "lucide-react";
import { isApiError, useWorldStatus } from "@/lib/api";
import { addSessionLocations, type LocationBatchPoint } from "@/lib/api/sessions";
import { getCaptureCoverage, coverageHeadline, type CaptureCoverageDto } from "@/lib/api/capture-coverage";
import { uploadPhoto } from "@/lib/mobile/upload";
import {
  clearCapturePhotos,
  loadPendingCapturePhotos,
  persistCapturePhoto,
  removeCapturePhoto,
  updateCapturePhotoStatus,
} from "@/lib/mobile/offline-queue";
import { EMPTY_QUEUE, nextToUpload, outcomeFromResponse, queueReducer, summarize, type QueueItem, type QueueItemLocation } from "@/lib/mobile/upload-queue";
import { photoResult, type Tone } from "@/lib/mobile/capture-result";
import WorldStatusCard, { TONE_COLOR } from "./WorldStatusCard";

/**
 * Mobile capture: REAL device camera (getUserMedia, with the phone's own camera
 * app and the photo library as fallbacks) -> per-photo upload queue with real
 * progress / failure / retry -> the engine's real per-photo result and the
 * world's real reconstruction state.
 *
 * Deliberately NOT here: coverage percentages, "enough footage" verdicts,
 * facade/area maps, suggested-path overlays. The backend does not measure
 * those, so the phone does not draw them. Guidance is the backend's own text.
 */

type GpsState =
  | { status: "loading" }
  | {
      status: "ready";
      lat: number;
      lng: number;
      accuracyM: number;
      altitude?: number | null;
      heading?: number | null;
      speed?: number | null;
      timestamp?: string;
    }
  | { status: "unavailable"; reason: string };

type CameraState = { kind: "starting" } | { kind: "live" } | { kind: "unavailable"; reason: string };

function describeCameraError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser settings, or use the phone's camera app below.";
  if (name === "NotFoundError") return "No camera was found on this device. Use the photo library below.";
  if (name === "NotSupportedError" || name === "SecurityError")
    return "A live camera preview needs a secure (HTTPS) connection. Use the phone's camera app below instead.";
  return `The camera could not start${err instanceof Error && err.message ? `: ${err.message}` : "."} Use the phone's camera app below instead.`;
}

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `p-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function toneStyle(tone: Tone) {
  return { color: TONE_COLOR[tone] };
}

/** Owns "capture a different place": remounting the screen drops the queue, object URLs, camera and polling in one step. */
export default function CaptureScreen() {
  const searchParams = useSearchParams();
  const [start, setStart] = useState({ epoch: 0, worldId: searchParams.get("world") });
  return (
    <CaptureScreenInner
      key={start.epoch}
      initialWorldId={start.worldId}
      onReset={() => {
        void clearCapturePhotos();
        window.history.replaceState(null, "", "/m/camera");
        setStart((s) => ({ epoch: s.epoch + 1, worldId: null }));
      }}
    />
  );
}

function CaptureScreenInner({ initialWorldId, onReset }: { initialWorldId: string | null; onReset: () => void }) {
  const [worldId, setWorldId] = useState<string | null>(initialWorldId);
  const [, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pendingFixesRef = useRef<LocationBatchPoint[]>([]);
  const isFlushingRef = useRef(false);

  const [queue, dispatch] = useReducer(queueReducer, EMPTY_QUEUE);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [camera, setCamera] = useState<CameraState>({ kind: "starting" });
  const [cameraKey, setCameraKey] = useState(0);
  const [gps, setGps] = useState<GpsState>({ status: "loading" });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const nativeInputRef = useRef<HTMLInputElement | null>(null);
  const libraryInputRef = useRef<HTMLInputElement | null>(null);
  const filesRef = useRef(new Map<string, File>());
  const urlsRef = useRef<string[]>([]);
  const controllersRef = useRef(new Map<string, AbortController>());
  const gpsRef = useRef<GpsState>(gps);

  const { status, error: statusError, refresh } = useWorldStatus(worldId);
  const [coverage, setCoverage] = useState<CaptureCoverageDto | null>(null);

  const updateCoverage = useCallback((wid: string) => {
    void getCaptureCoverage(wid)
      .then((res) => setCoverage(res))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (worldId) {
      updateCoverage(worldId);
    }
  }, [worldId, updateCoverage]);

  // Hydrate offline persistent queue from IndexedDB on mount
  useEffect(() => {
    let active = true;
    void loadPendingCapturePhotos().then((stored) => {
      if (!active || stored.length === 0) return;
      const recoveredItems: { id: string; name: string; size: number; location?: QueueItemLocation | null }[] = [];
      const previewEntries: [string, string][] = [];
      for (const item of stored) {
        const file = new File([item.blob], item.name, { type: item.type || "image/jpeg" });
        filesRef.current.set(item.id, file);
        const url = URL.createObjectURL(item.blob);
        urlsRef.current.push(url);
        previewEntries.push([item.id, url]);
        recoveredItems.push({
          id: item.id,
          name: item.name,
          size: item.size,
          location: item.location,
        });
        if (item.worldId && !worldId) {
          setWorldId(item.worldId);
        }
        if (item.sessionId && !sessionIdRef.current) {
          sessionIdRef.current = item.sessionId;
          setSessionId(item.sessionId);
        }
      }
      setPreviews((p) => ({ ...p, ...Object.fromEntries(previewEntries) }));
      dispatch({ type: "add", items: recoveredItems });
    });
    return () => {
      active = false;
    };
  }, [worldId]);

  // Automatically retry queued/retryable failed items when connection is restored
  useEffect(() => {
    const handleOnline = () => {
      queue.items
        .filter((i) => i.status === "failed" && i.retryable)
        .forEach((i) => {
          void updateCapturePhotoStatus(i.id, { status: "queued", error: null });
          dispatch({ type: "retry", id: i.id });
        });
    };
    window.addEventListener("online", handleOnline);
    return () => window.removeEventListener("online", handleOnline);
  }, [queue.items]);

  const flushFixes = useCallback(async (targetSessionId: string) => {
    if (isFlushingRef.current || pendingFixesRef.current.length === 0) return;
    isFlushingRef.current = true;
    const toSend = [...pendingFixesRef.current];
    try {
      await addSessionLocations(targetSessionId, toSend);
      pendingFixesRef.current = pendingFixesRef.current.slice(toSend.length);
    } catch {
      // Retained in buffer to retry on subsequent fix or upload completion
    } finally {
      isFlushingRef.current = false;
    }
  }, []);

  useEffect(() => {
    gpsRef.current = gps;
  }, [gps]);

  // Real continuous geolocation tracking (watchPosition) logged to session trajectory
  useEffect(() => {
    let live = true;
    if (!("geolocation" in navigator)) {
      void Promise.resolve().then(() => live && setGps({ status: "unavailable", reason: "Not supported" }));
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      (pos) => {
        if (!live) return;
        const fix = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracyM: Math.round(pos.coords.accuracy),
          altitude: pos.coords.altitude,
          heading: pos.coords.heading,
          speed: pos.coords.speed,
          timestamp: new Date(pos.timestamp).toISOString(),
        };
        setGps({
          status: "ready",
          ...fix,
        });

        const point: LocationBatchPoint = {
          latitude: fix.lat,
          longitude: fix.lng,
          altitude: fix.altitude,
          accuracy: fix.accuracyM,
          heading: fix.heading,
          speed: fix.speed,
          captured_at: fix.timestamp,
        };

        pendingFixesRef.current.push(point);
        const currentSid = sessionIdRef.current;
        if (currentSid) {
          void flushFixes(currentSid);
        }
      },
      (err) => {
        if (!live) return;
        setGps({
          status: "unavailable",
          reason: err.code === err.PERMISSION_DENIED ? "Permission denied" : "Unavailable",
        });
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 },
    );

    return () => {
      live = false;
      if (typeof navigator !== "undefined" && navigator.geolocation) {
        navigator.geolocation.clearWatch(watchId);
      }
    };
  }, [flushFixes]);

  // Real device camera. Re-runs when the user taps "Try camera again".
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    const request = navigator.mediaDevices?.getUserMedia
      ? navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1440 } },
          audio: false,
        })
      : Promise.reject(new DOMException("getUserMedia is not available", "NotSupportedError"));
    request
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          void v.play().catch(() => undefined);
        }
        setCamera({ kind: "live" });
      })
      .catch((err) => {
        if (!cancelled) setCamera({ kind: "unavailable", reason: describeCameraError(err) });
      });
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [cameraKey]);

  // Leaving the screen aborts in-flight uploads (they stay "failed, retry" in a fresh visit's world, never half-silent) and frees previews.
  useEffect(() => {
    const controllers = controllersRef.current;
    const urls = urlsRef.current;
    return () => {
      controllers.forEach((c) => c.abort());
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, []);

  // The pump: one photo at a time, strictly in order (see nextToUpload).
  useEffect(() => {
    const next = nextToUpload(queue, worldId);
    if (!next) return;
    const file = filesRef.current.get(next.id);
    dispatch({ type: "start", id: next.id });
    void updateCapturePhotoStatus(next.id, { status: "uploading", attempts: next.attempts + 1 });
    if (!file) {
      dispatch({ type: "failed", id: next.id, error: "This photo is no longer held in memory. Take it again.", retryable: false });
      return;
    }
    const controller = new AbortController();
    controllersRef.current.set(next.id, controller);
    const g = gpsRef.current;
    const fileLoc = next.location ?? (g.status === "ready" ? {
      latitude: g.lat,
      longitude: g.lng,
      altitude: g.altitude,
      accuracy: g.accuracyM,
      heading: g.heading,
      speed: g.speed,
    } : undefined);

    uploadPhoto(file, {
      worldId,
      sessionId: sessionIdRef.current ?? undefined,
      latitude: fileLoc?.latitude,
      longitude: fileLoc?.longitude,
      altitude: fileLoc?.altitude,
      accuracy: fileLoc?.accuracy,
      heading: fileLoc?.heading,
      speed: fileLoc?.speed,
      signal: controller.signal,
      onProgress: (loaded, total) => dispatch({ type: "progress", id: next.id, loaded, total }),
    })
      .then((res) => {
        const outcome = outcomeFromResponse(res);
        if (outcome.kind === "accepted") {
          dispatch({ type: "accepted", id: next.id, evidenceId: outcome.evidenceId });
          void removeCapturePhoto(next.id);
        } else {
          dispatch({ type: "rejected", id: next.id, reason: outcome.reason });
          void removeCapturePhoto(next.id);
        }
        if (res.world_id !== worldId) {
          setWorldId(res.world_id);
          window.history.replaceState(null, "", `/m/camera?world=${encodeURIComponent(res.world_id)}`);
        }
        if (res.session_id && res.session_id !== sessionIdRef.current) {
          setSessionId(res.session_id);
          sessionIdRef.current = res.session_id;
          void flushFixes(res.session_id);
        }
        refresh();
        void updateCoverage(res.world_id);
      })
      .catch((err: unknown) => {
        const retryable = isApiError(err) ? err.retryable : true;
        const msg = isApiError(err) ? err.message : "The upload failed.";
        void updateCapturePhotoStatus(next.id, { status: "failed", error: msg });
        dispatch({
          type: "failed",
          id: next.id,
          error: msg,
          retryable,
        });
      })
      .finally(() => controllersRef.current.delete(next.id));
  }, [queue, worldId, refresh, flushFixes, updateCoverage]);

  const addFiles = useCallback((files: File[]) => {
    const images = files.filter((f) => f.type.startsWith("image/") || f.type === "");
    if (images.length < files.length) setNotice("Only photos can be added here; other files were skipped.");
    else setNotice(null);
    if (images.length === 0) return;
    const g = gpsRef.current;
    const currentLoc: QueueItemLocation | null =
      g.status === "ready"
        ? {
            latitude: g.lat,
            longitude: g.lng,
            altitude: g.altitude,
            accuracy: g.accuracyM,
            heading: g.heading,
            speed: g.speed,
            timestamp: g.timestamp,
          }
        : null;

    const items = images.map((f) => {
      const id = newId();
      filesRef.current.set(id, f);
      const url = URL.createObjectURL(f);
      urlsRef.current.push(url);
      void persistCapturePhoto({
        id,
        name: f.name || "photo.jpg",
        size: f.size,
        type: f.type || "image/jpeg",
        blob: f,
        status: "queued",
        worldId,
        sessionId: sessionIdRef.current,
        location: currentLoc,
        attempts: 0,
        createdAt: Date.now(),
      });
      return { id, name: f.name || "photo.jpg", size: f.size, url, location: currentLoc };
    });
    setPreviews((p) => ({ ...p, ...Object.fromEntries(items.map((i) => [i.id, i.url])) }));
    dispatch({ type: "add", items: items.map(({ id, name, size, location }) => ({ id, name, size, location })) });
  }, [worldId]);

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    addFiles(files);
  };

  const snap = () => {
    const v = videoRef.current;
    if (!v || camera.kind !== "live" || !v.videoWidth) {
      setNotice("The camera is not ready yet.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d")?.drawImage(v, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setNotice("The frame could not be saved. Try again.");
          return;
        }
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        addFiles([new File([blob], `capture-${stamp}.jpg`, { type: "image/jpeg", lastModified: Date.now() })]);
      },
      "image/jpeg",
      0.92,
    );
  };

  const retryCamera = () => {
    setCamera({ kind: "starting" });
    setCameraKey((k) => k + 1);
  };

  const summary = summarize(queue);
  const evidenceById = new Map((status?.evidence ?? []).map((e) => [e.id, e]));
  const cardStyle = { background: "var(--bg-surface)", border: "1px solid var(--border)" };

  return (
    <div className="flex flex-col gap-4 pb-6" data-testid="capture-screen">
      {/* Viewfinder */}
      <section aria-label="Camera" className="relative overflow-hidden" style={{ background: "var(--bg-base)", height: "min(58dvh, 32rem)" }}>
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          aria-label="Live camera preview"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ display: camera.kind === "live" ? "block" : "none" }}
        />

        {camera.kind !== "live" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center" role={camera.kind === "unavailable" ? "alert" : "status"}>
            {camera.kind === "starting" ? (
              <>
                <Loader2 className="h-8 w-8 animate-spin" style={{ color: "var(--accent)" }} aria-hidden />
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  Starting the camera…
                </p>
              </>
            ) : (
              <>
                <AlertTriangle className="h-8 w-8" style={{ color: "var(--warning)" }} aria-hidden />
                <p className="max-w-xs text-sm" style={{ color: "var(--text-secondary)" }}>
                  {camera.reason}
                </p>
                <button
                  type="button"
                  onClick={retryCamera}
                  className="flex h-11 items-center gap-2 rounded-lg px-4 text-sm font-semibold"
                  style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
                >
                  <RotateCcw className="h-4 w-4" aria-hidden /> Try camera again
                </button>
              </>
            )}
          </div>
        )}

        {/* framing brackets (decorative) */}
        <div aria-hidden className="pointer-events-none absolute inset-x-[14%] inset-y-[18%]">
          {[
            "left-0 top-0 border-l-2 border-t-2",
            "right-0 top-0 border-r-2 border-t-2",
            "bottom-0 left-0 border-b-2 border-l-2",
            "bottom-0 right-0 border-b-2 border-r-2",
          ].map((c) => (
            <span key={c} className={`absolute h-8 w-8 ${c}`} style={{ borderColor: "var(--text-primary)", opacity: 0.7 }} />
          ))}
        </div>

        {/* status ribbon */}
        <div className="absolute inset-x-3 top-3 flex items-center justify-between gap-2 text-xs">
          <span
            className="flex min-h-[28px] items-center gap-1.5 rounded-full px-2.5"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          >
            <MapPin className="h-3.5 w-3.5" aria-hidden style={{ color: gps.status === "ready" ? "var(--accent)" : "var(--text-tertiary)" }} />
            {gps.status === "ready" ? `GPS ±${gps.accuracyM} m` : gps.status === "loading" ? "GPS searching…" : `No GPS: ${gps.reason}`}
          </span>
          <span
            className="font-mono-num flex min-h-[28px] items-center rounded-full px-2.5"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            aria-live="polite"
          >
            {summary.total === 0 ? "No photos yet" : `${summary.total} photo${summary.total === 1 ? "" : "s"}${summary.pending ? ` · ${summary.pending} uploading` : ""}`}
          </span>
        </div>
      </section>

      {/* Shutter row (thumb zone) */}
      <div className="-mt-1 flex items-center justify-between px-6">
        <button
          type="button"
          onClick={() => libraryInputRef.current?.click()}
          aria-label="Add photos from your library"
          className="flex h-14 w-14 flex-col items-center justify-center gap-0.5 rounded-xl text-[10px]"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          <ImageIcon className="h-5 w-5" aria-hidden />
          Library
        </button>

        <button
          type="button"
          onClick={snap}
          disabled={camera.kind !== "live"}
          aria-label="Take photo"
          className="flex h-[72px] w-[72px] items-center justify-center rounded-full disabled:opacity-40"
          style={{ border: "3px solid var(--text-primary)" }}
        >
          <span className="h-14 w-14 rounded-full" style={{ background: "var(--primary)" }} />
        </button>

        <button
          type="button"
          onClick={() => nativeInputRef.current?.click()}
          aria-label="Take a photo with the phone's camera app"
          className="flex h-14 w-14 flex-col items-center justify-center gap-0.5 rounded-xl text-[10px]"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          <Smartphone className="h-5 w-5" aria-hidden />
          Camera app
        </button>
      </div>

      <input ref={nativeInputRef} type="file" accept="image/*" capture="environment" onChange={onPick} className="hidden" tabIndex={-1} />
      <input ref={libraryInputRef} type="file" accept="image/*" multiple onChange={onPick} className="hidden" tabIndex={-1} />

      {notice && (
        <p role="status" className="mx-4 text-xs" style={{ color: "var(--warning)" }}>
          {notice}
        </p>
      )}

      <div className="flex flex-col gap-4 px-4">
        {/* Live coverage hint (subtle, non-nagging guidance directly from the coverage engine) */}
        {coverage && coverage.available && (
          <div
            role="status"
            aria-live="polite"
            className="flex items-center gap-2.5 rounded-xl px-3.5 py-2.5 text-xs leading-snug"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            data-testid="live-coverage-hint"
          >
            <Compass className="h-4 w-4 shrink-0" style={{ color: "var(--accent)" }} aria-hidden />
            <span className="font-medium text-[var(--text-primary)]">
              {coverage.hints[0] ?? coverageHeadline(coverage)}
            </span>
          </div>
        )}

        {/* World / session state */}
        {worldId ? (
          <WorldStatusCard worldId={worldId} status={status} error={statusError} refresh={refresh} />
        ) : (
          <section className="rounded-xl p-4" style={cardStyle}>
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Start with one photo of the place
            </p>
            <p className="mt-1 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              One photo gives a rough model. After each upload the engine tells you what it did with the photo and what would help next.
            </p>
          </section>
        )}

        {/* Upload queue */}
        <section aria-label="Upload queue" className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
              Photos
            </h2>
            {summary.total > 0 && (
              <span className="font-mono-num text-xs" style={{ color: "var(--text-tertiary)" }}>
                {summary.accepted} accepted
                {summary.rejected ? ` · ${summary.rejected} rejected` : ""}
                {summary.failed ? ` · ${summary.failed} failed` : ""}
              </span>
            )}
          </div>

          {queue.items.length === 0 ? (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              Nothing captured yet.
            </p>
          ) : (
            <ul className="flex flex-col gap-2" aria-live="polite">
              {[...queue.items].reverse().map((item) => (
                <QueueRow
                  key={item.id}
                  item={item}
                  preview={previews[item.id]}
                  result={photoResult(item, item.evidenceId ? evidenceById.get(item.evidenceId) : undefined)}
                  onRetry={() => {
                    void updateCapturePhotoStatus(item.id, { status: "queued", error: null });
                    dispatch({ type: "retry", id: item.id });
                  }}
                  onRemove={() => {
                    void removeCapturePhoto(item.id);
                    dispatch({ type: "remove", id: item.id });
                  }}
                />
              ))}
            </ul>
          )}
        </section>

        {worldId && (
          <button
            type="button"
            disabled={summary.pending > 0}
            onClick={onReset}
            className="h-11 rounded-lg text-sm font-medium disabled:opacity-40"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          >
            Capture a different place
          </button>
        )}
      </div>

    </div>
  );
}

function QueueRow({
  item,
  preview,
  result,
  onRetry,
  onRemove,
}: {
  item: QueueItem;
  preview: string | undefined;
  result: ReturnType<typeof photoResult>;
  onRetry: () => void;
  onRemove: () => void;
}) {
  const pct = Math.round(item.progress * 100);
  const removable = item.status === "queued" || item.status === "failed" || item.status === "rejected";
  return (
    <li
      className="flex items-start gap-3 rounded-xl p-3"
      style={{ background: "var(--bg-surface)", border: `1px solid ${result.tone === "bad" ? "var(--error)" : "var(--border)"}` }}
      data-testid="queue-row"
      data-status={item.status}
    >
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element -- local object URL of a just-captured photo
        <img src={preview} alt="" className="h-14 w-14 shrink-0 rounded-md object-cover" />
      ) : (
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-md" style={{ background: "var(--bg-elevated)" }}>
          <Camera className="h-5 w-5" style={{ color: "var(--text-tertiary)" }} aria-hidden />
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs" style={{ color: "var(--text-tertiary)" }}>
          {item.name}
          {item.attempts > 1 ? ` · attempt ${item.attempts}` : ""}
        </p>
        <p className="text-sm font-semibold" style={toneStyle(result.tone)}>
          {result.label}
        </p>
        {result.detail && (
          <p className="mt-0.5 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            {result.detail}
          </p>
        )}
        {item.status === "uploading" && (
          <div
            role="progressbar"
            aria-label={`Uploading ${item.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            className="mt-2 h-1.5 overflow-hidden rounded-full"
            style={{ background: "var(--border)" }}
          >
            <div className="h-full rounded-full" style={{ width: `${pct}%`, background: "var(--accent)" }} />
          </div>
        )}
        {item.status === "failed" && (
          <div className="mt-2 flex gap-2">
            {item.retryable && (
              <button
                type="button"
                onClick={onRetry}
                className="flex h-11 min-w-[44px] items-center gap-1.5 rounded-lg px-3 text-sm font-semibold"
                style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
              >
                <RotateCcw className="h-4 w-4" aria-hidden /> Retry
              </button>
            )}
          </div>
        )}
      </div>
      {removable && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${item.name} from the queue`}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg"
          style={{ color: "var(--text-tertiary)" }}
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
      )}
    </li>
  );
}
