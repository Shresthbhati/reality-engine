"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { AlertTriangle, Camera, Check, Loader2, Lightbulb, RotateCcw } from "lucide-react";
import { createReconstruction, isApiError, modelStateCopy, useWorldStatus } from "@/lib/api";

/**
 * Mobile capture: OPEN CAMERA -> CAPTURE -> SAVE -> ANALYZE -> BUILD ->
 * SHOW STATE -> OFFER THE NEXT BEST CAPTURE.
 *
 * The user never names a session, picks a world, or attaches anything. The
 * first photo creates the world (its id rides in `?world=`), later photos
 * join the SAME world, and every photo is saved before the engine even
 * starts. GPS is attached only when the device actually reports it.
 */

type GpsState =
  | { status: "loading" }
  | { status: "ready"; lat: number; lng: number; accuracyM: number }
  | { status: "unavailable"; reason: string };

type Step = "camera" | "preview" | "saving" | "result";

export default function MobileCameraPage() {
  return (
    <Suspense>
      <MobileCameraPageInner />
    </Suspense>
  );
}

function MobileCameraPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const worldId = searchParams.get("world");

  const [step, setStep] = useState<Step>("camera");
  const [gps, setGps] = useState<GpsState>(() =>
    typeof navigator === "undefined" || !("geolocation" in navigator)
      ? { status: "unavailable", reason: "Not supported" }
      : { status: "loading" },
  );
  const [captured, setCaptured] = useState<{ file: File; url: string; at: Date } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shots, setShots] = useState(0);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Real geolocation only. Denied/unsupported is shown, never defaulted.
  useEffect(() => {
    if (typeof navigator === "undefined" || !("geolocation" in navigator)) return;
    let live = true;
    navigator.geolocation.getCurrentPosition(
      (pos) =>
        live &&
        setGps({
          status: "ready",
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracyM: Math.round(pos.coords.accuracy),
        }),
      (err) =>
        live &&
        setGps({ status: "unavailable", reason: err.code === err.PERMISSION_DENIED ? "Permission denied" : "Unavailable" }),
      { enableHighAccuracy: true, timeout: 10000 },
    );
    return () => {
      live = false;
    };
  }, []);

  const { status } = useWorldStatus(step === "result" ? worldId : null);

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setCaptured({ file, url: URL.createObjectURL(file), at: new Date() });
    setError(null);
    setStep("preview");
  };

  const retake = () => {
    if (captured) URL.revokeObjectURL(captured.url);
    setCaptured(null);
    setStep("camera");
  };

  const save = async () => {
    if (!captured) return;
    setStep("saving");
    setError(null);
    try {
      const res = await createReconstruction([captured.file], {
        worldId,
        latitude: gps.status === "ready" ? gps.lat : undefined,
        longitude: gps.status === "ready" ? gps.lng : undefined,
      });
      setShots((n) => n + 1);
      if (res.world_id !== worldId) router.replace(`/m/camera?world=${encodeURIComponent(res.world_id)}`);
      URL.revokeObjectURL(captured.url);
      setCaptured(null);
      setStep("result");
    } catch (err) {
      const details = isApiError(err) ? (err.details as { message?: string } | null) : null;
      setError(details?.message ?? (isApiError(err) ? err.describe() : "The photo could not be saved."));
      setStep("preview");
    }
  };

  if (step === "result") {
    const model = status?.model;
    const copy = modelStateCopy(model?.model_state);
    return (
      <div className="flex h-full flex-col gap-4 p-5" data-testid="mobile-result">
        <div className="flex items-center gap-3">
          <div
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full"
            style={{ background: "var(--success-subtle)", color: "var(--success)" }}
          >
            <Check className="h-5 w-5" aria-hidden />
          </div>
          <div>
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Photo saved{shots > 1 ? ` (${shots} this session)` : ""}
            </p>
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Your photos are kept even if the model cannot be built yet.
            </p>
          </div>
        </div>

        <div className="rounded-lg p-3" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2 text-sm font-medium" style={{ color: "var(--text-primary)" }} role="status">
            {status?.in_progress || !status ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
            {status ? status.state_label : "Starting…"}
          </div>
          {model && (
            <p className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
              Current spatial reconstruction: <strong>{copy.label}</strong>. {copy.detail}{" "}
              {model.images_registered} of {model.images_used} photos placed.
            </p>
          )}
          {status?.failure && (
            <p className="mt-2 flex items-start gap-1.5 text-xs" style={{ color: "var(--error)" }}>
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              {status.failure.message ?? "The engine could not finish."}
            </p>
          )}
        </div>

        {status && status.guidance.length > 0 && (
          <div className="rounded-lg p-3" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
            <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
              <Lightbulb className="h-3.5 w-3.5" aria-hidden /> Best next photo
            </p>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {status.guidance[0].message}
            </p>
          </div>
        )}

        <div className="mt-auto flex flex-col gap-2">
          <button
            type="button"
            onClick={() => setStep("camera")}
            className="h-12 rounded-lg text-sm font-semibold"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            Take another photo
          </button>
          {worldId && (
            <Link
              href={`/m/worlds/${worldId}`}
              className="flex h-11 items-center justify-center rounded-md text-sm font-medium"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            >
              View model
            </Link>
          )}
        </div>
      </div>
    );
  }

  if ((step === "preview" || step === "saving") && captured) {
    return (
      <div className="flex h-full flex-col">
        <div className="relative flex-1">
          {/* eslint-disable-next-line @next/next/no-img-element -- dynamic camera capture, not a static asset */}
          <img src={captured.url} alt="Captured preview" className="absolute inset-0 h-full w-full object-cover" />
        </div>
        <div className="shrink-0 space-y-3 p-4" style={{ background: "var(--bg-surface)" }}>
          <div className="flex items-center justify-between font-mono-num text-xs" style={{ color: "var(--text-tertiary)" }}>
            <span>{captured.at.toLocaleString()}</span>
            <span>{gps.status === "ready" ? `${gps.lat.toFixed(4)}°, ${gps.lng.toFixed(4)}° (±${gps.accuracyM} m)` : "No GPS lock"}</span>
          </div>
          {error && (
            <div role="alert" className="flex items-start gap-2 text-xs" style={{ color: "var(--error)" }}>
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <span>{error}</span>
            </div>
          )}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={retake}
              disabled={step === "saving"}
              className="flex h-12 flex-1 items-center justify-center gap-2 rounded-lg text-sm font-medium disabled:opacity-40"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Retake
            </button>
            <button
              type="button"
              onClick={save}
              disabled={step === "saving"}
              className="flex h-12 flex-1 items-center justify-center gap-2 rounded-lg text-sm font-semibold disabled:opacity-60"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              {step === "saving" ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Check className="h-4 w-4" aria-hidden />}
              {step === "saving" ? "Saving…" : "Use photo"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col" style={{ background: "#000" }}>
      <div className="relative flex flex-1 items-center justify-center">
        <Camera className="h-10 w-10" style={{ color: "rgba(255,255,255,0.15)" }} aria-hidden />
        <p className="absolute left-4 right-4 top-4 text-center text-xs" style={{ color: "rgba(255,255,255,0.6)" }}>
          {worldId
            ? "Adding to your model. Move to a new position for the next photo."
            : "Take a photo of the place. One photo gives a rough model; more views improve it."}
        </p>
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-center gap-4">
          <span className="flex items-center gap-1.5 text-xs" style={{ color: "rgba(255,255,255,0.5)" }}>
            <span
              className={`h-1.5 w-1.5 rounded-full ${gps.status === "loading" ? "animate-pulse" : ""}`}
              style={{ background: gps.status === "ready" ? "var(--accent)" : gps.status === "loading" ? "var(--warning)" : "rgba(255,255,255,0.25)" }}
            />
            GPS {gps.status === "ready" ? "locked" : gps.status === "loading" ? "searching" : "off"}
          </span>
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-center p-5" style={{ background: "#000" }}>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Capture photo"
          className="flex h-16 w-16 items-center justify-center rounded-full"
          style={{ border: "3px solid var(--accent)" }}
        >
          <span className="h-12 w-12 rounded-full" style={{ background: "var(--accent)" }} />
        </button>
      </div>

      <input ref={fileInputRef} type="file" accept="image/*" capture="environment" onChange={onFile} className="hidden" />
    </div>
  );
}
