"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Camera, Image as ImageIcon, Video, Check, RotateCcw, AlertTriangle } from "lucide-react";
import { getWorld, createSession, attachSessionToWorld, uploadEvidence, isApiError } from "@/lib/api";
import type { WorldRow } from "@/lib/types";

type GpsState =
  | { status: "loading" }
  | { status: "ready"; lat: number; lng: number; accuracyM: number }
  | { status: "unavailable"; reason: string };

type CaptureMode = "photo" | "video";
type FlowStep = "camera" | "preview" | "session" | "creating" | "done";

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

  const [world, setWorld] = useState<WorldRow | null | "loading">(worldId ? "loading" : null);
  useEffect(() => {
    if (!worldId) {
      setWorld(null);
      return;
    }
    let cancelled = false;
    setWorld("loading");
    getWorld(worldId)
      .then((r) => {
        if (!cancelled) setWorld(r.row);
      })
      .catch(() => {
        if (!cancelled) setWorld(null);
      });
    return () => {
      cancelled = true;
    };
  }, [worldId]);

  const [mode, setMode] = useState<CaptureMode>("photo");
  const [step, setStep] = useState<FlowStep>("camera");
  // Initialize GPS state based on whether geolocation is available
  const [gps, setGps] = useState<GpsState>(() => {
    if (typeof navigator === "undefined" || !("geolocation" in navigator)) {
      return { status: "unavailable", reason: "Not supported" };
    }
    return { status: "loading" };
  });
  const [captured, setCaptured] = useState<{ file: File; url: string; at: Date } | null>(null);
  const [sessionName, setSessionName] = useState("");
  const [createdSessionId, setCreatedSessionId] = useState<string | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mountedRef = useRef(false);

  // Real geolocation — no fabricated coordinates. Denied/unsupported states
  // are shown honestly rather than silently defaulted.
  useEffect(() => {
    mountedRef.current = true;
    if (!("geolocation" in navigator)) {
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        if (!mountedRef.current) return;
        setGps({
          status: "ready",
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracyM: Math.round(pos.coords.accuracy),
        });
      },
      (err) => {
        if (!mountedRef.current) return;
        setGps({ status: "unavailable", reason: err.code === err.PERMISSION_DENIED ? "Permission denied" : "Unavailable" });
      },
      { enableHighAccuracy: true, timeout: 10000 },
    );
    return () => { mountedRef.current = false; };
  }, []);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCaptured({ file, url: URL.createObjectURL(file), at: new Date() });
    setStep("preview");
  };

  const retake = () => {
    if (captured) URL.revokeObjectURL(captured.url);
    setCaptured(null);
    setStep("camera");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const useCapture = () => {
    setSessionName(`Capture — ${captured?.at.toLocaleDateString()}`);
    setStep("session");
  };

  const createSessionAndUpload = async () => {
    if (!captured || !sessionName.trim()) return;
    setCreateError(null);
    setStep("creating");
    try {
      const input = {
        name: sessionName.trim(),
        captured_at: captured.at.toISOString(),
        location:
          gps.status === "ready"
            ? { latitude: gps.lat, longitude: gps.lng, accuracy: gps.accuracyM }
            : undefined,
      };
      const session = await createSession(input);
      if (world) {
        await attachSessionToWorld(world.id, session.id).catch(() => {
          // Session is created regardless; the World link is best-effort here —
          // the user can attach it from the Session detail screen if this fails.
        });
      }
      await uploadEvidence(captured.file, session.id);
      setCreatedSessionId(session.id);
      setStep("done");
    } catch (err) {
      setCreateError(isApiError(err) ? err.describe() : "Failed to create Session.");
      setStep("session");
    }
  };

  if (step === "done") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 h-full p-6 text-center">
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center"
          style={{ background: "var(--success-subtle)", color: "var(--success)" }}
        >
          <Check className="w-6 h-6" />
        </div>
        <p className="text-sm" style={{ color: "var(--text-primary)" }}>
          Session created and Evidence uploaded.
        </p>
        {createdSessionId && (
          <Link
            href={`/m/sessions/${createdSessionId}`}
            className="text-sm font-medium"
            style={{ color: "var(--accent)" }}
          >
            View Session →
          </Link>
        )}
        <div className="flex gap-3 mt-2">
          <button
            type="button"
            onClick={() => router.push("/m")}
            className="h-10 px-4 rounded-md text-sm font-medium"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          >
            Back to Home
          </button>
          <button
            type="button"
            onClick={() => {
              setCreatedSessionId(null);
              retake();
            }}
            className="h-10 px-4 rounded-md text-sm font-medium"
            style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
          >
            Capture Another
          </button>
        </div>
      </div>
    );
  }

  if ((step === "session" || step === "creating") && captured) {
    return (
      <div className="flex flex-col gap-5 p-4">
        <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
          New Session
        </h1>

        {/* eslint-disable-next-line @next/next/no-img-element -- dynamic camera capture, not a static asset */}
        <img src={captured.url} alt="Captured preview" className="w-full h-40 object-cover rounded-lg" />

        <Field label="Name">
          <input
            value={sessionName}
            onChange={(e) => setSessionName(e.target.value)}
            disabled={step === "creating"}
            className="mobile-input"
          />
        </Field>

        <Field label="Location" hint={gps.status === "ready" ? `±${gps.accuracyM}m` : undefined}>
          <div className="mobile-input flex items-center" style={{ color: gps.status === "ready" ? "var(--text-primary)" : "var(--text-tertiary)" }}>
            {gps.status === "ready" ? `${gps.lat.toFixed(4)}°, ${gps.lng.toFixed(4)}°` : "Unavailable"}
          </div>
        </Field>

        <Field label="World">
          <div className="mobile-input flex items-center" style={{ color: world && world !== "loading" ? "var(--text-primary)" : "var(--text-tertiary)" }}>
            {world === "loading" ? "Loading…" : world ? world.name : "None — standalone Session"}
          </div>
        </Field>

        {createError && (
          <div className="flex items-start gap-2 text-xs" style={{ color: "var(--error)" }}>
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{createError}</span>
          </div>
        )}

        <button
          type="button"
          disabled={!sessionName.trim() || step === "creating"}
          onClick={createSessionAndUpload}
          className="h-12 rounded-lg text-sm font-semibold disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {step === "creating" ? "Creating…" : "Create Session"}
        </button>
        <button
          type="button"
          disabled={step === "creating"}
          onClick={retake}
          className="h-11 rounded-md text-sm font-medium disabled:opacity-40"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          Retake
        </button>
      </div>
    );
  }

  if (step === "preview" && captured) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex-1 relative">
          {/* eslint-disable-next-line @next/next/no-img-element -- dynamic camera capture, not a static asset */}
          <img src={captured.url} alt="Captured preview" className="absolute inset-0 w-full h-full object-cover" />
        </div>
        <div className="shrink-0 p-4 flex flex-col gap-3" style={{ background: "var(--bg-surface)" }}>
          <div className="flex items-center justify-between text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
            <span>{captured.at.toLocaleString()}</span>
            <span>{gps.status === "ready" ? `${gps.lat.toFixed(4)}°, ${gps.lng.toFixed(4)}°` : "No GPS lock"}</span>
          </div>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={retake}
              className="flex-1 flex items-center justify-center gap-2 h-12 rounded-lg text-sm font-medium"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
            >
              <RotateCcw className="w-4 h-4" />
              Retake
            </button>
            <button
              type="button"
              onClick={useCapture}
              className="flex-1 flex items-center justify-center gap-2 h-12 rounded-lg text-sm font-semibold"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              <Check className="w-4 h-4" />
              Use Capture
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full" style={{ background: "#000" }}>
      {/* Camera surface — no live preview stream is wired (file-input capture
          opens the OS camera directly); this is the pre-capture HUD. */}
      <div className="flex-1 relative flex items-center justify-center">
        <Camera className="w-10 h-10" style={{ color: "rgba(255,255,255,0.15)" }} />

        <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
          <span
            className="text-xs font-mono-num px-2 h-6 flex items-center rounded"
            style={{ background: "rgba(8,9,11,0.7)", color: "var(--text-secondary)" }}
          >
            {world === "loading" ? "Loading World…" : world ? world.name : "No World selected"}
          </span>
        </div>

        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-center gap-4">
          <HudDot label="GPS" active={gps.status === "ready"} pending={gps.status === "loading"} />
          <HudDot label="Track" active={false} />
          <HudDot label="Quality" active={false} />
        </div>
      </div>

      <div className="shrink-0 p-5 flex items-center justify-center gap-8" style={{ background: "#000" }}>
        <button
          type="button"
          onClick={() => setMode("photo")}
          className="flex flex-col items-center gap-1 text-xs"
          style={{ color: mode === "photo" ? "var(--accent)" : "rgba(255,255,255,0.4)" }}
        >
          <ImageIcon className="w-4 h-4" />
          Photo
        </button>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label={mode === "photo" ? "Capture photo" : "Capture video"}
          className="w-16 h-16 rounded-full flex items-center justify-center"
          style={{ border: "3px solid var(--accent)" }}
        >
          <span className="w-12 h-12 rounded-full" style={{ background: "var(--accent)" }} />
        </button>

        <button
          type="button"
          onClick={() => setMode("video")}
          className="flex flex-col items-center gap-1 text-xs"
          style={{ color: mode === "video" ? "var(--accent)" : "rgba(255,255,255,0.4)" }}
        >
          <Video className="w-4 h-4" />
          Video
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={mode === "photo" ? "image/*" : "video/*"}
        capture="environment"
        onChange={handleFile}
        className="hidden"
      />

      <style jsx>{`
        .mobile-input {
          height: 40px;
          padding: 0 12px;
          border-radius: 8px;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          font-size: 14px;
          outline: none;
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}

function HudDot({ label, active, pending }: { label: string; active: boolean; pending?: boolean }) {
  const color = active ? "var(--accent)" : pending ? "var(--warning)" : "rgba(255,255,255,0.25)";
  return (
    <div className="flex items-center gap-1.5 text-xs" style={{ color: active ? "var(--text-primary)" : "rgba(255,255,255,0.4)" }}>
      <span className={`w-1.5 h-1.5 rounded-full ${pending ? "animate-pulse" : ""}`} style={{ background: color }} />
      {label}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          {label}
        </span>
        {hint && (
          <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {hint}
          </span>
        )}
      </div>
      {children}
    </label>
  );
}
