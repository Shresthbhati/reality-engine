"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Camera, Image as ImageIcon, Video, Check, RotateCcw } from "lucide-react";
import { getWorld } from "@/lib/data";

type GpsState =
  | { status: "loading" }
  | { status: "ready"; lat: number; lng: number; accuracyM: number }
  | { status: "unavailable"; reason: string };

type CaptureMode = "photo" | "video";
type FlowStep = "camera" | "preview" | "session" | "done";

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
  const world = worldId ? getWorld(worldId) : undefined;

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
          Capture saved locally.
        </p>
        <p className="text-xs max-w-xs" style={{ color: "var(--text-tertiary)" }}>
          Session creation isn&apos;t wired to a backend yet — nothing was uploaded or persisted.
        </p>
        <button
          type="button"
          onClick={() => router.push("/m")}
          className="mt-2 h-10 px-4 rounded-md text-sm font-medium"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          Back to Home
        </button>
      </div>
    );
  }

  if (step === "session" && captured) {
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
            className="mobile-input"
          />
        </Field>

        <Field label="Location" hint={gps.status === "ready" ? `±${gps.accuracyM}m` : undefined}>
          <div className="mobile-input flex items-center" style={{ color: gps.status === "ready" ? "var(--text-primary)" : "var(--text-tertiary)" }}>
            {gps.status === "ready" ? `${gps.lat.toFixed(4)}°, ${gps.lng.toFixed(4)}°` : "Unavailable"}
          </div>
        </Field>

        <Field label="World">
          <div className="mobile-input flex items-center" style={{ color: world ? "var(--text-primary)" : "var(--text-tertiary)" }}>
            {world ? world.name : "None — standalone Session"}
          </div>
        </Field>

        <button
          type="button"
          disabled={!sessionName.trim()}
          onClick={() => setStep("done")}
          className="h-12 rounded-lg text-sm font-semibold disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          Create Session
        </button>
        <button
          type="button"
          onClick={retake}
          className="h-11 rounded-md text-sm font-medium"
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
            {world ? world.name : "No World selected"}
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
