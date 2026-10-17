"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { AlertCircle, ArrowLeft, ArrowRight, Check, Info, Loader2, Target } from "lucide-react";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";
import { useSessions } from "@/lib/api";
import { attachSessionToWorld, createWorld } from "@/lib/api/worlds";
import { estimateIntent, putWorldIntent } from "@/lib/api/intent";
import {
  CAPABILITIES,
  DEFAULT_CAPABILITIES,
  SCALES,
  WIZARD_STEPS,
  scaleLabel,
  stepProblems,
  summarizeFiles,
  toggleCapability,
  type CapabilityId,
  type ResourceProfileDto,
  type ScaleId,
} from "@/lib/intent/model";
import CapabilityList from "./CapabilityList";
import ResourceProfile from "./ResourceProfile";
import ScaleCards from "./ScaleCards";

interface PickedFile {
  name: string;
  size: number;
  type: string;
}

interface EstimateResult {
  key: string;
  profile: ResourceProfileDto | null;
  error: string | null;
}

const errMsg = (e: unknown, fallback: string) => (e as Error | undefined)?.message || fallback;

export default function WorldIntentWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [scale, setScale] = useState<ScaleId | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilityId[]>(DEFAULT_CAPABILITIES);
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [files, setFiles] = useState<PickedFile[]>([]);
  const [created, setCreated] = useState<{ id: string; intentSaved: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<EstimateResult | null>(null);
  const [sessionIds, setSessionIds] = useState<string[]>([]);
  const { data: sessionData, error: sessionsError } = useSessions();
  const sessions = sessionData ?? [];

  const summary = useMemo(() => summarizeFiles(files), [files]);
  const reqKey = scale === null ? null : JSON.stringify({ scale, capabilities, ...summary });

  useEffect(() => {
    if (scale === null || reqKey === null) return;
    let stale = false;
    estimateIntent({ scale, capabilities, ...summary })
      .then((profile) => !stale && setResult({ key: reqKey, profile, error: null }))
      .catch((e) => !stale && setResult({ key: reqKey, profile: null, error: errMsg(e, "estimate failed") }));
    return () => {
      stale = true;
    };
  }, [reqKey, scale, capabilities, summary]);

  const loading = reqKey !== null && result?.key !== reqKey;
  const current = reqKey !== null && result?.key === reqKey ? result : null;
  const problems = stepProblems(step, { scale, capabilities, name });
  const locked = created !== null; // the world exists: intent can no longer be edited here

  async function run() {
    if (scale === null || busy) return;
    setBusy(true);
    setRunError(null);
    let id = created?.id ?? null;
    try {
      if (id === null) {
        id = (await createWorld({ name: name.trim(), description: location.trim() || undefined })).id;
        setCreated({ id, intentSaved: false });
        for (const sid of sessionIds) await attachSessionToWorld(id, sid);
      }
      await putWorldIntent(id, scale, capabilities);
      setCreated({ id, intentSaved: true });
    } catch (e) {
      setRunError(
        id === null
          ? `Could not create the world: ${errMsg(e, "unknown error")}`
          : `The world was created, but linking sessions or saving its intent failed:${errMsg(e, "unknown error")}`,
      );
    } finally {
      setBusy(false);
    }
  }

  const next = () => problems.length === 0 && setStep((s) => Math.min(s + 1, 3));

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto" style={{ background: "var(--bg-base)" }}>
      <header
        className="flex shrink-0 items-center justify-between gap-6 px-6 py-4"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <div className="flex items-center gap-3">
          <Target className="h-8 w-8 shrink-0" style={{ color: "var(--accent)" }} aria-hidden />
          <div>
            <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
              World Intent
            </h1>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Tell us what you&apos;re trying to construct. The engine says plainly what that needs and what it cannot promise.
            </p>
          </div>
        </div>
        <ol className="flex items-center gap-3" aria-label="Progress">
          {WIZARD_STEPS.map((label, i) => (
            <li key={label} className="flex items-center gap-3">
              <button
                type="button"
                disabled={locked || i > step}
                onClick={() => setStep(i)}
                aria-current={i === step ? "step" : undefined}
                className="flex items-center gap-2 text-xs disabled:cursor-default"
                style={{ color: i === step ? "var(--text-primary)" : "var(--text-secondary)" }}
              >
                <span
                  className="flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold"
                  style={{
                    background: i === step ? "var(--accent)" : "var(--bg-elevated)",
                    color: i === step ? "var(--bg-base)" : "var(--text-secondary)",
                    border: "1px solid var(--border)",
                  }}
                >
                  {i < step ? <Check className="h-3.5 w-3.5" aria-hidden /> : i + 1}
                </span>
                {label}
              </button>
              {i < WIZARD_STEPS.length - 1 && <span className="h-px w-8" style={{ background: "var(--border)" }} aria-hidden />}
            </li>
          ))}
        </ol>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_340px] gap-6 p-6 max-[1023px]:grid-cols-1">
        <main className="flex min-w-0 flex-col gap-5">
          {step === 0 && (
            <>
              <Section n={1} title="What are you trying to construct?" hint="Choose the scale and type of environment.">
                <ScaleCards value={scale} onChange={setScale} />
              </Section>
              <Section
                n={2}
                title="What level of reconstruction is required?"
                hint="Select the capabilities you need. They shape the evidence the engine asks for."
              >
                <CapabilityList value={capabilities} onToggle={(id) => setCapabilities((c) => toggleCapability(c, id))} />
              </Section>
              {(scale === "city" || scale === "district") && (
                <div
                  className="flex items-start gap-3 rounded-lg px-4 py-3"
                  style={{ background: "var(--info-subtle)", border: "1px solid var(--border)" }}
                >
                  <Info className="mt-0.5 h-5 w-5 shrink-0" style={{ color: "var(--info)" }} aria-hidden />
                  <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                      A {scale} is not a bigger photograph.
                    </p>
                    A {scale}-scale world needs multi-source data, spatial alignment, semantic understanding and
                    region-based processing. It cannot be created by stitching a large number of photos.
                  </div>
                </div>
              )}
            </>
          )}

          {step === 1 && (
            <Section n={3} title="Configure" hint="Name the world and say what evidence you plan to give it.">
              <div className="flex max-w-xl flex-col gap-4">
                <label className="flex flex-col gap-1.5">
                  <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    Name
                  </span>
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. South Building Floor 2"
                    className="h-9 rounded-md px-2.5 text-sm outline-none focus-visible:outline-2"
                    style={fieldStyle}
                  />
                </label>
                <label className="flex flex-col gap-1.5">
                  <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    Location <span style={{ color: "var(--text-tertiary)" }}>(optional)</span>
                  </span>
                  <input
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder="e.g. Kolkata, West Bengal, India"
                    className="h-9 rounded-md px-2.5 text-sm outline-none focus-visible:outline-2"
                    style={fieldStyle}
                  />
                  <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                    A site description. Coordinates come from the photographs&apos; own GPS, never from this text.
                  </span>
                </label>
                {sessions.length > 0 && (
                  <fieldset className="m-0 flex min-w-0 flex-col gap-1.5 border-0 p-0">
                    <legend className="mb-1 text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                      Existing capture sessions <span style={{ color: "var(--text-tertiary)" }}>(optional)</span>
                    </legend>
                    {sessions.map((s) => (
                      <label key={s.id} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                        <input
                          type="checkbox"
                          checked={sessionIds.includes(s.id)}
                          onChange={() =>
                            setSessionIds((p) => (p.includes(s.id) ? p.filter((x) => x !== s.id) : [...p, s.id]))
                          }
                          style={{ accentColor: "var(--accent)" }}
                        />
                        {s.name}
                      </label>
                    ))}
                  </fieldset>
                )}
                {sessionsError && (
                  <p role="alert" className="text-xs" style={{ color: "var(--error)" }}>
                    Could not load capture sessions: {sessionsError.message}
                  </p>
                )}
                <label className="flex flex-col gap-1.5">
                  <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    New evidence files <span style={{ color: "var(--text-tertiary)" }}>(optional)</span>
                  </span>
                  <input
                    type="file"
                    multiple
                    accept="image/*,video/*"
                    onChange={(e) =>
                      setFiles(Array.from(e.target.files ?? []).map((f) => ({ name: f.name, size: f.size, type: f.type })))
                    }
                    className="text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  />
                  <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                    {files.length === 0
                      ? "Pick the files you intend to use so the profile can reflect them. Nothing is uploaded here; you upload in the Run step."
                      : `${summary.file_count} file(s), ${summary.video_count} video(s) counted for the profile. Nothing is uploaded yet.`}
                  </span>
                </label>
              </div>
            </Section>
          )}

          {step === 2 && (
            <Section n={4} title="Review" hint="Nothing is created until you press Create world in the next step.">
              <dl className="grid max-w-xl grid-cols-[140px_1fr] gap-y-2 text-sm">
                <Row k="World type" v={scaleLabel(scale) ?? "—"} />
                <Row k="Name" v={name.trim() || "—"} />
                <Row k="Location" v={location.trim() || "Not given"} />
                <Row
                  k="Capabilities"
                  v={CAPABILITIES.filter((c) => capabilities.includes(c.id)).map((c) => c.label).join(", ") || "—"}
                />
                <Row k="Planned evidence" v={files.length ? `${summary.file_count} file(s)` : "None selected"} />
              </dl>
            </Section>
          )}

          {step === 3 && (
            <Section n={5} title="Run" hint="Create the world, save its intent, then give it evidence.">
              {runError && (
                <div
                  role="alert"
                  className="mb-3 flex items-start gap-2 rounded-md px-3 py-2 text-sm"
                  style={{ background: "var(--error-subtle)", color: "var(--error)" }}
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  <span>{runError}</span>
                </div>
              )}
              {!created?.intentSaved ? (
                <button
                  type="button"
                  onClick={run}
                  disabled={busy || problems.length > 0}
                  className="flex h-10 items-center gap-2 self-start rounded-md px-4 text-sm font-medium disabled:opacity-40"
                  style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
                >
                  {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
                  {created ? "Retry saving intent" : busy ? "Creating world…" : "Create world"}
                </button>
              ) : (
                <div className="flex max-w-xl flex-col gap-3">
                  <p className="flex items-center gap-2 text-sm" style={{ color: "var(--success)" }}>
                    <Check className="h-4 w-4" aria-hidden /> World created and intent saved.
                  </p>
                  <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    Add photographs to start the reconstruction. They join this world and it is rebuilt from all of its
                    evidence.
                  </p>
                  <PhotoDropZone worldId={created.id} onCreated={(res) => router.push(`/worlds/${res.world_id}`)} />
                  <div className="flex gap-4 text-xs">
                    <Link href={`/reconstruct?world=${encodeURIComponent(created.id)}`} className="underline" style={{ color: "var(--accent)" }}>
                      Open the ingest page for this world
                    </Link>
                    <Link href={`/worlds/${created.id}`} className="underline" style={{ color: "var(--text-secondary)" }}>
                      Skip — open the empty world
                    </Link>
                  </div>
                </div>
              )}
            </Section>
          )}

          <div className="mt-auto flex items-center justify-between gap-3 pt-2">
            <button
              type="button"
              onClick={() => setStep((s) => Math.max(s - 1, 0))}
              disabled={step === 0 || locked}
              className="flex h-9 items-center gap-2 rounded-md px-3 text-sm disabled:opacity-30"
              style={{ border: "1px solid var(--border)", color: "var(--text-primary)" }}
            >
              <ArrowLeft className="h-4 w-4" aria-hidden /> Back
            </button>
            {step < 3 && (
              <div className="flex items-center gap-3">
                {problems.length > 0 && (
                  <span role="status" className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                    {problems[0]}
                  </span>
                )}
                <button
                  type="button"
                  onClick={next}
                  disabled={problems.length > 0}
                  className="flex h-10 items-center gap-2 rounded-md px-4 text-sm font-medium disabled:opacity-40"
                  style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
                >
                  Continue to {WIZARD_STEPS[step + 1]} <ArrowRight className="h-4 w-4" aria-hidden />
                </button>
              </div>
            )}
          </div>
        </main>

        <aside className="flex min-w-0 flex-col gap-5 self-start rounded-lg p-4" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }} aria-label="Intent Summary">
          <div>
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                Intent Summary
              </h2>
              {step > 0 && !locked && (
                <button type="button" onClick={() => setStep(0)} className="rounded px-2 py-1 text-xs" style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
                  Edit
                </button>
              )}
            </div>
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              World Type
            </p>
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              {scaleLabel(scale) ?? "Not chosen yet"}
            </p>
            {location.trim() && (
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {location.trim()}
              </p>
            )}
            <ul className="mt-3 flex flex-col gap-1" aria-label="Capabilities">
              {CAPABILITIES.map((c) => {
                const on = capabilities.includes(c.id);
                return (
                  <li key={c.id} className="flex items-center gap-2 text-xs" style={{ color: on ? "var(--text-primary)" : "var(--text-tertiary)" }}>
                    <span
                      className="flex h-4 w-4 items-center justify-center rounded-sm"
                      style={{ background: on ? "var(--accent)" : "transparent", border: "1px solid var(--border)", color: "var(--bg-base)" }}
                      aria-hidden
                    >
                      {on && <Check className="h-3 w-3" />}
                    </span>
                    <span>
                      {c.label}
                      <span className="sr-only">{on ? " (selected)" : " (not selected)"}</span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
          <ResourceProfile profile={current?.profile ?? null} loading={loading} error={current?.error ?? null} />
          <p className="text-[10px]" style={{ color: "var(--text-tertiary)" }}>
            {SCALES.length} scales · figures are detected on this machine or derived from this database&apos;s past runs;
            where neither exists the profile says Unknown.
          </p>
        </aside>
      </div>
    </div>
  );
}

const fieldStyle = { background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" };

function Section({ n, title, hint, children }: { n: number; title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          <span
            className="flex h-6 w-6 items-center justify-center rounded-full text-xs"
            style={{ background: "var(--accent)", color: "var(--bg-base)" }}
            aria-hidden
          >
            {n}
          </span>
          {title}
        </h2>
        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
          {hint}
        </p>
      </div>
      {children}
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt style={{ color: "var(--text-tertiary)" }}>{k}</dt>
      <dd style={{ color: "var(--text-primary)" }}>{v}</dd>
    </>
  );
}
