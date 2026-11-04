"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AlertTriangle, Check, CircleDashed, Loader2, X } from "lucide-react";
import {
  cancelReconstruction,
  checkUploads,
  createReconstruction,
  getJobProgress,
  getWorldStatus,
  isApiError,
  retryReconstruction,
  type CheckResult,
  type CreateReconstructionResult,
  type JobView,
} from "@/lib/api";
import { BUILD, BUILD_STEPS, CHECK, HOME, HOME_STEPS, photosPlaced } from "@/lib/copy";
import { WORKING_STEPS, buildOutcome, buildStep, getStageLabel, fileProblems, formatElapsed, suggestName, verdictMessage } from "@/lib/places";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";

/** Preview URLs for picked files. Created in the effect (not memoised) so a dev-mode remount cannot leave revoked URLs. */
function useObjectUrls(files: File[], max = files.length): string[] {
  const [urls, setUrls] = useState<string[]>([]);
  useEffect(() => {
    const made = files.slice(0, max).map((f) => URL.createObjectURL(f));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the URLs are an external resource owned by this effect
    setUrls(made);
    return () => made.forEach((u) => URL.revokeObjectURL(u));
  }, [files, max]);
  return urls;
}

type Phase = "drop" | "checking" | "review" | "building";

const POLL_MS = 1500;

/**
 * Photos in, place out: drop -> check -> name -> build with real stage progress.
 * Nothing here invents a step: the build page shows exactly the stages the engine reports.
 */
export default function NewPlaceFlow({ hero = false }: { hero?: boolean }) {
  const [phase, setPhase] = useState<Phase>("drop");
  const [files, setFiles] = useState<File[]>([]);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checkFailed, setCheckFailed] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<CreateReconstructionResult | null>(null);

  const runCheck = useCallback(async (picked: File[]) => {
    setPhase("checking");
    setCheckFailed(false);
    try {
      setCheck(await checkUploads(picked));
    } catch {
      setCheck(null);
      setCheckFailed(true);
    }
    setPhase("review");
  }, []);

  const onFiles = useCallback(
    (picked: File[]) => {
      setFiles(picked);
      setName((n) => n || suggestName(picked, new Date()));
      setError(null);
      void runCheck(picked);
    },
    [runCheck],
  );

  const removeAt = (i: number) => {
    const next = files.filter((_, idx) => idx !== i);
    setFiles(next);
    if (next.length === 0) {
      setCheck(null);
      setPhase("drop");
    } else void runCheck(next);
  };

  const build = async () => {
    setError(null);
    setPhase("building");
    try {
      setCreated(await createReconstruction(files, { name: name.trim() || null }));
    } catch (err) {
      setError(isApiError(err) ? err.describe() : String(err));
      setPhase("review");
    }
  };

  return (
    <div className="flex h-full w-full flex-col items-center overflow-y-auto p-6">
      <div className={`w-full ${phase === "building" ? "max-w-xl" : "max-w-2xl"} ${hero ? "my-auto" : "pt-6"}`}>
        {phase === "drop" && (
          <>
            <h1 className="mb-1 text-center text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
              {hero ? HOME.emptyTitle : HOME.newPlace}
            </h1>
            <ol className="mb-6 flex flex-wrap justify-center gap-x-6 gap-y-1 text-xs" style={{ color: "var(--text-secondary)" }}>
              {HOME_STEPS.map((s, i) => (
                <li key={s}>
                  <span className="font-semibold" style={{ color: "var(--accent)" }}>
                    {i + 1}
                  </span>{" "}
                  {s}
                </li>
              ))}
            </ol>
            <PhotoDropZone onCreated={() => undefined} onFiles={onFiles} />
          </>
        )}

        {phase === "checking" && (
          <div className="flex flex-col items-center gap-3 py-16" role="status" aria-live="polite">
            <Loader2 className="h-7 w-7 animate-spin" style={{ color: "var(--accent)" }} aria-hidden />
            <p className="text-sm" style={{ color: "var(--text-primary)" }}>
              {CHECK.checking}
            </p>
          </div>
        )}

        {phase === "review" && (
          <ReviewStep
            files={files}
            check={check}
            checkFailed={checkFailed}
            name={name}
            setName={setName}
            error={error}
            onRemove={removeAt}
            onBack={() => {
              setPhase("drop");
              setFiles([]);
              setCheck(null);
            }}
            onBuild={() => void build()}
          />
        )}

        {phase === "building" && <BuildStep files={files} created={created} onFailedToStart={() => setPhase("review")} />}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ review

function ReviewStep(props: {
  files: File[];
  check: CheckResult | null;
  checkFailed: boolean;
  name: string;
  setName: (v: string) => void;
  error: string | null;
  onRemove: (i: number) => void;
  onBack: () => void;
  onBuild: () => void;
}) {
  const { files, check, checkFailed, name, setName, error } = props;
  const urls = useObjectUrls(files);
  const verdict = check?.summary.verdict;
  const blocked = verdict === "no_usable_photos";

  return (
    <div>
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        {CHECK.title}
      </h1>
      <p
        className="mt-2 rounded-md px-3 py-2 text-sm"
        role="status"
        style={{
          background: verdict === "good" ? "var(--success-subtle)" : "var(--warning-subtle)",
          color: verdict === "good" ? "var(--success)" : "var(--warning)",
        }}
      >
        {checkFailed ? CHECK.unavailable : verdictMessage(verdict ?? "has_problems")}
      </p>

      <ul className="mt-4 grid max-h-[46vh] gap-2 overflow-y-auto pr-1 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
        {files.map((f, i) => {
          const row = check?.files[i];
          const problems = row ? fileProblems(row) : [];
          return (
            <li key={`${f.name}-${i}`} className="flex gap-2 rounded-md p-2" style={{ background: "var(--bg-surface)", border: `1px solid ${problems.length ? "var(--warning)" : "var(--border)"}` }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={urls[i] ?? ""} alt="" className="photo-outline h-14 w-14 shrink-0 rounded object-cover" />
              <div className="min-w-0 flex-1 text-xs">
                <p className="truncate font-medium" style={{ color: "var(--text-primary)" }}>
                  {f.name}
                </p>
                {problems.length === 0 ? (
                  <p className="flex items-center gap-1" style={{ color: "var(--text-tertiary)" }}>
                    <Check className="h-3 w-3" aria-hidden /> {row?.has_gps ? CHECK.problems.hasLocation : CHECK.problems.ok}
                  </p>
                ) : (
                  problems.map((p) => (
                    <p key={p} className="flex items-start gap-1" style={{ color: "var(--warning)" }}>
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden /> {p}
                    </p>
                  ))
                )}
              </div>
              <button
                type="button"
                onClick={() => props.onRemove(i)}
                aria-label={`${CHECK.remove} ${f.name}`}
                className="h-6 w-6 shrink-0 rounded"
                style={{ color: "var(--text-tertiary)" }}
              >
                <X className="mx-auto h-3.5 w-3.5" aria-hidden />
              </button>
            </li>
          );
        })}
      </ul>

      <label className="mt-5 block text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
        {CHECK.nameLabel}
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={CHECK.namePlaceholder}
          maxLength={200}
          className="mt-1 h-10 w-full rounded-md px-3 text-sm"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border-strong)", color: "var(--text-primary)" }}
        />
      </label>
      {error && (
        <p role="alert" className="mt-3 text-xs" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
      <div className="mt-5 flex items-center justify-between">
        <button type="button" onClick={props.onBack} className="h-9 rounded-md px-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          {CHECK.back}
        </button>
        <button
          type="button"
          onClick={props.onBuild}
          disabled={blocked || files.length === 0}
          className="btn-press h-10 rounded-md px-4 text-sm font-semibold disabled:opacity-40"
          style={{ background: "var(--accent)", color: "var(--bg-base)" }}
        >
          {verdict === "has_problems" || verdict === "add_angles" ? CHECK.continueAnyway : CHECK.build}
        </button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ build

function BuildStep({ files, created, onFailedToStart }: { files: File[]; created: CreateReconstructionResult | null; onFailedToStart: () => void }) {
  const router = useRouter();
  const [retryJobId, setRetryJobId] = useState<string | null>(null);
  const jobId = retryJobId ?? created?.job_id ?? null; // a retry replaces the original job
  const [job, setJob] = useState<JobView | null>(null);
  const [startedAt, setStartedAt] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  const [failure, setFailure] = useState<string | null>(null);
  const [unplaced, setUnplaced] = useState<string[]>([]);
  const [retrying, setRetrying] = useState(false);
  const navigated = useRef(false);

  // elapsed clock, only while building
  useEffect(() => {
    if (!jobId || job?.terminal) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [jobId, job?.terminal]);

  // poll the job until it settles
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const j = await getJobProgress(jobId);
        if (!alive) return;
        setJob(j);
        if (!j.terminal) timer = setTimeout(poll, POLL_MS);
      } catch {
        if (alive) timer = setTimeout(poll, POLL_MS * 2); // a missed poll is not a failed build
      }
    };
    void poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId]);

  // when settled: which photos were not used / why it failed
  useEffect(() => {
    if (!job?.terminal || !created) return;
    const outcome = buildOutcome(job);
    if (outcome === "ready" && !navigated.current) {
      navigated.current = true;
      router.push(`/worlds/${created.world_id}`);
      return;
    }
    let alive = true;
    getWorldStatus(created.world_id)
      .then((s) => {
        if (!alive) return;
        setFailure(s.failure?.message ?? null);
        setUnplaced(s.evidence.filter((e) => e.type === "photo" && e.registered === false).map((e) => e.name));
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [job, created, router]);

  const retry = async () => {
    if (!created) return;
    setRetrying(true);
    try {
      const r = await retryReconstruction(created.session_id);
      setJob(null);
      setFailure(null);
      setStartedAt(Date.now());
      setRetryJobId(r.job_id);
    } catch (err) {
      setFailure(isApiError(err) ? err.describe() : String(err));
    }
    setRetrying(false);
  };

  const outcome = job ? buildOutcome(job) : "building";
  const current = job ? buildStep(job) : "waiting";
  const currentIdx = WORKING_STEPS.indexOf(current as (typeof WORKING_STEPS)[number]);
  const urls = useObjectUrls(files, 12);

  if (!created) {
    return (
      <div className="flex flex-col items-center gap-3 py-16" role="status">
        <Loader2 className="h-7 w-7 animate-spin" style={{ color: "var(--accent)" }} aria-hidden />
        <p className="text-sm" style={{ color: "var(--text-primary)" }}>
          {`Uploading ${files.length} photo${files.length === 1 ? "" : "s"}…`}
        </p>
        <button type="button" onClick={onFailedToStart} className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          {HOME.cancel}
        </button>
      </div>
    );
  }

  const title = outcome === "ready" ? BUILD.doneTitle : outcome === "partial" ? BUILD.partialTitle : outcome === "failed" || outcome === "cancelled" ? BUILD.failedTitle : BUILD.title;

  return (
    <div>
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        {title}
      </h1>

      <p role="status" className="sr-only">
        {outcome === "building" ? BUILD_STEPS[current] : title}
      </p>

      {outcome === "building" && (
        <>
          <ol className="mt-5 flex flex-col gap-2" aria-label={BUILD.title}>
            {WORKING_STEPS.map((s, i) => {
              const done = i < currentIdx;
              const active = i === currentIdx;
              return (
                <li key={s} className="flex items-center gap-2 text-sm" aria-current={active ? "step" : undefined} style={{ color: active ? "var(--text-primary)" : "var(--text-tertiary)" }}>
                  {done ? (
                    <Check className="h-4 w-4" style={{ color: "var(--success)" }} aria-hidden />
                  ) : active ? (
                    <Loader2 className="h-4 w-4 animate-spin" style={{ color: "var(--accent)" }} aria-hidden />
                  ) : (
                    <CircleDashed className="h-4 w-4" aria-hidden />
                  )}
                  <span>
                    {BUILD_STEPS[s]}
                    {active && job?.stage ? ` — ${getStageLabel(job.stage)}` : ""}
                  </span>
                  {active && job?.stage_counts && Object.keys(job.stage_counts).length > 0 && (
                    <span className="text-[11px] opacity-75" style={{ color: "var(--accent)" }}>
                      ({Object.entries(job.stage_counts).map(([k, v]) => `${v} ${k}`).join(", ")})
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
          {current === "waiting" && (
            <p className="mt-2 text-xs" style={{ color: "var(--text-tertiary)" }}>
              {BUILD_STEPS.waiting}
            </p>
          )}
          <p className="mt-4 text-xs tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {BUILD.elapsed}: {formatElapsed(now - startedAt)}
          </p>
          <p className="mt-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
            {BUILD.leave}
          </p>
        </>
      )}

      {urls.length > 0 && outcome === "building" && (
        <div className="mt-4 grid grid-cols-6 gap-1.5">
          {urls.map((u, i) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={i} src={u} alt="" className="aspect-square w-full rounded object-cover opacity-80" />
          ))}
        </div>
      )}

      {(outcome === "partial" || outcome === "ready") && job && (
        <div className="mt-4 text-sm" style={{ color: "var(--text-secondary)" }}>
          {job.photos_total != null && job.photos_placed != null && <p>{photosPlaced(job.photos_placed, job.photos_total)}</p>}
          {unplaced.length > 0 && (
            <ul className="mt-2 text-xs" style={{ color: "var(--warning)" }}>
              {unplaced.slice(0, 6).map((n) => (
                <li key={n}>{n}</li>
              ))}
              {unplaced.length > 6 && <li>{`…and ${unplaced.length - 6} more`}</li>}
            </ul>
          )}
        </div>
      )}

      {(outcome === "failed" || outcome === "cancelled") && (
        <div className="mt-4 text-sm" role="alert">
          {(failure || job?.failure_reason || job?.error) && (
            <p className="font-medium" style={{ color: "var(--error)" }}>
              {failure || job?.failure_reason || job?.error}
            </p>
          )}
          <p className="mt-2" style={{ color: "var(--text-secondary)" }}>
            {BUILD.failedSafe}
          </p>
        </div>
      )}

      <div className="mt-6 flex items-center gap-3">
        {outcome === "building" && jobId && (
          <button type="button" onClick={() => void cancelReconstruction(jobId)} className="btn-press h-9 rounded-md px-3 text-sm" style={{ color: "var(--text-secondary)", border: "1px solid var(--border-strong)" }}>
            {BUILD.cancel}
          </button>
        )}
        {(outcome === "failed" || outcome === "cancelled") && (
          <button type="button" disabled={retrying} onClick={() => void retry()} className="btn-press h-10 rounded-md px-4 text-sm font-semibold" style={{ background: "var(--accent)", color: "var(--bg-base)" }}>
            {BUILD.tryAgain}
          </button>
        )}
        {outcome === "partial" && (
          <button type="button" onClick={() => router.push(`/worlds/${created.world_id}`)} className="btn-press h-10 rounded-md px-4 text-sm font-semibold" style={{ background: "var(--accent)", color: "var(--bg-base)" }}>
            {BUILD.open}
          </button>
        )}
        <Link href="/" className="text-sm" style={{ color: "var(--text-secondary)" }}>
          {BUILD.home}
        </Link>
      </div>
    </div>
  );
}
