"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";
import GuidedFlow from "@/components/reconstruct/GuidedFlow";
import { retryReconstruction, useWorldStatus } from "@/lib/api";
import { ctaTarget, flowNav } from "@/lib/reconstruct/flow-model";

const LINK_CLS = "rounded text-xs underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2";

/**
 * The primary product action: give the engine photographs, get a world.
 * No World, Session, backend or version choices come first; the result opens
 * straight in the Studio. `?world=<id>` (set by the World Intent wizard) adds
 * the photographs to that existing world instead of creating a new one.
 * Focus order: back link, heading, the five steps, the photo drop zone, next link.
 */
function CreateReconstruction() {
  const router = useRouter();
  const worldId = useSearchParams().get("world");
  // With ?world= the flow reflects that world's real state; without it nothing exists yet, so step 1 is the only action.
  const { status, loading, refresh } = useWorldStatus(worldId);
  const nav = flowNav(worldId, Boolean(status?.has_model));
  return (
    <section aria-labelledby="reconstruct-title" className="flex h-full w-full flex-col items-center justify-center overflow-y-auto p-6">
      {nav.back && (
        <div className="mb-3 w-full max-w-4xl">
          <Link href={nav.back.href} className={LINK_CLS} style={{ color: "var(--accent)" }}>&larr; {nav.back.label}</Link>
        </div>
      )}
      <h1 id="reconstruct-title" className="mb-1 text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
        {worldId ? "Add photographs to your world" : "Create a reconstruction"}
      </h1>
      <p className="mb-6 max-w-lg text-center text-xs" style={{ color: "var(--text-tertiary)" }}>
        Give Reality Engine photographs of a real place. It builds the best spatial model the evidence
        supports, says plainly what it is unsure about, and improves the same model as you add more views.
      </p>
      {worldId && (
        <div className="mb-4 flex items-center justify-between gap-3 rounded-lg border border-info/30 bg-info-subtle px-4 py-2.5 text-xs text-info w-full max-w-xl">
          <span>This world has intelligent capture requests to resolve occlusions and low-coverage surfaces.</span>
          <Link href={`/capture?world=${worldId}`} className="shrink-0 font-semibold underline hover:text-fg">
            View Requests &rarr;
          </Link>
        </div>
      )}
      <div className="mb-6 w-full max-w-4xl">
        {worldId && loading ? (
          <p className="text-center text-xs text-fg-muted" role="status">Reading this world&apos;s state…</p>
        ) : (
        <GuidedFlow
          status={status}
          showTips
          onCta={(kind) => {
            const t = ctaTarget(kind, worldId);
            if (t.type === "route") router.push(t.href);
            else if (t.type === "retry" && status?.job) void retryReconstruction(status.job.session_id).then(refresh, refresh);
            else document.getElementById("photo-drop")?.scrollIntoView({ block: "center" });
          }}
        />
        )}
      </div>
      <div id="photo-drop" className="w-full max-w-xl">
        <PhotoDropZone worldId={worldId} onCreated={(res) => router.push(`/worlds/${res.world_id}`)} />
      </div>
      {nav.next && (
        <div className="mt-4 w-full max-w-xl text-right">
          <Link href={nav.next.href} className={LINK_CLS} style={{ color: "var(--accent)" }}>{nav.next.label} &rarr;</Link>
        </div>
      )}
    </section>
  );
}

export default function CreateReconstructionPage() {
  return (
    <Suspense fallback={null}>
      <CreateReconstruction />
    </Suspense>
  );
}
