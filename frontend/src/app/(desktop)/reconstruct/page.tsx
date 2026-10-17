"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";

/**
 * The primary product action: give the engine photographs, get a world.
 * No World, Session, backend or version choices come first; the result opens
 * straight in the Studio. `?world=<id>` (set by the World Intent wizard) adds
 * the photographs to that existing world instead of creating a new one.
 */
function CreateReconstruction() {
  const router = useRouter();
  const worldId = useSearchParams().get("world");
  return (
    <div className="flex h-full w-full flex-col items-center justify-center p-6">
      <h1 className="mb-1 text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
        {worldId ? "Add photographs to your world" : "Create a reconstruction"}
      </h1>
      <p className="mb-6 max-w-lg text-center text-xs" style={{ color: "var(--text-tertiary)" }}>
        Give Reality Engine photographs of a real place. It builds the best spatial model the evidence
        supports, says plainly what it is unsure about, and improves the same model as you add more views.
      </p>
      <div className="w-full max-w-xl">
        <PhotoDropZone worldId={worldId} onCreated={(res) => router.push(`/worlds/${res.world_id}`)} />
      </div>
    </div>
  );
}

export default function CreateReconstructionPage() {
  return (
    <Suspense fallback={null}>
      <CreateReconstruction />
    </Suspense>
  );
}
