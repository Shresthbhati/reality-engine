"use client";

import React from "react";
import { AlertTriangle, Box, Loader2 } from "lucide-react";
import type { CreateReconstructionResult, WorldStatus } from "@/lib/api";
import PhotoDropZone from "./PhotoDropZone";

/**
 * The Studio never shows an unexplained empty viewport. When there is no
 * geometry to draw, this says WHY, from what the engine actually reported:
 * still working, needs more evidence, failed (with the real message), or
 * nothing has been uploaded yet.
 */
export default function NoGeometryExplanation({
  status,
  worldId,
  onCreated,
}: {
  status: WorldStatus | null;
  worldId: string | null;
  onCreated: (r: CreateReconstructionResult) => void;
}) {
  const hasEvidence = (status?.evidence_summary.count ?? 0) > 0;

  let icon = <Box className="h-7 w-7 text-[#00e5ff]" aria-hidden />;
  let title = "No model yet";
  let body: React.ReactNode =
    "Drop one or more photographs to build a spatial model. One photo gives a rough model; more views improve it.";
  let showDrop = true;

  if (status?.in_progress) {
    icon = <Loader2 className="h-7 w-7 animate-spin text-[#00e5ff]" aria-hidden />;
    title = status.state_label;
    body = "Your photos are being analysed. The first model appears here as soon as it is ready.";
    showDrop = false;
  } else if (status?.failure) {
    icon = <AlertTriangle className="h-7 w-7 text-amber-300" aria-hidden />;
    title = status.state === "NEEDS_MORE_EVIDENCE" ? "Needs more evidence" : "The model could not be built";
    body = status.failure.message ?? "The engine reported a failure without details.";
  } else if (hasEvidence && status && !status.has_model) {
    title = "No geometry was produced";
    body = "There are photos, but no reconstruction has produced geometry from them yet.";
  }

  return (
    <div
      className="absolute inset-0 z-[15] flex flex-col items-center justify-center bg-[#08090b]/95 p-6 text-center"
      data-testid="no-geometry-explanation"
      role="status"
    >
      <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-neutral-800 bg-neutral-900">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-white">{title}</h3>
      <p className="mt-2 max-w-md text-xs leading-relaxed text-neutral-400">{body}</p>
      {status?.evidence_summary.excluded.length ? (
        <ul className="mt-3 max-w-md space-y-0.5 text-[11px] text-amber-200">
          {status.evidence_summary.excluded.slice(0, 4).map((x) => (
            <li key={x.evidence_id}>
              {x.name}: {x.reason}
            </li>
          ))}
        </ul>
      ) : null}
      {showDrop && (
        <div className="mt-5 w-full max-w-lg">
          <PhotoDropZone worldId={worldId} compact onCreated={onCreated} />
        </div>
      )}
    </div>
  );
}
