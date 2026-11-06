"use client";

import React, { useState } from "react";
import { Camera, CheckCircle2, ChevronRight, Eye, Sparkles, X, MapPin } from "lucide-react";
import type { CaptureRequestItem } from "@/lib/api/capture-requests";
import { describeSuggestedPose, formatExpectedGainPct } from "@/lib/studio/capture-requests-model";

export interface CaptureRequestCardProps {
  request: CaptureRequestItem;
  onAddPhoto?: (request: CaptureRequestItem) => void;
  onDismiss?: (requestId: string) => void;
  onSatisfy?: (requestId: string) => void;
  className?: string;
}

export function CaptureRequestCard({
  request,
  onAddPhoto,
  onDismiss,
  onSatisfy,
  className = "",
}: CaptureRequestCardProps) {
  const [updating, setUpdating] = useState(false);

  const gainBadge = formatExpectedGainPct(request.expected_gain);
  const regionType = request.target_region?.region_type || "scene";
  const pose = request.suggested_poses?.[0];
  const poseGuidance = pose ? describeSuggestedPose(pose) : null;

  return (
    <div
      data-testid={`capture-request-card-${request.id}`}
      className={`rounded-xl border border-line/80 bg-surface/90 p-4 shadow-lg transition-all hover:border-line-strong/80 ${className}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-info-subtle px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-info border border-info/20">
            {regionType}
          </span>
          <span className="flex items-center gap-1 rounded-md bg-success-subtle px-2 py-0.5 text-[10px] font-semibold text-success border border-success/20">
            <Sparkles className="h-3 w-3" />
            {gainBadge}
          </span>
          {request.status !== "open" && (
            <span className="rounded-md bg-elevated px-2 py-0.5 text-[10px] font-medium text-fg-muted">
              {request.status}
            </span>
          )}
        </div>

        {request.status === "open" && onDismiss && (
          <button
            type="button"
            disabled={updating}
            onClick={() => {
              setUpdating(true);
              onDismiss(request.id);
            }}
            aria-label="Dismiss capture request"
            className="rounded p-1 text-fg-muted hover:bg-elevated hover:text-fg"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <p className="mt-2 text-sm font-medium text-fg">
        {request.instruction}
      </p>

      {poseGuidance && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-fg-muted">
          <MapPin className="h-3.5 w-3.5 text-warning shrink-0" />
          <span>{poseGuidance}</span>
        </div>
      )}

      {request.status === "open" && (
        <div className="mt-4 flex items-center justify-between gap-2 border-t border-line-subtle/80 pt-3">
          {onSatisfy && (
            <button
              type="button"
              disabled={updating}
              onClick={() => {
                setUpdating(true);
                onSatisfy(request.id);
              }}
              className="flex items-center gap-1 text-xs text-fg-muted hover:text-success"
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              <span>Mark satisfied</span>
            </button>
          )}

          {onAddPhoto && (
            <button
              type="button"
              onClick={() => onAddPhoto(request)}
              className="ml-auto flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-on-primary shadow-sm hover:bg-primary-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-info transition-colors"
            >
              <Camera className="h-3.5 w-3.5" />
              <span>Add photo</span>
              <ChevronRight className="h-3 w-3" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
