"use client";

import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  GitMerge,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import {
  commitAdoption,
  getAdoption,
  proposeAdoption,
  rejectAdoption,
  type AdoptionProposalDto,
} from "@/lib/api/adoptions";
import { isApiError } from "@/lib/api/client";
import {
  ALIGNMENT_TIER_META,
  CHANGE_CATEGORY_META,
  defaultConflictResolutions,
  formatConfidencePercent,
  type ChangeCategoryKey,
} from "@/lib/adoptions/adoption-model";

interface AdoptionReviewPanelProps {
  worldId: string;
  initialAdoptionId?: string;
  onCommitted?: (versionId: string) => void;
}

export default function AdoptionReviewPanel({
  worldId,
  initialAdoptionId,
  onCommitted,
}: AdoptionReviewPanelProps) {
  const [adoptionId, setAdoptionId] = useState<string | null>(initialAdoptionId || null);
  const [proposal, setProposal] = useState<AdoptionProposalDto | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successResult, setSuccessResult] = useState<{ versionId: string; message: string } | null>(null);

  // Form input for proposing a new adoption
  const [candidateSessionId, setCandidateSessionId] = useState<string>("");
  const [activeTab, setActiveTab] = useState<ChangeCategoryKey>("new");

  // Conflict resolutions (Rule 18 protection)
  const [conflictResolutions, setConflictResolutions] = useState<
    Record<string, "keep_edit" | "accept_reconstruction">
  >({});

  // Uncertain entity inclusions
  const [acceptedUncertain, setAcceptedUncertain] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;
    if (!adoptionId) return;

    getAdoption(worldId, adoptionId)
      .then((data) => {
        if (cancelled) return;
        setProposal(data);
        setConflictResolutions(defaultConflictResolutions(data.change_set));
        const uncMap: Record<string, boolean> = {};
        for (const id of data.change_set.uncertain) {
          uncMap[id] = false;
        }
        setAcceptedUncertain(uncMap);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(isApiError(err) ? err.describe() : String(err));
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [adoptionId, worldId]);

  async function handlePropose() {
    if (!candidateSessionId.trim()) return;
    setLoading(true);
    setError(null);
    setSuccessResult(null);
    try {
      const prop = await proposeAdoption(worldId, {
        sessionId: candidateSessionId.trim(),
        minConfidence: 0.60,
      });
      setProposal(prop);
      setAdoptionId(prop.adoption_id);
      setConflictResolutions(defaultConflictResolutions(prop.change_set));
    } catch (err) {
      if (isApiError(err)) {
        setError(err.describe());
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleCommit() {
    if (!proposal) return;
    setActionLoading(true);
    setError(null);
    try {
      const res = await commitAdoption(worldId, proposal.adoption_id, {
        resolutions: conflictResolutions,
        note: `Adopted candidate session ${proposal.session_id}`,
      });
      setSuccessResult({
        versionId: res.version_id,
        message: res.message,
      });
      if (onCommitted) {
        onCommitted(res.version_id);
      }
    } catch (err) {
      if (isApiError(err)) {
        setError(err.describe());
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setActionLoading(false);
    }
  }

  async function handleReject() {
    if (!proposal) return;
    setActionLoading(true);
    setError(null);
    try {
      await rejectAdoption(worldId, proposal.adoption_id, "User rejected change set");
      setProposal(null);
      setAdoptionId(null);
    } catch (err) {
      if (isApiError(err)) {
        setError(err.describe());
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setActionLoading(false);
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* 1. Header & Proposal Trigger */}
      <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg p-5">
        <h2 className="text-sm font-semibold tracking-wide text-[var(--text-primary)] uppercase mb-2 flex items-center gap-2">
          <GitMerge className="w-4 h-4 text-[var(--accent)]" />
          Multi-Session World Adoption
        </h2>
        <p className="text-xs text-[var(--text-secondary)] mb-4">
          Align and merge new room captures or follow-up site surveys into this project without
          creating duplicate entities or overwriting manual edits (Rule 18).
        </p>

        {!proposal && !loading && (
          <div className="flex gap-2 items-center">
            <input
              type="text"
              placeholder="Enter survey Session ID (e.g. ses_site_visit_02)..."
              value={candidateSessionId}
              onChange={(e) => setCandidateSessionId(e.target.value)}
              className="flex-1 px-3 py-2 text-xs bg-[var(--bg-canvas)] border border-[var(--border-subtle)] rounded text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)]"
            />
            <button
              onClick={handlePropose}
              disabled={!candidateSessionId.trim() || loading}
              className="px-4 py-2 text-xs font-medium rounded bg-[var(--accent)] text-on-accent hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center gap-1.5"
            >
              {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              Compare & Review
            </button>
          </div>
        )}
      </div>

      {/* 2. Loading and Error Banner */}
      {loading && (
        <div className="flex items-center justify-center p-8 text-xs text-[var(--text-secondary)] gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-[var(--accent)]" />
          Calculating relative Sim(3) alignment & evaluating 5-state change set...
        </div>
      )}

      {error && (
        <div className="bg-[var(--error-subtle)] border border-[var(--error)] rounded-lg p-4 flex items-start gap-3 text-xs text-[var(--error)]">
          <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold">Adoption Check Failed</p>
            <p className="opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* 3. Successful Commit State */}
      {successResult && (
        <div className="bg-[var(--success-subtle)] border border-[var(--success)] rounded-lg p-4 flex items-start gap-3 text-xs text-[var(--success)]">
          <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold">Session Successfully Adopted into World Model</p>
            <p className="opacity-90">{successResult.message}</p>
            <p className="font-mono text-[11px] pt-1 opacity-80">
              Committed Version: {successResult.versionId}
            </p>
          </div>
        </div>
      )}

      {/* 4. Adoption Proposal Review */}
      {proposal && !successResult && (
        <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg overflow-hidden">
          {/* Alignment Banner */}
          <div className="p-4 border-b border-[var(--border-subtle)] flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-[var(--text-primary)]">
                  Session {proposal.session_id}
                </span>
                <span
                  className="text-[10px] uppercase font-bold px-2 py-0.5 rounded border"
                  style={{
                    color: ALIGNMENT_TIER_META[proposal.alignment_tier].fg,
                    backgroundColor: ALIGNMENT_TIER_META[proposal.alignment_tier].bg,
                    borderColor: ALIGNMENT_TIER_META[proposal.alignment_tier].fg,
                  }}
                >
                  {ALIGNMENT_TIER_META[proposal.alignment_tier].label}
                </span>
                <span className="text-xs text-[var(--text-secondary)]">
                  Alignment: {formatConfidencePercent(proposal.alignment_confidence)}
                </span>
              </div>
              <p className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                {ALIGNMENT_TIER_META[proposal.alignment_tier].description}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleReject}
                disabled={actionLoading}
                className="px-3 py-1.5 text-xs rounded border border-[var(--border-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors flex items-center gap-1"
              >
                <XCircle className="w-3.5 h-3.5" />
                Reject
              </button>
              <button
                onClick={handleCommit}
                disabled={actionLoading}
                className="px-4 py-1.5 text-xs font-medium rounded bg-[var(--accent)] text-on-accent hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center gap-1.5"
              >
                {actionLoading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <CheckCircle2 className="w-3.5 h-3.5" />
                )}
                Adopt Changes into World
              </button>
            </div>
          </div>

          {/* 5-State Category Tabs */}
          <div className="flex border-b border-[var(--border-subtle)] bg-[var(--bg-canvas)]">
            {(
              [
                "new",
                "matched",
                "updated",
                "uncertain",
                "conflicting",
              ] as ChangeCategoryKey[]
            ).map((cat) => {
              const meta = CHANGE_CATEGORY_META[cat];
              const count = proposal.change_set[cat]?.length || 0;
              const isSelected = activeTab === cat;
              return (
                <button
                  key={cat}
                  onClick={() => setActiveTab(cat)}
                  className={`flex-1 py-2.5 px-3 text-xs font-medium border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
                    isSelected
                      ? "border-[var(--accent)] text-[var(--text-primary)] bg-[var(--bg-surface)]"
                      : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <span>{meta.label}</span>
                  <span
                    className="text-[10px] px-1.5 py-0.2 rounded-full font-bold"
                    style={{
                      backgroundColor: count > 0 ? "var(--bg-surface-elevated)" : "transparent",
                      color: count > 0 ? meta.badgeFg : "var(--text-secondary)",
                    }}
                  >
                    {count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Tab Content */}
          <div className="p-4 space-y-3">
            <p className="text-xs text-[var(--text-secondary)]">
              {CHANGE_CATEGORY_META[activeTab].description}
            </p>

            {proposal.change_set[activeTab].length === 0 ? (
              <div className="text-center py-6 text-xs text-[var(--text-secondary)]">
                No entities categorized as {CHANGE_CATEGORY_META[activeTab].label.toLowerCase()}.
              </div>
            ) : (
              <div className="space-y-2">
                {proposal.change_set[activeTab].map((entityId) => {
                  if (activeTab === "conflicting") {
                    const currentRes = conflictResolutions[entityId] || "keep_edit";
                    return (
                      <div
                        key={entityId}
                        className="p-3 bg-[var(--bg-canvas)] border border-[var(--error)]/30 rounded flex items-center justify-between text-xs"
                      >
                        <div className="space-y-0.5">
                          <span className="font-mono text-[var(--text-primary)] font-medium">
                            {entityId}
                          </span>
                          <p className="text-[11px] text-[var(--text-secondary)]">
                            Clash detected with intentional manual user edit (Rule 18 Protected).
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() =>
                              setConflictResolutions((prev) => ({
                                ...prev,
                                [entityId]: "keep_edit",
                              }))
                            }
                            className={`px-2.5 py-1 rounded text-xs transition-colors flex items-center gap-1 ${
                              currentRes === "keep_edit"
                                ? "bg-[var(--success-subtle)] text-[var(--success)] border border-[var(--success)] font-semibold"
                                : "bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-[var(--border-subtle)]"
                            }`}
                          >
                            <ShieldCheck className="w-3.5 h-3.5" />
                            Keep Manual Edit
                          </button>
                          <button
                            onClick={() =>
                              setConflictResolutions((prev) => ({
                                ...prev,
                                [entityId]: "accept_reconstruction",
                              }))
                            }
                            className={`px-2.5 py-1 rounded text-xs transition-colors flex items-center gap-1 ${
                              currentRes === "accept_reconstruction"
                                ? "bg-[var(--error-subtle)] text-[var(--error)] border border-[var(--error)] font-semibold"
                                : "bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-[var(--border-subtle)]"
                            }`}
                          >
                            <AlertTriangle className="w-3.5 h-3.5" />
                            Overwrite with Survey
                          </button>
                        </div>
                      </div>
                    );
                  }

                  if (activeTab === "uncertain") {
                    const isAccepted = !!acceptedUncertain[entityId];
                    return (
                      <div
                        key={entityId}
                        className="p-2.5 bg-[var(--bg-canvas)] border border-[var(--border-subtle)] rounded flex items-center justify-between text-xs"
                      >
                        <div className="space-y-0.5">
                          <span className="font-mono text-[var(--text-primary)]">
                            {entityId}
                          </span>
                          <span className="ml-2 text-[10px] text-[var(--warning)] font-semibold uppercase">
                            Partially Observed
                          </span>
                        </div>
                        <label className="flex items-center gap-1.5 cursor-pointer text-xs text-[var(--text-secondary)]">
                          <input
                            type="checkbox"
                            checked={isAccepted}
                            onChange={(e) =>
                              setAcceptedUncertain((prev) => ({
                                ...prev,
                                [entityId]: e.target.checked,
                              }))
                            }
                            className="rounded accent-[var(--accent)]"
                          />
                          Include in World
                        </label>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={entityId}
                      className="p-2 bg-[var(--bg-canvas)] border border-[var(--border-subtle)] rounded flex items-center justify-between text-xs"
                    >
                      <span className="font-mono text-[var(--text-primary)]">{entityId}</span>
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                        style={{
                          color: CHANGE_CATEGORY_META[activeTab].badgeFg,
                          border: `1px solid ${CHANGE_CATEGORY_META[activeTab].badgeFg}`,
                        }}
                      >
                        {CHANGE_CATEGORY_META[activeTab].label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
