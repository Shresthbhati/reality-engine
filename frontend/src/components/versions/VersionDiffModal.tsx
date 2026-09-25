"use client";

import { useEffect, useState } from "react";
import { GitCompare, X } from "lucide-react";
import { getWorldDiff } from "@/lib/api/worlds";
import type { WorldDiffDto, EntityDiffDto } from "@/lib/api/types";

interface VersionDiffModalProps {
  worldId: string;
  isOpen: boolean;
  onClose: () => void;
  baseVersion?: string;
  /** Real version id, or omitted to mean "current version" -- never the
   * literal string "latest", which is not a valid WorldStore version id
   * and always 404s. */
  headVersion?: string;
  onSelectEntity?: (id: string) => void;
}

export default function VersionDiffModal({
  worldId,
  isOpen,
  onClose,
  baseVersion = "",
  headVersion = "",
  onSelectEntity,
}: VersionDiffModalProps) {
  const [diff, setDiff] = useState<WorldDiffDto | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    if (!worldId || !baseVersion) {
      Promise.resolve().then(() => {
        if (cancelled) return;
        setError("No base version specified. Create and commit world revisions to compare lineage differentials.");
        setLoading(false);
        setDiff(null);
      });
      return () => {
        cancelled = true;
      };
    }
    Promise.resolve().then(() => {
      if (!cancelled) {
        setLoading(true);
        setError(null);
      }
    });
    getWorldDiff(worldId, baseVersion, headVersion)
      .then((data) => {
        if (cancelled) return;
        setDiff(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err?.message || "No version differential available between these versions.");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, worldId, baseVersion, headVersion]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm select-none"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl max-h-[85vh] rounded-lg flex flex-col overflow-hidden bg-[#0e1013] border border-[#1f222b] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 h-14 border-b border-[#1f222b] bg-[#12141a]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md flex items-center justify-center bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/20">
              <GitCompare className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">WorldStore Version Comparison</h3>
              <p className="text-[11px] font-mono text-neutral-400">
                Comparing <span className="text-[#00e5ff]">{baseVersion || "—"}</span> → <span className="text-[#2ecc71]">{headVersion || "Current"}</span>
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-xs">
          {loading ? (
            <div className="py-12 flex flex-col items-center justify-center gap-3">
              <div className="w-6 h-6 border-2 border-[#00e5ff] border-t-transparent rounded-full animate-spin" />
              <p className="text-neutral-400 font-mono text-xs">Computing WorldStore geometric & semantic delta...</p>
            </div>
          ) : error ? (
            <div className="py-12 px-6 rounded-lg border border-[#1f222b] bg-[#12141a] text-center space-y-3">
              <div className="w-10 h-10 rounded-full bg-[#1c202a] text-neutral-400 mx-auto flex items-center justify-center border border-neutral-700/50">
                <GitCompare className="w-5 h-5 text-neutral-400" />
              </div>
              <div className="space-y-1">
                <h4 className="font-semibold text-white">No Version Comparison Available</h4>
                <p className="text-neutral-400 text-xs max-w-md mx-auto leading-relaxed">
                  {error}
                </p>
              </div>
            </div>
          ) : diff ? (
            <>
              {/* Summary Stats -- from the real WorldDiff.summary() */}
              <div className="grid grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-[#151821] border border-[#1f222b]">
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Modified Entities</span>
                  <span className="text-lg font-mono font-bold text-[#00e5ff] mt-0.5 block">
                    {diff.summary?.entities_modified ?? 0}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#151821] border border-[#1f222b]">
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Added Entities</span>
                  <span className="text-lg font-mono font-bold text-[#2ecc71] mt-0.5 block">
                    {diff.summary?.entities_added ?? 0}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#151821] border border-[#1f222b]">
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Removed Entities</span>
                  <span className="text-lg font-mono font-bold text-neutral-400 mt-0.5 block">
                    {diff.summary?.entities_removed ?? 0}
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#151821] border border-[#1f222b]">
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Geometry Changes</span>
                  <span className="text-lg font-mono font-bold text-[#2ecc71] mt-0.5 block">
                    {diff.geometry_diffs?.length ?? 0}
                  </span>
                </div>
              </div>

              {/* Entity Diffs -- kind is added/removed/modified; changes is the
                  real per-field {field, old, new} list from world_ir.diff */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Entity Changes ({diff.entity_diffs?.length || 0})
                </h4>

                {(diff.entity_diffs || []).length === 0 ? (
                  <p className="text-neutral-500 italic text-xs">No entity changes between these versions.</p>
                ) : (
                  (diff.entity_diffs || []).map((d: EntityDiffDto) => (
                    <div
                      key={d.entity_id}
                      className="p-4 rounded-lg bg-[#151821] border border-[#1f222b] space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-semibold text-white text-sm">{d.entity_id}</span>
                          <span
                            className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                              d.kind === "added"
                                ? "text-[#2ecc71] bg-[#2ecc71]/10"
                                : d.kind === "removed"
                                ? "text-neutral-400 bg-neutral-700/30"
                                : "text-[#00e5ff] bg-[#00e5ff]/10"
                            }`}
                          >
                            {d.kind.toUpperCase()}
                          </span>
                        </div>
                        {onSelectEntity && (
                          <button
                            type="button"
                            onClick={() => {
                              onSelectEntity(d.entity_id);
                              onClose();
                            }}
                            className="px-2.5 py-1 rounded text-xs font-medium text-[#00e5ff] hover:bg-[#00e5ff]/10 transition-colors"
                          >
                            Inspect in 3D →
                          </button>
                        )}
                      </div>

                      {d.changes.length > 0 && (
                        <div className="rounded bg-[#0e1013] border border-[#1f222b]/60 text-[11px] divide-y divide-neutral-800">
                          {d.changes.map((c) => (
                            <div key={c.field} className="grid grid-cols-3 gap-2 p-2.5">
                              <span className="text-neutral-500 font-mono">{c.field}</span>
                              <span className="font-mono text-neutral-400 truncate" title={String(c.old)}>
                                {c.old === null || c.old === undefined ? "—" : String(c.old)}
                              </span>
                              <span className="font-mono text-[#00e5ff] truncate" title={String(c.new)}>
                                {c.new === null || c.new === undefined ? "—" : String(c.new)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 h-12 border-t border-[#1f222b] bg-[#12141a] text-xs">
          <span className="font-mono text-neutral-500">WorldStore Version Lineage Engine</span>
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-md font-medium text-white bg-neutral-800 hover:bg-neutral-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
