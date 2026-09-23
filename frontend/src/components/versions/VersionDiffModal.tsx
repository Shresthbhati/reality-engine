"use client";

import { useEffect, useState } from "react";
import { GitCompare, X, Check, ArrowRight, Shield, Layers, Camera, AlertCircle } from "lucide-react";
import { getWorldDiff } from "@/lib/api/worlds";

interface VersionDiffModalProps {
  worldId: string;
  isOpen: boolean;
  onClose: () => void;
  baseVersion?: string;
  headVersion?: string;
  onSelectEntity?: (id: string) => void;
}

export default function VersionDiffModal({
  worldId,
  isOpen,
  onClose,
  baseVersion = "",
  headVersion = "latest",
  onSelectEntity,
}: VersionDiffModalProps) {
  const [diff, setDiff] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    if (!worldId || !baseVersion) {
      setError("No base version specified. Create and commit world revisions to compare lineage differentials.");
      setLoading(false);
      setDiff(null);
      return;
    }
    setLoading(true);
    setError(null);
    getWorldDiff(worldId, baseVersion, headVersion)
      .then((data) => {
        setDiff(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err?.message || "No version differential available between these versions.");
        setLoading(false);
      });
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
                Comparing <span className="text-[#00e5ff]">{baseVersion}</span> → <span className="text-[#2ecc71]">{headVersion}</span>
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
              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-[#151821] border border-[#1f222b]">
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Modified Entities</span>
                  <span className="text-lg font-mono font-bold text-[#00e5ff] mt-0.5 block">
                    {diff.summary?.entities_modified ?? diff.entity_diffs?.length ?? 0}
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
                  <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Confidence Delta</span>
                  <span className="text-lg font-mono font-bold text-[#2ecc71] mt-0.5 block">
                    {diff.summary?.confidence_delta ?? "+0.00"}
                  </span>
                </div>
              </div>

              {/* Entity Diffs */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                  Detailed Entity Variations ({diff.entity_diffs?.length || 0})
                </h4>

                {(diff.entity_diffs || []).map((d: any) => (
                  <div
                    key={d.entity_id}
                    className="p-4 rounded-lg bg-[#151821] border border-[#1f222b] space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold text-white text-sm">{d.entity_id}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded text-[#00e5ff] bg-[#00e5ff]/10">
                          {d.change_type}
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

                    {/* Change details grid */}
                    <div className="grid grid-cols-2 gap-3 p-2.5 rounded bg-[#0e1013] border border-[#1f222b]/60 text-[11px]">
                      <div>
                        <span className="text-neutral-500 block mb-0.5">Semantic Type Before</span>
                        <div className="font-mono text-neutral-300">{d.type_before || "—"}</div>
                      </div>
                      <div>
                        <span className="text-neutral-500 block mb-0.5">Semantic Type After</span>
                        <div className="font-mono text-[#00e5ff] font-semibold">{d.type_after || "—"}</div>
                      </div>

                      <div className="pt-2 border-t border-neutral-800">
                        <span className="text-neutral-500 block mb-0.5">Confidence Before</span>
                        <div className="font-mono-num text-neutral-400">
                          {d.confidence_before != null ? `${(d.confidence_before * 100).toFixed(1)}%` : "—"}
                        </div>
                      </div>
                      <div className="pt-2 border-t border-neutral-800">
                        <span className="text-neutral-500 block mb-0.5">Confidence After</span>
                        <div className="font-mono-num text-[#2ecc71] font-semibold">
                          {d.confidence_after != null ? `${(d.confidence_after * 100).toFixed(1)}%` : "—"}
                        </div>
                      </div>
                    </div>

                    {d.note && (
                      <p className="text-[11px] text-neutral-400 italic">
                        &quot;{d.note}&quot;
                      </p>
                    )}

                    {/* Linked Evidence Attribution */}
                    {d.source_evidence && d.source_evidence.length > 0 && (
                      <div className="flex items-center gap-2 pt-2 border-t border-neutral-800/80 text-[11px]">
                        <Camera className="w-3.5 h-3.5 text-neutral-500 shrink-0" />
                        <span className="text-neutral-500">Source Evidence:</span>
                        <div className="flex items-center gap-1.5 font-mono">
                          {d.source_evidence.map((ev: string) => (
                            <span key={ev} className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 text-[10px]">
                              {ev}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
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
