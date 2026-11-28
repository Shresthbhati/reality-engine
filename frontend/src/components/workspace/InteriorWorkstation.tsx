"use client";

import { useMemo, useState } from "react";
import { Loader2, GitBranch } from "lucide-react";
import {
  useInterior,
  useWorlds,
  useWorldVersions,
  commitWorldCorrection,
  type CommitRequest,
} from "@/lib/api";
import type { InteriorSpace } from "@/lib/interior/model";
import InteriorHierarchy from "./InteriorHierarchy";
import InteriorSpaceInspector from "./InteriorSpaceInspector";
import ReconstructionStatusBanner from "./ReconstructionStatusBanner";
import EvidenceTrace from "./EvidenceTrace";

/**
 * Interior reconstruction workspace.
 *
 * Everything on screen is derived from a real WorldIR version: the
 * hierarchy, the room/corridor measurements, the provenance chain, and the
 * reconstruction state. There is no local entity cache and no placeholder
 * geometry -- with no version, the workspace says so and shows why.
 *
 * Corrections go through `POST /api/worlds/{id}/commit`, which mints a real
 * WorldStore version. The API's stale-parent 409 and duplicate-retry response
 * are surfaced rather than swallowed, so the operator knows whether their
 * edit landed as a new version or was rejected.
 */
export default function InteriorWorkstation({ worldId }: { worldId: string }) {
  const { model, status, versions, isLoading, error, refetch } = useInterior(worldId);
  const { data: worlds } = useWorlds();
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [commitError, setCommitError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const world = useMemo(
    () => (worlds || []).find((w) => w.id === worldId) ?? null,
    [worlds, worldId],
  );

  const selectedSpace: InteriorSpace | null = useMemo(() => {
    if (!model || !selectedEntityId) return null;
    return model.spaces.find((s) => s.id === selectedEntityId) ?? null;
  }, [model, selectedEntityId]);

  const currentVersionId = versions.find((v) => v.isCurrent)?.id ?? null;

  async function handleCommit(commitMessage: string) {
    if (!model || !selectedEntityId) return;
    setPending(true);
    setCommitError(null);
    try {
      const payload: CommitRequest = {
        entity_id: selectedEntityId,
        changes: { name: commitMessage || undefined },
        parent_version_id: currentVersionId,
        commit_message: commitMessage,
      };
      await commitWorldCorrection(worldId, payload);
      refetch();
    } catch (e) {
      setCommitError(
        e instanceof Error ? e.message : "The correction was not accepted.",
      );
    } finally {
      setPending(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 h-full text-neutral-400 text-sm">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading reconstruction...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-sm text-[#ff6b6b] border border-[#e74c3c]/40 rounded">
        Could not load this world&rsquo;s reconstruction: {error.message}
      </div>
    );
  }

  return (
    <div className="h-full flex bg-[#0e1013] text-neutral-200">
      {/* Hierarchy */}
      <aside className="w-[320px] shrink-0 border-r border-[#1f222b] overflow-y-auto p-3">
        <h2 className="text-sm font-semibold text-white mb-0.5">
          {world?.name ?? model?.worldName ?? worldId}
        </h2>
        <p className="text-[11px] font-mono text-neutral-500 mb-3">
          {worldId}
          {currentVersionId && ` \u00b7 ${currentVersionId}`}
        </p>

        {model ? (
          <InteriorHierarchy
            model={model}
            selectedEntityId={selectedEntityId}
            onSelect={setSelectedEntityId}
          />
        ) : (
          <div className="text-[11px] text-neutral-500 border border-dashed border-neutral-800 rounded px-2.5 py-2">
            No WorldIR version is available for this world.
          </div>
        )}
      </aside>

      {/* Center: state + selection detail */}
      <main className="flex-1 min-w-0 overflow-y-auto p-4 space-y-4">
        <ReconstructionStatusBanner status={status} />

        {model && (
          <section className="rounded-lg border border-[#1f222b] bg-[#12141a] p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                Version history
              </h3>
              <span className="text-[10px] font-mono text-neutral-600">
                {versions.length} version(s)
              </span>
            </div>
            {versions.length === 0 ? (
              <div className="text-[11px] text-neutral-500">
                No versions yet. A reconstruction commit creates the first one.
              </div>
            ) : (
              <ol className="space-y-1">
                {versions.map((v, i) => (
                  <li key={v.id} className="flex items-center gap-2 text-[11px]">
                    <GitBranch className="w-3 h-3 text-neutral-600 shrink-0" />
                    <span className="font-mono text-neutral-300">{v.id}</span>
                    <span className="text-neutral-500 truncate">
                      {v.changeSummary}
                    </span>
                    {v.isCurrent && (
                      <span className="ml-auto text-[10px] text-[#2ecc71] shrink-0">
                        current
                      </span>
                    )}
                    {i < versions.length - 1 && (
                      <span className="text-neutral-700">|</span>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </section>
        )}

        {selectedSpace && (
          <section className="rounded-lg border border-[#1f222b] bg-[#12141a] p-4">
            <InteriorSpaceInspector
              space={selectedSpace}
              onSelectEntity={setSelectedEntityId}
            />
          </section>
        )}
      </main>

      {/* Right: provenance + correction */}
      <aside className="w-[300px] shrink-0 border-l border-[#1f222b] overflow-y-auto p-3 space-y-4">
        {selectedEntityId ? (
          <>
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-2">
                Evidence trace
              </h3>
              <EvidenceTrace
                worldId={worldId}
                entityId={selectedEntityId}
              />
            </div>

            <div className="pt-3 border-t border-neutral-800">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-2">
                Correct this entity
              </h3>
              <p className="text-[10px] text-neutral-600 leading-relaxed mb-2">
                A correction creates a new immutable WorldStore version. The
                API rejects a stale parent version, and reports a duplicate
                retry without minting a second version.
              </p>
              <button
                type="button"
                disabled={pending}
                onClick={() =>
                  handleCommit(`Reviewed ${selectedEntityId}`)
                }
                className="w-full px-3 py-1.5 rounded text-xs font-semibold bg-[#00e5ff] text-black hover:bg-[#33ebff] disabled:opacity-50 transition-colors"
              >
                {pending ? "Committing..." : "Commit review note"}
              </button>
              {commitError && (
                <div className="mt-2 text-[11px] text-[#ff6b6b] leading-relaxed">
                  {commitError}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="text-[11px] text-neutral-500 border border-dashed border-neutral-800 rounded px-2.5 py-2">
            Select a room, wall, floor, or ceiling to inspect its evidence and
            record a correction.
          </div>
        )}
      </aside>
    </div>
  );
}