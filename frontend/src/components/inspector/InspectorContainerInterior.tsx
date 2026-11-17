"use client";

import { useState } from "react";
import { EyeOff, User, Plus, Edit2, Check, X, Box } from "lucide-react";
import type { Entity } from "@/types/worldir";
import { Disclosure, EntityChip, SectionTitle } from "./InspectorParts";

interface InspectorContainerInteriorProps {
  entity: Entity;
  entities?: Record<string, Entity>;
  onCommitCorrection?: (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => Promise<{ version_id: string; parent_version_id?: string; world_id?: string } | void>;
  onSelectEntity?: (id: string) => void;
}

const CONTAINER_TYPES = new Set([
  "cabinet",
  "wardrobe",
  "almirah",
  "cupboard",
  "drawer",
  "chest_of_drawers",
  "refrigerator",
  "closet",
  "storage",
]);

export function isContainerEntity(entity: Entity): boolean {
  const t = String(entity.type || "").toLowerCase();
  if (CONTAINER_TYPES.has(t)) return true;
  const labels = entity.semantic_labels || [];
  if (labels.some((l) => CONTAINER_TYPES.has(l.toLowerCase()) || l.toLowerCase() === "container")) {
    return true;
  }
  const cp = entity.custom_properties || {};
  if (cp.interior !== undefined || cp.user_stated_contents !== undefined || cp.is_container) {
    return true;
  }
  return false;
}

export function InspectorContainerInterior({
  entity,
  entities = {},
  onCommitCorrection,
  onSelectEntity,
}: InspectorContainerInteriorProps) {
  const cp = (entity.custom_properties || {}) as Record<string, unknown>;
  const interiorState = String(cp.interior || "not_observed");
  const userStated = (cp.user_stated_contents as string | undefined) || "";

  const [editing, setEditing] = useState(false);
  const [statedText, setStatedText] = useState(userStated);
  const [saving, setSaving] = useState(false);

  // Find any observed entities contained inside this container
  const containedIds: string[] = [];
  for (const rel of entity.relationships || []) {
    const kind = String(rel.kind || rel.type || "").toLowerCase();
    if (kind === "contains" && rel.target_id) {
      containedIds.push(rel.target_id);
    }
  }
  // Also check entities where this is parent or container
  for (const other of Object.values(entities)) {
    if (other.id === entity.id) continue;
    for (const rel of other.relationships || []) {
      const kind = String(rel.kind || rel.type || "").toLowerCase();
      if (kind === "inside" && rel.target_id === entity.id) {
        if (!containedIds.includes(other.id)) containedIds.push(other.id);
      }
    }
  }

  const handleSaveStated = async () => {
    if (!onCommitCorrection) return;
    setSaving(true);
    try {
      const trimmed = statedText.trim();
      const updatedCp = {
        ...cp,
        interior: trimmed ? "user_stated" : "not_observed",
        user_stated_contents: trimmed || null,
      };
      await onCommitCorrection(
        entity.id,
        { custom_properties: updatedCp },
        `User stated contents of ${entity.name || entity.id}: ${trimmed || "none"}`
      );
      setEditing(false);
    } catch {
      // error handled by parent commit banner
    } finally {
      setSaving(false);
    }
  };

  const isNotObserved = interiorState === "not_observed" && !userStated;

  return (
    <Disclosure title="Interior Contents" testId="container-interior-disclosure">
      <div className="space-y-3 text-[12px]">
        {/* Status indicator */}
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium text-[var(--text-secondary)]">Observation state</span>
          {isNotObserved ? (
            <span
              className="inline-flex items-center gap-1 rounded bg-[var(--border-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-tertiary)]"
              title="Closed container — internal contents were not visible in any photograph"
            >
              <EyeOff className="h-3 w-3" aria-hidden />
              Not observed
            </span>
          ) : userStated ? (
            <span
              className="inline-flex items-center gap-1 rounded bg-[var(--info-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--info)]"
              title="Contents stated by user, not verified by camera sensor"
            >
              <User className="h-3 w-3" aria-hidden />
              User stated
            </span>
          ) : (
            <span
              className="inline-flex items-center gap-1 rounded bg-[var(--accent-subtle)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--accent)]"
              title="Contents observed in open container photos"
            >
              <Box className="h-3 w-3" aria-hidden />
              Observed
            </span>
          )}
        </div>

        {/* Not observed explainer */}
        {isNotObserved && !editing && (
          <p className="rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-2 text-[11px] leading-relaxed text-[var(--text-secondary)]">
            Closed container. Internal contents were not visible in any photograph and are not assumed empty. You can state known contents below.
          </p>
        )}

        {/* User stated contents */}
        {userStated && !editing && (
          <div className="rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-[var(--text-tertiary)]">
              <span>Stated items:</span>
              <button
                type="button"
                onClick={() => setEditing(true)}
                className="text-[var(--accent)] hover:underline inline-flex items-center gap-1 cursor-pointer"
              >
                <Edit2 className="h-3 w-3" /> Edit
              </button>
            </div>
            <p className="font-medium text-[var(--text-primary)]">{userStated}</p>
          </div>
        )}

        {/* Editor form */}
        {editing && (
          <div className="space-y-2 rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-2.5">
            <label htmlFor="user-stated-input" className="block text-[11px] text-[var(--text-tertiary)]">
              State known contents:
            </label>
            <input
              id="user-stated-input"
              type="text"
              value={statedText}
              onChange={(e) => setStatedText(e.target.value)}
              placeholder="e.g. Snack boxes, utensils, towels..."
              className="w-full rounded border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-1 text-[12px] text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none"
            />
            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => {
                  setStatedText(userStated);
                  setEditing(false);
                }}
                disabled={saving}
                className="rounded px-2 py-0.5 text-[11px] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveStated}
                disabled={saving}
                className="rounded bg-[var(--primary)] px-2.5 py-0.5 text-[11px] font-medium text-[var(--on-primary)] hover:bg-[var(--primary-hover)] cursor-pointer disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        )}

        {!userStated && !editing && onCommitCorrection && (
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="inline-flex items-center gap-1.5 text-[11px] text-[var(--accent)] hover:underline cursor-pointer"
          >
            <Plus className="h-3 w-3" /> State contents
          </button>
        )}

        {/* Observed contained entities */}
        {containedIds.length > 0 && (
          <div className="space-y-1.5 pt-2 border-t border-[var(--border-subtle)]">
            <span className="text-[11px] text-[var(--text-tertiary)]">Observed items inside:</span>
            <div className="flex flex-wrap gap-1.5">
              {containedIds.map((cid) => {
                const target = entities[cid];
                const label = target ? target.name || target.id : cid;
                return (
                  <EntityChip
                    key={cid}
                    label={label}
                    hint={`Show ${label}`}
                    onClick={() => onSelectEntity?.(cid)}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>
    </Disclosure>
  );
}

export default InspectorContainerInterior;
