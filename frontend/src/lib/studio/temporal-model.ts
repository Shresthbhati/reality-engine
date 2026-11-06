/**
 * Studio Multi-Epoch Temporal Alignment & Diffing Client Model (Epic E07-1).
 *
 * Provides models and helper algorithms for multi-epoch temporal comparison:
 * - Temporal epochs and timestamps
 * - Entity change kinds: Added, Removed, Moved (with displacement), Renovated, Damaged, Unchanged
 * - Color-coded semantic styles for 3D viewport entity overlays
 * - Interactive split-view comparison slider state
 */

export type TemporalChangeKind =
  | "added"
  | "removed"
  | "moved"
  | "renovated"
  | "damaged"
  | "unchanged";

export interface TemporalEntityChangeItem {
  entity_id: string;
  name: string;
  change_kind: TemporalChangeKind;
  displacement_m?: number;
  details?: string;
  confidence: number;
}

export interface TemporalEpochSummary {
  epoch_id: string;
  capture_date_iso: string;
  label: string;
  entity_count: number;
}

export interface TemporalDiffSummary {
  from_epoch_id: string;
  to_epoch_id: string;
  is_aligned: boolean;
  static_anchor_count: number;
  residual_error_m: number;
  changes: TemporalEntityChangeItem[];
}

export interface TemporalColorToken {
  border: string;
  bg: string;
  text: string;
  hex: number;
}

/**
 * Returns standardized color tokens for each change kind in the UI and 3D Viewport.
 */
export function getTemporalChangeColor(kind: TemporalChangeKind): TemporalColorToken {
  switch (kind) {
    case "added":
      return { border: "border-success", bg: "bg-success-subtle", text: "text-success", hex: 0x10b981 };
    case "removed":
      return { border: "border-error", bg: "bg-error-subtle", text: "text-error", hex: 0xf43f5e };
    case "moved":
      return { border: "border-warning", bg: "bg-warning-subtle", text: "text-warning", hex: 0xf59e0b };
    case "renovated":
      return { border: "border-info", bg: "bg-info-subtle", text: "text-info", hex: 0x0ea5e9 };
    case "damaged":
      return { border: "border-purple", bg: "bg-purple-subtle", text: "text-purple", hex: 0xa855f7 };
    case "unchanged":
    default:
      return { border: "border-line-strong", bg: "bg-elevated/40", text: "text-fg-muted", hex: 0x71717a };
  }
}

/**
 * Summarizes counts by change kind.
 */
export function countChangesByKind(changes: TemporalEntityChangeItem[]): Record<TemporalChangeKind, number> {
  const counts: Record<TemporalChangeKind, number> = {
    added: 0,
    removed: 0,
    moved: 0,
    renovated: 0,
    damaged: 0,
    unchanged: 0,
  };
  for (const c of changes) {
    if (c.change_kind in counts) {
      counts[c.change_kind]++;
    }
  }
  return counts;
}

/**
 * Filters changes by kind or search query.
 */
export function filterTemporalChanges(
  changes: TemporalEntityChangeItem[],
  kindFilter: TemporalChangeKind | "all",
  searchQuery: string = "",
): TemporalEntityChangeItem[] {
  const q = searchQuery.toLowerCase().trim();
  return changes.filter((c) => {
    if (kindFilter !== "all" && c.change_kind !== kindFilter) return false;
    if (q && !c.name.toLowerCase().includes(q) && !c.entity_id.toLowerCase().includes(q)) return false;
    return true;
  });
}
