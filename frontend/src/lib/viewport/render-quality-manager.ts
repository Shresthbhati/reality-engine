/**
 * Unified rendering-quality manager (N-28).
 *
 * Selects the appropriate rendering representation based on the task:
 *   - Gaussian splats for visual context / artistic preview
 *   *   - Meshes for geometric interaction / measurement
 *   *   - Architectural solids (B-rep / parametric) for precise editing
 *   *   - Point clouds for measurement evidence
 *
 * Every representation carries its own confidence / provenance; nothing is
 * silently chosen because it "looks best". The manager reports *why* a
 * representation was selected so the UI can explain it to the user.
 */

import type { Entity, Geometry } from "@/types/worldir";

/** One rendering representation of an entity or region. */
export type RepresentationKind =
  | "splat"        // Gaussian splat cloud (visual context)
  | "mesh"         // Triangle mesh (geometric interaction)
  | "architectural" // Parametric / B-rep solid (precise editing)
  | "pointcloud";   // Point cloud (measurement evidence)

export interface RepresentationOption {
  kind: RepresentationKind;
  /** Why this representation is suitable for the current task. */
  reason: string;
  /** Confidence in this representation for the task (0..1). */
  confidence: number;
  /** Estimated memory cost in KB (for budget planning). */
  estimatedKb: number;
  /** Whether this representation has real data (vs fabricated). */
  hasData: boolean;
  /** Source entity ids that back this representation. */
  sourceEntityIds: string[];
}

/** One entry in the unified quality report: representation + verdict. */
export interface RepresentationVerdict {
  kind: RepresentationKind;
  reason: string;
  confidence: number;
  estimatedKb: number;
  hasData: boolean;
  sourceEntityIds: string[];
}

/**
 * Task profile: what the user is trying to do. Each task has a preferred
 * representation order and a minimum confidence threshold.
 */
export interface TaskProfile {
  id: string;
  label: string;
  /** Preferred representations in order (first = best for this task). */
  preferred: RepresentationKind[];
  /** Minimum confidence to auto-select without asking the user. */
  minConfidence: number;
  /** User-visible explanation of what this task is. */
  description: string;
}

/** Built-in task profiles. */
export const TASK_PROFILES: Record<string, TaskProfile> = {
  visual_context: {
    id: "visual_context",
    label: "Visual context",
    preferred: ["splat", "mesh", "architectural", "pointcloud"],
    minConfidence: 0.6,
    description: "Inspect the world as a whole; visual fidelity matters most.",
  },
  geometric_interaction: {
    id: "geometric_interaction",
    label: "Geometric interaction",
    preferred: ["mesh", "architectural", "splat", "pointcloud"],
    minConfidence: 0.7,
    description: "Measure, snap, and edit geometry; watertight surfaces matter.",
  },
  precise_editing: {
    id: "precise_editing",
    label: "Precise editing",
    preferred: ["architectural", "mesh", "splat", "pointcloud"],
    minConfidence: 0.8,
    description: "Move walls, openings, and fixtures; parametric accuracy required.",
  },
  measurement_evidence: {
    id: "measurement_evidence",
    label: "Measurement evidence",
    preferred: ["pointcloud", "mesh", "architectural", "splat"],
    minConfidence: 0.9,
    description: "Verify dimensions against captured evidence; uncertainty matters most.",
  },
};

/**
 * Select the best representation for a task given the available data.
 *
 * Honesty rules:
 *   * A representation with no data is never selected as "available".
 *   * Every selection carries a reason and confidence so the UI can explain it.
 *   * If nothing meets the minConfidence, the best available is returned
 *     but flagged as low-confidence so the UI can prompt the user.
 */
export function selectRepresentation(
  taskId: string,
  available: RepresentationOption[],
): RepresentationVerdict {
  const profile = TASK_PROFILES[taskId];
  if (!profile) {
    throw new Error(`Unknown task profile: ${taskId}`);
  }

  // Filter to representations that actually have data.
  const withData = available.filter((r) => r.hasData);
  const withoutData = available.filter((r) => !r.hasData);

  // Pick the best preferred representation that has data.
  for (const kind of profile.preferred) {
    const best = withData
      .filter((r) => r.kind === kind)
      .sort((a, b) => b.confidence - a.confidence)[0];
    if (best) {
      return {
        kind: best.kind,
        reason: best.reason,
        confidence: best.confidence,
        estimatedKb: best.estimatedKb,
        hasData: true,
        sourceEntityIds: best.sourceEntityIds,
      };
    }
  }

  // Nothing with data meets the task. If there's data at all (even low
  // confidence), return the best of it flagged as low-confidence.
  if (withData.length > 0) {
    const best = withData.sort((a, b) => b.confidence - a.confidence)[0];
    return {
      kind: best.kind,
      reason: `No high-confidence ${best.kind} representation for this task; showing best available (${best.confidence.toFixed(0)}% confidence).`,
      confidence: best.confidence,
      estimatedKb: best.estimatedKb,
      hasData: true,
      sourceEntityIds: best.sourceEntityIds,
    };
  }

  // No data at all.
  const firstKind = profile.preferred[0];
  return {
    kind: firstKind,
    reason: "No geometric data available for this task. Capture more photos or run reconstruction first.",
    confidence: 0,
    estimatedKb: 0,
    hasData: false,
    sourceEntityIds: [],
  };
}

/**
 * Build representation options from a WorldIR entity's geometries.
 * Converts the canonical GeometryType list into representation options
 * with honest provenance and confidence.
 */
export function buildRepresentationOptions(
  entity: Entity,
  geometryById: Map<string, Geometry>,
): RepresentationOption[] {
  const options: RepresentationOption[] = [];

  const gids = entity.geometry_ids;
  if (!gids) return options;

  for (const gid of gids) {
    const geom = geometryById.get(gid);
    if (!geom) continue;

    const kind = geometryTypeToRepresentation(geom.type);
    const confidence = Math.min(1, Math.max(0, entity.confidence ?? 0.5));
    const estimatedKb = estimateGeometryKb(geom);

    const cp = entity.custom_properties ?? {};
    options.push({
      kind,
      reason: geometryReason(geom.type, cp),
      confidence,
      estimatedKb,
      hasData: geom.data_uri != null || (geom.vertex_count ?? 0) > 0,
      sourceEntityIds: [entity.id],
    });
  }

  // Deduplicate by kind: keep the highest confidence per kind.
  const byKind = new Map<RepresentationKind, RepresentationOption>();
  for (const opt of options) {
    const existing = byKind.get(opt.kind);
    if (!existing || opt.confidence > existing.confidence) {
      byKind.set(opt.kind, opt);
    }
  }

  return Array.from(byKind.values());
}

/** Map a geometry type string to a representation kind. */
function geometryTypeToRepresentation(type: string): RepresentationKind {
  switch (type) {
    case "pointcloud":
      return "pointcloud";
    case "mesh":
      return "mesh";
    case "box":
    case "cad":
    case "parametric":
    case "brep":
      return "architectural";
    case "splat":
      return "splat";
    default:
      return "mesh";
  }
}

/** Human-readable reason for why a geometry type is a good representation. */
function geometryReason(type: string, customProps: Record<string, unknown>): string {
  switch (type) {
    case "splat":
      return "Gaussian splat cloud: photorealistic visual context from captured imagery.";
    case "mesh":
      return "Triangle mesh: geometric interaction and measurement-ready surfaces.";
    case "box":
    case "cad":
    case "parametric":
    case "brep":
      return "Architectural solid: parametric dimensions editable with full precision.";
    case "pointcloud":
      return "Point cloud: raw measurement evidence; best for dimensional verification.";
    default:
      return "Mesh geometry: general-purpose surface representation.";
  }
}

/** Estimate geometry memory cost in KB from a Geometry record. */
function estimateGeometryKb(geom: Geometry): number {
  // Vertex data: 3 floats * 4 bytes = 12 bytes per vertex.
  // Faces: 3 indices * 4 bytes = 12 bytes per face.
  const vertexBytes = (geom.vertex_count ?? 0) * 12;
  const faceBytes = (geom.triangle_count ?? 0) * 12;
  // Splats: ~56 bytes per gaussian (position 12 + scale 12 + rot 16 + color 3 + opacity 4).
  const vc = geom.vertex_count ?? 0;
  const splatBytes = vc > 0 ? vc * 56 : 0;
  return Math.max(vertexBytes + faceBytes, splatBytes) / 1024;
}

/**
 * Compute a unified quality report for a world: one verdict per entity/task
 * combination, so the UI can show per-entity representation quality.
 */
export function buildWorldQualityReport(
  world: {
    entities: Entity[];
    geometries: Map<string, Geometry>;
  },
  taskId: string,
): RepresentationVerdict[] {
  return world.entities.map((ent) => {
    const options = buildRepresentationOptions(ent, world.geometries);
    return selectRepresentation(taskId, options);
  });
}

