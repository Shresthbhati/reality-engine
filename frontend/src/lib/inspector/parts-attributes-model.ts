/**
 * Parts Tree & Attributes Parser for Studio Inspector (BUILD_SPEC B18).
 *
 * Enforces:
 * 1. Parts tree derived strictly from `part_of` relationships.
 * 2. Attributes parser where unknown shows "Unknown — not visible in any photo".
 * 3. No value is ever rendered without its basis.
 */

import type { Entity } from "@/types/worldir";

export interface PartTreeNode {
  id: string;
  type: string;
  label: string;
  children: PartTreeNode[];
}

export interface AttributeRow {
  name: string;
  value: string | null;
  displayValue: string;
  basis: "visual" | "user_stated" | "unknown";
  confidence: number | null;
  obsIds: string[];
}

/**
 * Builds a hierarchical tree of parts belonging to the root entity via `part_of` relationships.
 */
export function parsePartsTree(
  rootId: string,
  entities: Record<string, Entity>,
): PartTreeNode[] {
  // Map parent -> list of child entities
  const childrenMap = new Map<string, string[]>();

  for (const [id, entity] of Object.entries(entities)) {
    if (!entity || !entity.relationships) continue;
    for (const rel of entity.relationships) {
      if ((rel.kind === "part_of" || rel.type === "part_of") && rel.target_id) {
        const parentId = rel.target_id;
        if (!childrenMap.has(parentId)) {
          childrenMap.set(parentId, []);
        }
        childrenMap.get(parentId)!.push(id);
      }
    }
  }

  function buildSubTree(parentId: string, visited: Set<string>): PartTreeNode[] {
    const childIds = childrenMap.get(parentId) || [];
    const nodes: PartTreeNode[] = [];

    for (const cid of childIds) {
      if (visited.has(cid)) continue; // prevent cycles
      visited.add(cid);

      const childEntity = entities[cid];
      if (!childEntity) continue;

      const label =
        childEntity.semantic_labels?.[0] ||
        childEntity.custom_properties?.label ||
        childEntity.type;

      nodes.push({
        id: cid,
        type: childEntity.type,
        label: String(label),
        children: buildSubTree(cid, new Set(visited)),
      });
    }

    return nodes;
  }

  return buildSubTree(rootId, new Set([rootId]));
}

/**
 * Parses classified or user-stated attributes of an entity.
 * Unknown or missing values explicitly display "Unknown — not visible in any photo".
 * Every row guarantees an authoritative basis.
 */
export function parseAttributes(entity: Entity | null | undefined): AttributeRow[] {
  if (!entity) return [];

  // JSON from the API: any field may be missing or of another type, so everything is `unknown` until checked below
  type RawAttr = { value?: unknown; basis?: unknown; p?: unknown; obs?: unknown };
  type RawAttrs = Record<string, RawAttr | string | number | boolean | null>;
  const withExtra = entity as unknown as { extra_fields?: { attributes?: RawAttrs } };
  const rawAttrs: RawAttrs =
    withExtra.extra_fields?.attributes ||
    (entity.custom_properties?.attributes as RawAttrs | undefined) ||
    {};

  const rows: AttributeRow[] = [];

  for (const [attrName, attrData] of Object.entries(rawAttrs)) {
    if (typeof attrData !== "object" || attrData === null) {
      rows.push({
        name: attrName,
        value: null,
        displayValue: "Unknown — not visible in any photo",
        basis: "unknown",
        confidence: null,
        obsIds: [],
      });
      continue;
    }

    const val = attrData.value;
    const isUnknown =
      val === null ||
      val === undefined ||
      val === "unknown" ||
      val === "" ||
      val === "None";

    const basis: "visual" | "user_stated" | "unknown" = isUnknown
      ? "unknown"
      : attrData.basis === "user_stated"
      ? "user_stated"
      : "visual";

    const displayValue = isUnknown
      ? "Unknown — not visible in any photo"
      : String(val);

    const conf =
      typeof attrData.p === "number" && !isUnknown ? attrData.p : null;

    const obsIds = Array.isArray(attrData.obs)
      ? attrData.obs.map(String)
      : [];

    rows.push({
      name: attrName,
      value: isUnknown ? null : String(val),
      displayValue,
      basis,
      confidence: conf,
      obsIds,
    });
  }

  return rows;
}
