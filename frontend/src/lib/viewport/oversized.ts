/**
 * Whether an entity is "oversized depth noise" relative to the reconstructed point cloud.
 *
 * Only meaningful when a point cloud exists: a procedural world has no cloud to be noisy against, so its large
 * entities (ground, sky, buildings) must never be classed as oversized and hidden.
 */
export function isOversizedEntity(extent: number, hasCloud: boolean, robustExtent: number): boolean {
  return hasCloud && robustExtent > 0 && extent > 4 * robustExtent;
}
