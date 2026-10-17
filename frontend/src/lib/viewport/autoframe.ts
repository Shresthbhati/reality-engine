/**
 * Whether the viewport should auto-frame after a scene build.
 *
 * A procedural world has no point cloud, so the default camera (4,3,6 -> origin) leaves a 100 m city off-screen.
 * Frame once per world id; later rebuilds of the same world must not yank the user's camera.
 */
export function shouldAutoFrame(
  hasCloud: boolean,
  meshCount: number,
  framedWorldId: string | null,
  worldId: string,
): boolean {
  return !hasCloud && meshCount > 0 && framedWorldId !== worldId;
}
