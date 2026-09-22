import SpatialWorkstation from "@/components/workspace/SpatialWorkstation";

/**
 * Reality Engine Desktop World Workspace:
 *
 * Immersive persistent spatial workstation centered on the 3D reconstructed
 * representation of the World, with adaptive inspector, world navigation,
 * and contextual timeline/status bar.
 */
export default async function WorldWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <SpatialWorkstation worldId={id} />;
}
