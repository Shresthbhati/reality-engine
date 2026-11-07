import SpatialWorkstation from "@/components/workspace/SpatialWorkstation";

/**
 * A world. One Studio for everyone (decision 2026-11-07): the workstation, with world navigation, adaptive
 * inspector and contextual timeline. The former Simple "place viewer" and the Pro-mode switch were removed.
 */
export default async function WorldWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <SpatialWorkstation worldId={id} />;
}
