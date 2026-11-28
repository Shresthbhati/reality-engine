import InteriorWorkstation from "@/components/workspace/InteriorWorkstation";

/**
 * Interior reconstruction workspace for one world.
 *
 * Renders only what the backend actually reconstructed: the WorldIR version
 * is the single source for the hierarchy, measurements, provenance, and
 * version lineage. With no version the page shows an honest "not
 * reconstructed" state rather than placeholder geometry.
 */
export default async function InteriorPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <InteriorWorkstation worldId={id} />;
}