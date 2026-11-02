import WorldPageSwitch from "@/components/simple/WorldPageSwitch";

/**
 * A place. Simple mode is the place viewer (the model is the whole screen); Pro mode is the engineer workstation
 * (world navigation, adaptive inspector, contextual timeline). The mode is a per-viewer preference (Settings).
 */
export default async function WorldWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <WorldPageSwitch worldId={id} />;
}
