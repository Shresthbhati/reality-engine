import { notFound } from "next/navigation";
import {
  getWorld,
  sessionsForWorld,
  placesForWorld,
  versionsForWorld,
  resultsForWorld,
  analysisForWorld,
  EVIDENCE,
} from "@/lib/data";
import WorldWorkspaceClient from "./WorldWorkspaceClient";

export default async function WorldWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const world = getWorld(id);
  if (!world) notFound();

  return (
    <WorldWorkspaceClient
      world={world}
      sessions={sessionsForWorld(world.id)}
      evidence={EVIDENCE.filter((e) => e.worldId === world.id)}
      places={placesForWorld(world.id)}
      versions={versionsForWorld(world.id)}
      results={resultsForWorld(world.id)}
      analysis={analysisForWorld(world.id)}
    />
  );
}
