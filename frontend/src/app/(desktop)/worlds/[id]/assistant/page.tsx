"use client";

import AssistantPanel from "@/components/studio/AssistantPanel";

/**
 * AI Assistant page for a world.
 *
 * Provides a conversational interface over the WorldIR using the backend
 * assistant tool layer (perception/assistant/). All tools are read-only or
 * propose edits that must be confirmed before any world mutation.
 */
export default async function AssistantPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <AssistantPanel worldId={id} onClose={() => {}} />;
}
