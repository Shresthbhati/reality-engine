import { redirect } from "next/navigation";

/**
 * There is one way to start a session: the guided flow at /reconstruct (photos in, world out, with real stage names).
 * The old manual form (name, source, location, world) asked for configuration before any photo existed; it is gone.
 * This route stays only so existing links and bookmarks keep working. `?world=<id>` is preserved.
 */
export default async function NewSessionRedirect({
  searchParams,
}: {
  searchParams: Promise<{ world?: string | string[] }>;
}) {
  const { world } = await searchParams;
  const id = Array.isArray(world) ? world[0] : world;
  redirect(id ? `/reconstruct?world=${encodeURIComponent(id)}` : "/reconstruct");
}
