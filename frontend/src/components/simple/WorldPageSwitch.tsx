"use client";

import PlaceViewer from "@/components/simple/PlaceViewer";
import SpatialWorkstation from "@/components/workspace/SpatialWorkstation";
import { useProMode } from "@/lib/prefs";

/** Simple mode shows the place viewer; Pro mode keeps the full engineer workstation, unchanged. */
export default function WorldPageSwitch({ worldId }: { worldId: string }) {
  const pro = useProMode();
  return pro ? <SpatialWorkstation worldId={worldId} /> : <PlaceViewer worldId={worldId} />;
}
