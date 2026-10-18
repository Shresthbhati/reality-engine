import { Suspense } from "react";
import CaptureScreen from "@/components/mobile/CaptureScreen";

// The world id rides in `?world=` so a reload keeps adding to the SAME world.
export default function MobileCameraPage() {
  return (
    <Suspense>
      <CaptureScreen />
    </Suspense>
  );
}
