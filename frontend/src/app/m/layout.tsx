import type { Viewport } from "next";
import MobileShell from "@/components/mobile/MobileShell";

// viewport-fit=cover lets the shell paint under the notch / home indicator and
// pad with env(safe-area-inset-*) instead of leaving unreadable letterbox bars.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#050b16",
};

export default function MobileLayout({ children }: { children: React.ReactNode }) {
  return <MobileShell>{children}</MobileShell>;
}
