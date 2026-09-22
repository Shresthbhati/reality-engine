import MobileShell from "@/components/mobile/MobileShell";

export default function MobileLayout({ children }: { children: React.ReactNode }) {
  return <MobileShell>{children}</MobileShell>;
}
