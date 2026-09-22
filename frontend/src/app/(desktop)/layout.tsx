import AppShell from "@/components/shell/AppShell";

export default function DesktopLayout({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
