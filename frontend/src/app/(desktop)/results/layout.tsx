import { notFound } from "next/navigation";
import { SHOW_UNFINISHED } from "@/lib/unfinished";

export default function Layout({ children }: { children: React.ReactNode }) {
  if (!SHOW_UNFINISHED) notFound();
  return children;
}
