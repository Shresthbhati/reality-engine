"use client";

import { useRouter } from "next/navigation";
import DecisionActions from "./DecisionActions";
import type { QualityDecision } from "@/lib/api/evidence-quality-model";

/** Inspector wrapper: saves via DecisionActions, then refreshes the server-rendered page. */
export default function InspectorDecision(props: {
  evidenceId: string;
  current: QualityDecision | null;
  initialReason: string | null;
}) {
  const router = useRouter();
  return <DecisionActions {...props} onSaved={() => router.refresh()} />;
}
