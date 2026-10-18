"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import EvidenceQualityList from "@/components/mobile/EvidenceQualityList";

function Inner() {
  const sessionId = useSearchParams().get("session");
  return (
    <div className="flex flex-col gap-5 p-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Evidence
      </h1>
      <EvidenceQualityList sessionId={sessionId} />
    </div>
  );
}

export default function MobileEvidencePage() {
  return (
    <Suspense>
      <Inner />
    </Suspense>
  );
}
