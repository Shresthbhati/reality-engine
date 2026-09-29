"use client";

import { useRouter } from "next/navigation";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";

/**
 * The primary product action: give the engine photographs, get a world.
 * No World, Session, backend or version choices come first; the result opens
 * straight in the Studio.
 */
export default function CreateReconstructionPage() {
  const router = useRouter();
  return (
    <div className="flex h-full w-full flex-col items-center justify-center p-6">
      <h1 className="mb-1 text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
        Create a reconstruction
      </h1>
      <p className="mb-6 max-w-lg text-center text-xs" style={{ color: "var(--text-tertiary)" }}>
        Give Reality Engine photographs of a real place. It builds the best spatial model the evidence
        supports, says plainly what it is unsure about, and improves the same model as you add more views.
      </p>
      <div className="w-full max-w-xl">
        <PhotoDropZone onCreated={(res) => router.push(`/worlds/${res.world_id}`)} />
      </div>
    </div>
  );
}
