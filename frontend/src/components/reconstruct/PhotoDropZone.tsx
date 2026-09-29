"use client";

import React, { useCallback, useRef, useState } from "react";
import { Camera, FolderUp, ImagePlus, Loader2, UploadCloud } from "lucide-react";
import {
  createReconstruction,
  isApiError,
  type CreateReconstructionResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const IMAGE_EXT = /\.(jpe?g|png|webp|heic|heif|tiff?|bmp)$/i;

function isImage(file: File): boolean {
  return file.type.startsWith("image/") || IMAGE_EXT.test(file.name);
}

/** Read dropped files, descending into dropped folders where the browser exposes them. */
async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const items = Array.from(dt.items ?? []);
  const entries = items
    .map((i) => (typeof i.webkitGetAsEntry === "function" ? i.webkitGetAsEntry() : null))
    .filter((e): e is FileSystemEntry => e !== null);
  if (entries.length === 0) return Array.from(dt.files);

  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) =>
        (entry as FileSystemFileEntry).file(res, rej),
      );
      out.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns batches; keep reading until it returns an empty one
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (batch.length === 0) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

export interface PhotoDropZoneProps {
  /** Given: the photos join this world and rebuild it. Omitted: a new world is created. */
  worldId?: string | null;
  onCreated: (result: CreateReconstructionResult) => void;
  /** Smaller layout for use beside an existing model. */
  compact?: boolean;
  className?: string;
}

const EVIDENCE_CLASS_OPTIONS = [
  { value: "", label: "Ordinary photographs" },
  { value: "floor_plan", label: "Floor plans / layouts" },
  { value: "render", label: "Renders / CGI (visual reference only)" },
  { value: "infographic", label: "Infographics / diagrams" },
  { value: "historical_photo", label: "Historical photographs" },
] as const;

export default function PhotoDropZone({ worldId, onCreated, compact = false, className }: PhotoDropZoneProps) {
  const photosRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [evidenceClass, setEvidenceClass] = useState<string>("");

  const submit = useCallback(
    async (picked: File[]) => {
      const images = picked.filter(isImage);
      const skipped = picked.length - images.length;
      setError(null);
      setNotes([]);
      if (images.length === 0) {
        setError(
          picked.length === 0
            ? "No files were selected."
            : "None of those files look like images. Drop photographs (JPEG, PNG, HEIC, WebP, TIFF).",
        );
        return;
      }
      setBusy(`Uploading ${images.length} photo${images.length === 1 ? "" : "s"}…`);
      try {
        const result = await createReconstruction(images, {
          worldId: worldId ?? null,
          evidenceClass: evidenceClass || null,
        });
        const problems = [
          ...(skipped > 0 ? [`${skipped} non-image file${skipped === 1 ? "" : "s"} skipped`] : []),
          ...result.rejected.map((r) => `${r.name}: ${r.reason}`),
        ];
        if (problems.length > 0) setNotes(problems);
        onCreated(result);
      } catch (err) {
        if (isApiError(err) && err.details && typeof err.details === "object") {
          const d = err.details as { message?: string; rejected?: { name: string; reason: string }[] };
          setError(d.message ?? err.message);
          if (d.rejected?.length) setNotes(d.rejected.map((r) => `${r.name}: ${r.reason}`));
        } else {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setBusy(null);
      }
    },
    [worldId, evidenceClass, onCreated],
  );

  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (busy) return;
      try {
        await submit(await filesFromDrop(e.dataTransfer));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [busy, submit],
  );

  const onPick = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      e.target.value = ""; // allow re-selecting the same files
      void submit(files);
    },
    [submit],
  );

  const refining = Boolean(worldId);

  return (
    <div className={cn("w-full", className)}>
      <div
        role="group"
        aria-label={refining ? "Add photos to this reconstruction" : "Create a reconstruction from photos"}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "relative rounded-xl border-2 border-dashed text-center transition-colors",
          compact ? "px-4 py-5" : "px-6 py-12",
          dragging ? "border-[#00e5ff] bg-[#00e5ff]/10" : "border-[#2a2f3a] bg-[#0c0d12] hover:border-[#3a4150]",
        )}
      >
        {busy ? (
          <div className="flex flex-col items-center gap-3" role="status" aria-live="polite">
            <Loader2 className="h-7 w-7 animate-spin text-[#00e5ff]" />
            <p className="text-sm text-neutral-200">{busy}</p>
          </div>
        ) : (
          <>
            <UploadCloud className={cn("mx-auto text-[#00e5ff]", compact ? "h-6 w-6" : "h-10 w-10")} aria-hidden />
            <p className={cn("mt-3 font-semibold text-white", compact ? "text-sm" : "text-lg")}>
              {refining ? "Drop more photos to improve this model" : "Drop photos here to build a model"}
            </p>
            {!compact && (
              <p className="mx-auto mt-1 max-w-md text-xs text-neutral-400">
                One photograph gives a rough model. More views from different positions make it better.
                You never need to set anything up first.
              </p>
            )}
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={() => photosRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-md bg-[#00e5ff] px-3 py-1.5 text-xs font-semibold text-black hover:bg-[#33ebff]"
              >
                <ImagePlus className="h-4 w-4" aria-hidden /> Select photos
              </button>
              <button
                type="button"
                onClick={() => folderRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-md border border-[#2a2f3a] bg-[#151821] px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-neutral-800"
              >
                <FolderUp className="h-4 w-4" aria-hidden /> Upload folder
              </button>
              <button
                type="button"
                onClick={() => cameraRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-md border border-[#2a2f3a] bg-[#151821] px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-neutral-800"
              >
                <Camera className="h-4 w-4" aria-hidden /> Take photo
              </button>
            </div>
            {!compact && (
              <details className="mx-auto mt-4 max-w-xs text-left text-[11px] text-neutral-400">
                <summary className="cursor-pointer select-none text-center hover:text-neutral-200">
                  Not ordinary photographs?
                </summary>
                <label className="mt-2 block">
                  <span className="mb-1 block">These files are:</span>
                  <select
                    value={evidenceClass}
                    onChange={(e) => setEvidenceClass(e.target.value)}
                    className="w-full rounded border border-[#2a2f3a] bg-[#0c0d12] px-2 py-1 text-xs text-neutral-200"
                  >
                    {EVIDENCE_CLASS_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                  <span className="mt-1 block">
                    Non-photographs are kept as context and never mixed into observed geometry.
                  </span>
                </label>
              </details>
            )}
          </>
        )}
      </div>

      <input ref={photosRef} type="file" accept="image/*" multiple hidden onChange={onPick} data-testid="photo-input" />
      <input
        ref={folderRef}
        type="file"
        multiple
        hidden
        onChange={onPick}
        data-testid="folder-input"
        // non-standard but widely supported directory picker
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
      />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        onChange={onPick}
        data-testid="camera-input"
      />

      {error && (
        <div role="alert" className="mt-3 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}
      {notes.length > 0 && (
        <ul className="mt-2 space-y-0.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
          {notes.slice(0, 6).map((n) => (
            <li key={n}>{n}</li>
          ))}
          {notes.length > 6 && <li>…and {notes.length - 6} more</li>}
        </ul>
      )}
    </div>
  );
}
