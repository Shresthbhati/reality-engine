"use client";

import { useEffect, useState, type ComponentType } from "react";
import {
  X,
  Loader2,
  Download,
  Box,
  Package,
  Building2,
  Code2,
  FileJson,
  Map as MapIcon,
  Globe,
  Gamepad2,
} from "lucide-react";
import {
  exportWorld,
  listExportFormats,
  type ExportFormat,
  type ExportFormatStatus,
} from "@/lib/api/export";
import { downloadArtifact } from "@/lib/api/download";
import { isApiError } from "@/lib/api/client";

const FORMATS: { value: ExportFormat; label: string; ext: string; Icon: ComponentType<{ className?: string }> }[] = [
  { value: "gltf", label: "glTF", ext: "gltf", Icon: Box },
  { value: "usda", label: "USD", ext: "usda", Icon: Package },
  { value: "ifc", label: "IFC", ext: "ifc", Icon: Building2 },
  { value: "blender", label: "Blender", ext: "py", Icon: Code2 },
  { value: "cityjson", label: "CityJSON", ext: "json", Icon: FileJson },
  { value: "citygml", label: "CityGML", ext: "gml", Icon: Globe },
  { value: "geojson", label: "GeoJSON", ext: "geojson", Icon: MapIcon },
  { value: "godot", label: "Godot", ext: "json", Icon: Gamepad2 },
];

/** Real export via sdk.reality.export() (apps/api/routes_export.py) -- no
 * client-side format conversion, no fabricated file content. One click
 * exports and downloads; a format the world can't produce is disabled with
 * the server's one-line reason, and a failed export shows its reason. */
export default function ExportPanel({
  worldId,
  onClose,
}: {
  worldId: string;
  onClose: () => void;
}) {
  const [status, setStatus] = useState<Record<string, ExportFormatStatus>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [running, setRunning] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<{ title: string; detail: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    listExportFormats(worldId)
      .then((list) => {
        if (cancelled) return;
        setStatus(Object.fromEntries(list.map((f) => [f.format, f])));
        setLoadError(null);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(isApiError(err) ? err.describe() : "Could not check export formats.");
      });
    return () => {
      cancelled = true;
    };
  }, [worldId]);

  async function handleExport(format: ExportFormat, ext: string) {
    setRunning(format);
    setError(null);
    setDone(null);
    try {
      const result = await exportWorld(worldId, format);
      await downloadArtifact(result.download_url, `${worldId}.${ext}`);
      const skipped = result.entities_skipped.length;
      setDone({
        title:
          `${format} downloaded · ${result.entities_exported.length} entities` +
          (skipped > 0 ? `, ${skipped} skipped` : ""),
        detail: result.entities_skipped
          .map((id, i) => `${id}: ${result.skip_reasons[i] ?? "unknown reason"}`)
          .join("\n"),
      });
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Export failed.");
    } finally {
      setRunning(null);
    }
  }

  return (
    <div
      className="absolute top-4 right-4 z-20 w-72 rounded-lg p-4 flex flex-col gap-3 shadow-xl"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-1.5" style={{ color: "var(--text-primary)" }}>
          <Download className="w-3.5 h-3.5" />
          Export
        </h3>
        <button type="button" onClick={onClose} aria-label="Close">
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {FORMATS.map(({ value, label, ext, Icon }) => {
          const s = status[value];
          const reason = loadError ?? (s && !s.available ? s.reason : null);
          const disabled = running !== null || Boolean(reason) || (!s && !loadError);
          return (
            <button
              key={value}
              type="button"
              onClick={() => handleExport(value, ext)}
              disabled={disabled}
              title={reason ?? `Export ${label}`}
              aria-label={reason ? `${label} unavailable: ${reason}` : `Export ${label}`}
              className="flex flex-col items-center justify-center gap-1 h-16 rounded-md text-[11px] disabled:opacity-35 transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              {running === value ? <Loader2 className="w-4 h-4 animate-spin" /> : <Icon className="w-4 h-4" />}
              {label}
            </button>
          );
        })}
      </div>

      {loadError && (
        <p role="alert" className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {loadError}
        </p>
      )}
      {error && (
        <p role="alert" className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {error}
        </p>
      )}
      {done && (
        <p className="text-xs" style={{ color: "var(--text-tertiary)" }} title={done.detail || undefined}>
          {done.title}
        </p>
      )}
    </div>
  );
}
