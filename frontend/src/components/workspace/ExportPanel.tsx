"use client";

import { useState } from "react";
import { X, Loader2, Download } from "lucide-react";
import { exportWorld, type ExportFormat, type ExportResult, downloadArtifact } from "@/lib/api/export";
import { isApiError } from "@/lib/api/client";
import { apiBaseUrl } from "@/lib/api/client";

const FORMATS: { value: ExportFormat; label: string }[] = [
  { value: "gltf", label: "glTF (.gltf)" },
  { value: "usda", label: "USD ASCII (.usda)" },
  { value: "ifc", label: "IFC (.ifc)" },
  { value: "blender", label: "Blender script (.py)" },
  { value: "cityjson", label: "CityJSON" },
  { value: "citygml", label: "CityGML" },
  { value: "geojson", label: "GIS footprints (GeoJSON)" },
  { value: "godot", label: "Godot bundle (JSON: glb + tscn + manifest)" },
];

/** Real export via sdk.reality.export() (apps/api/routes_export.py) --
 * no client-side format conversion, no fabricated file content. The
 * download link points straight at the backend's content-addressed
 * artifact, same host apiGet/apiPost already talk to. */
export default function ExportPanel({
  worldId,
  onClose,
}: {
  worldId: string;
  onClose: () => void;
}) {
  const [format, setFormat] = useState<ExportFormat>("gltf");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);

  async function handleExport() {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const result = await exportWorld(worldId, format);
      setResult(result);

      // Attempt download with proper error handling/toasts
      await downloadArtifact(result, (msg) => setError(msg));
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Export failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div
      className="absolute top-4 right-4 z-20 w-80 rounded-lg p-4 flex flex-col gap-3 shadow-xl"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-1.5" style={{ color: "var(--text-primary)" }}>
          <Download className="w-3.5 h-3.5" />
          Export World
        </h3>
        <button type="button" onClick={onClose} aria-label="Close">
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      < div className="flex flex-col gap-2" style={{ marginTop: "0.5rem" }}>
        {FORMATS.map((f) => {
          const isDisabled = running || (result && result.entities_skipped.includes(f.value) /* simplified check */);
          return (
            <button
              key={f.value}
              type="button"
              onClick={() => { setFormat(f.value);
                if (!isDisabled && result) {
                  void downloadArtifact(result, setError);
                }
              }}
              disabled={isDisabled || !result}
              className="flex flex-col items-center gap-1 rounded-md p-2 transition-colors"
              style={{
                background: isDisabled ? "var(--bg-subtle)" : "var(--accent-subtle)",
                color: isDisabled ? "var(--text-tertiary)" : "var(--accent)",
                border: "1px solid " + (isDisabled ? "var(--border-subtle)" : "var(--accent-border)"),
              }}
            >
              <Download className="w-4 h-4" />
              <span className="text-xs">{f.label}</span>
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={handleExport}
        disabled={running}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-medium disabled:opacity-40 transition-colors"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        {running && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {running ? "Exporting…" : "Export"}
      </button>

      {error && (
        <p className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {error}
        </p>
      )}

      {result && (
        <div className="flex flex-col gap-2">
          <a
            href={`${apiBaseUrl()}${result.download_url}`}
            className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-medium transition-colors"
            style={{ background: "var(--accent)", color: "var(--bg-base)" }}
          >
            <Download className="w-3.5 h-3.5" />
            Download {result.format} via helper
          </a>
          <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {result.entities_exported.length} entities exported
            {result.entities_skipped.length > 0 && `, ${result.entities_skipped.length} skipped`}
          </p>
          {result.entities_skipped.length > 0 && (
            <div className="flex flex-col gap-0.5 text-[11px]" style={{ color: "var(--text-tertiary)" }}>
              {result.entities_skipped.map((id, i) => (
                <div key={id}>
                  {id}: {result.skip_reasons[i] ?? "unknown reason"}
                </div>
              ))}
            </div>
          )}
{/* New: Download via artifact helper with error handling */}
          {result.download_url && (
            <button
              type="button"
              onClick={() => void downloadArtifact(result, setError)}
              className="mt-2 flex items-center justify-center gap-2 h-8 rounded-md text-xs font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              <Download className="w-3 h-3" /> Download via helper
            </button>
          )}
        </div>
      )}
    </div>
  );
}
