"use client";

import { useState } from "react";
import { X, Loader2, Download } from "lucide-react";
import { exportWorld, type ExportFormat, type ExportResult } from "@/lib/api/export";
import { isApiError } from "@/lib/api/client";
import { apiBaseUrl } from "@/lib/api/client";

const FORMATS: { value: ExportFormat; label: string }[] = [
  { value: "gltf", label: "glTF (.gltf)" },
  { value: "usda", label: "USD ASCII (.usda)" },
  { value: "blender", label: "Blender script (.py)" },
  { value: "cityjson", label: "CityJSON" },
  { value: "citygml", label: "CityGML" },
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
      setResult(await exportWorld(worldId, format));
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

      <label className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
        Format
        <select
          value={format}
          onChange={(e) => {
            setFormat(e.target.value as ExportFormat);
            setResult(null);
          }}
          className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
          style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
        >
          {FORMATS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
      </label>

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
            Download {result.format}
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
        </div>
      )}
    </div>
  );
}
