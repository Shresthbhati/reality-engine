"use client";

import { useEffect, useState, useMemo, type ComponentType } from "react";
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
  FileText,
  FileCode,
  Layers,
  Grid,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  exportWorld,
  listExportFormats,
  type ExportFormat,
  type ExportFormatStatus,
} from "@/lib/api/export";
import { downloadArtifact } from "@/lib/api/download";
import { isApiError } from "@/lib/api/client";

interface FormatDefinition {
  value: ExportFormat;
  label: string;
  ext: string;
  category: "3d" | "bim" | "docs";
  description: string;
  Icon: ComponentType<{ className?: string }>;
}

const ALL_FORMATS: FormatDefinition[] = [
  // 3D Geometry
  { value: "gltf", label: "glTF 2.0", ext: "gltf", category: "3d", description: "Standard 3D JSON bundle", Icon: Box },
  { value: "glb", label: "GLB Binary", ext: "glb", category: "3d", description: "Self-contained binary glTF", Icon: Box },
  { value: "usda", label: "USD", ext: "usda", category: "3d", description: "Universal Scene Description", Icon: Package },
  { value: "obj", label: "Wavefront OBJ", ext: "obj", category: "3d", description: "Polygon geometry format", Icon: FileCode },
  { value: "ply", label: "Stanford PLY", ext: "ply", category: "3d", description: "Dense point cloud / mesh", Icon: Layers },
  { value: "stl", label: "StereoLithography", ext: "stl", category: "3d", description: "3D printing mesh", Icon: Grid },

  // BIM, Engines & GIS
  { value: "ifc", label: "IFC4 BIM", ext: "ifc", category: "bim", description: "Industry Foundation Classes", Icon: Building2 },
  { value: "blender", label: "Blender Script", ext: "py", category: "bim", description: "Native Python scene importer", Icon: Code2 },
  { value: "godot", label: "Godot Engine", ext: "json", category: "bim", description: "Game engine asset pack", Icon: Gamepad2 },
  { value: "cityjson", label: "CityJSON", ext: "json", category: "bim", description: "3D city & GIS model", Icon: FileJson },
  { value: "citygml", label: "CityGML", ext: "gml", category: "bim", description: "OpenGIS urban features", Icon: Globe },
  { value: "geojson", label: "GeoJSON", ext: "geojson", category: "bim", description: "2D spatial coordinates", Icon: MapIcon },

  // 2D & Documentation
  { value: "floorplan_svg", label: "SVG Plan", ext: "svg", category: "docs", description: "Scalable vector floor plan", Icon: FileText },
  { value: "floorplan_dxf", label: "DXF CAD Plan", ext: "dxf", category: "docs", description: "AutoCAD 2D drawing exchange", Icon: FileCode },
  { value: "room_report_pdf", label: "Room Report PDF", ext: "pdf", category: "docs", description: "Room dimensions & compliance", Icon: FileText },
  { value: "floorplan_pdf", label: "Floor Plan PDF", ext: "pdf", category: "docs", description: "Architectural 1-sheet PDF", Icon: FileText },
  { value: "multisheet_floorplan_pdf", label: "Multi-Sheet PDF", ext: "pdf", category: "docs", description: "Full multi-page drawing set", Icon: FileText },
];

export default function ExportPanel({
  worldId,
  onClose,
}: {
  worldId: string;
  onClose: () => void;
}) {
  const [status, setStatus] = useState<Record<string, ExportFormatStatus>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<"all" | "3d" | "bim" | "docs">("all");
  const [running, setRunning] = useState<ExportFormat | null>(null);
  const [lastAttempt, setLastAttempt] = useState<{ format: ExportFormat; ext: string } | null>(null);
  const [error, setError] = useState<{ message: string; code?: string; detail?: string } | null>(null);
  const [done, setDone] = useState<{
    format: string;
    title: string;
    entitiesExported: number;
    entitiesSkipped: number;
    reasons: string[];
    showDetails: boolean;
  } | null>(null);

  const [formatsTick, setFormatsTick] = useState(0);
  const fetchFormats = () => {
    setLoadError(null);
    setFormatsTick((t) => t + 1);
  };

  useEffect(() => {
    let cancelled = false;
    listExportFormats(worldId)
      .then((list) => {
        if (!cancelled) setStatus(Object.fromEntries(list.map((f) => [f.format, f])));
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(isApiError(err) ? err.describe() : "Could not check export format availability from server.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [worldId, formatsTick]);
  const filteredFormats = useMemo(() => {
    if (activeCategory === "all") return ALL_FORMATS;
    return ALL_FORMATS.filter((f) => f.category === activeCategory);
  }, [activeCategory]);

  async function handleExport(format: ExportFormat, ext: string) {
    setRunning(format);
    setLastAttempt({ format, ext });
    setError(null);
    setDone(null);
    try {
      const result = await exportWorld(worldId, format);
      await downloadArtifact(result.download_url, `${worldId}.${ext}`);
      const skipped = result.entities_skipped.length;
      setDone({
        format,
        title: `${format.toUpperCase()} export completed and downloaded.`,
        entitiesExported: result.entities_exported.length,
        entitiesSkipped: skipped,
        reasons: result.entities_skipped.map(
          (id, i) => `${id}: ${result.skip_reasons[i] ?? "unknown constraint"}`
        ),
        showDetails: false,
      });
    } catch (err: unknown) {
      if (isApiError(err)) {
        setError({
          message: err.describe(),
          code: err.code,
          detail: err.message,
        });
      } else {
        setError({
          message: err instanceof Error ? err.message : "Export process encountered an unexpected failure.",
        });
      }
    } finally {
      setRunning(null);
    }
  }

  return (
    <div
      role="dialog"
      aria-labelledby="export-panel-title"
      className="absolute top-4 right-4 z-20 w-96 max-h-[85vh] overflow-y-auto rounded-lg p-4 flex flex-col gap-3 shadow-2xl backdrop-blur-md"
      style={{
        background: "var(--bg-surface)",
        border: "1px solid var(--border)",
        color: "var(--text-primary)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-1 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <Download className="w-4 h-4 text-[var(--accent)]" />
          <h3 id="export-panel-title" className="text-sm font-semibold">
            Universal Exporter
          </h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close export dialog"
          className="p-1 rounded hover:bg-[var(--surface-hover)] transition-colors"
        >
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      {/* Category Tabs */}
      <div className="flex items-center gap-1 p-0.5 rounded-md bg-[var(--surface-sunken)] border text-[11px]" style={{ borderColor: "var(--border)" }}>
        {(
          [
            { id: "all", label: "All Formats" },
            { id: "3d", label: "3D Meshes" },
            { id: "bim", label: "BIM / CAD" },
            { id: "docs", label: "Plans & Docs" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveCategory(tab.id)}
            className={`flex-1 py-1 rounded text-center font-medium transition-colors cursor-pointer ${
              activeCategory === tab.id
                ? "bg-[var(--surface-raised)] text-[var(--text-primary)] shadow-sm"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Format Grid */}
      <div className="grid grid-cols-2 gap-2 max-h-60 overflow-y-auto pr-1">
        {filteredFormats.map(({ value, label, ext, description, Icon }) => {
          const s = status[value];
          const reason = loadError ?? (s && !s.available ? s.reason : null);
          const disabled = running !== null || Boolean(reason) || (!s && !loadError);

          return (
            <button
              key={value}
              type="button"
              onClick={() => void handleExport(value, ext)}
              disabled={disabled}
              title={reason ?? `${label}: ${description}`}
              aria-label={reason ? `${label} unavailable: ${reason}` : `Export ${label}`}
              className="flex items-center gap-2.5 p-2 rounded-md text-left transition-all border disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed group relative"
              style={{
                background: "var(--surface-sunken)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-8 h-8 rounded shrink-0 flex items-center justify-center transition-colors group-hover:scale-105"
                style={{
                  background: "var(--accent-subtle)",
                  color: "var(--accent)",
                }}
              >
                {running === value ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold truncate leading-tight group-hover:text-[var(--accent)]">
                  {label}
                </div>
                <div className="text-[10px] text-[var(--text-tertiary)] truncate leading-tight mt-0.5">
                  {reason ?? description}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Prominent User-Facing Error Banner */}
      {error && (
        <div
          role="alert"
          className="rounded-md p-3 flex flex-col gap-1.5 border text-xs"
          style={{
            background: "rgba(239, 68, 68, 0.08)",
            borderColor: "rgba(239, 68, 68, 0.3)",
            color: "var(--error)",
          }}
        >
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-error" />
            <div className="flex-1">
              <span className="font-semibold block">Export Failed</span>
              <p className="text-[11px] leading-relaxed text-fg-muted mt-0.5">{error.message}</p>
            </div>
            {lastAttempt && (
              <button
                type="button"
                onClick={() => void handleExport(lastAttempt.format, lastAttempt.ext)}
                disabled={running !== null}
                className="flex items-center gap-1 px-2 py-1 rounded bg-error-subtle hover:bg-error-subtle text-error text-[10px] font-medium border border-error/30 transition-colors"
              >
                <RefreshCw className="w-3 h-3" />
                Retry
              </button>
            )}
          </div>
        </div>
      )}

      {/* Formats Availability Load Error */}
      {loadError && (
        <div
          role="alert"
          className="rounded-md p-2.5 flex items-center justify-between border text-xs"
          style={{
            background: "rgba(245, 158, 11, 0.08)",
            borderColor: "rgba(245, 158, 11, 0.3)",
            color: "var(--warning)",
          }}
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-warning" />
            <span className="text-[11px]">{loadError}</span>
          </div>
          <button
            type="button"
            onClick={fetchFormats}
            className="p-1 rounded hover:bg-warning-subtle text-warning transition-colors"
            title="Retry format status check"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Success Notification with Skip Diagnostics */}
      {done && (
        <div
          className="rounded-md p-3 flex flex-col gap-1.5 border text-xs"
          style={{
            background: "rgba(16, 185, 129, 0.08)",
            borderColor: "rgba(16, 185, 129, 0.3)",
            color: "var(--success)",
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
              <span className="font-semibold text-success">{done.title}</span>
            </div>
          </div>
          <div className="text-[11px] text-fg-muted">
            Exported {done.entitiesExported} entities successfully.
            {done.entitiesSkipped > 0 && ` (${done.entitiesSkipped} skipped due to non-mesh geometry)`}
          </div>
          {done.reasons.length > 0 && (
            <div className="mt-1 pt-1 border-t border-success/20">
              <button
                type="button"
                onClick={() => setDone({ ...done, showDetails: !done.showDetails })}
                className="flex items-center gap-1 text-[10px] text-fg-muted hover:text-fg transition-colors"
              >
                <span>{done.showDetails ? "Hide skipped details" : "View skipped reasons"}</span>
                {done.showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
              {done.showDetails && (
                <div className="mt-1 p-1.5 rounded bg-black/40 font-mono text-[10px] text-fg-muted max-h-24 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                  {done.reasons.join("\n")}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
