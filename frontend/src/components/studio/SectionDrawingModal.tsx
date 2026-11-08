"use client";

import React, { useState, useEffect } from "react";
import {
  Download,
  FileText,
  Layers,
  Moon,
  Sun,
  X,
  Compass,
} from "lucide-react";
import {
  type DrawingPreset,
  type DrawingType,
  type SheetSize,
  getDrawingTypeMeta,
} from "@/lib/studio/drawing-sheet-model";
import {
  downloadDrawingDxf,
  downloadDrawingSvg,
  fetchDrawingPresets,
  getDrawingSvgUrl,
} from "@/lib/api/drawings";

export interface SectionDrawingModalProps {
  worldId: string;
  projectName?: string;
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

export function SectionDrawingModal({
  worldId,
  projectName = "Reality Engine Architecture",
  isOpen,
  onClose,
  className = "",
}: SectionDrawingModalProps) {
  const [presets, setPresets] = useState<DrawingPreset[]>([]);
  const [selectedType, setSelectedType] = useState<DrawingType>("section_longitudinal");
  const [scale, setScale] = useState<string>("1:100");
  const [sheetSize, setSheetSize] = useState<SheetSize>("A2");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    if (!isOpen || !worldId) return;
    let active = true;

    fetchDrawingPresets(worldId)
      .then((res) => {
        if (active) {
          setPresets(res.presets);
          if (res.presets.length > 0 && !res.presets.some((p) => p.type === selectedType)) {
            setSelectedType(res.presets[0].type);
          }
        }
      })
      .catch((err) => {
        console.warn("Failed to load drawing presets:", err);
      });

    return () => {
      active = false;
    };
  }, [isOpen, worldId]);

  if (!isOpen) return null;

  const svgUrl = getDrawingSvgUrl(worldId, {
    drawingType: selectedType,
    scaleLabel: scale,
    sheetSize: sheetSize,
    theme: theme,
  });

  const handleDownloadDxf = async () => {
    setIsExporting(true);
    try {
      await downloadDrawingDxf(worldId, {
        drawingType: selectedType,
        scaleLabel: scale,
        sheetSize: sheetSize,
      });
    } catch (err) {
      console.warn("DXF export failed:", err);
    } finally {
      setIsExporting(false);
    }
  };

  const handleDownloadSvg = async () => {
    setIsExporting(true);
    try {
      await downloadDrawingSvg(worldId, {
        drawingType: selectedType,
        scaleLabel: scale,
        sheetSize: sheetSize,
        theme: theme,
      });
    } catch (err) {
      console.warn("SVG export failed:", err);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className={`relative w-full max-w-6xl h-[88vh] bg-panel border border-border rounded-xl shadow-2xl flex flex-col overflow-hidden ${className}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3.5 border-b border-border bg-surface/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 text-accent">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-fg">
                Architectural Building Sections & Elevations
              </h2>
              <p className="text-xs text-muted-fg">
                {projectName}  *  AutoCAD DXF & Vector Sheet Packages
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center rounded-lg border border-border bg-panel p-0.5">
              <button
                onClick={() => setTheme("dark")}
                className={`p-1.5 rounded-md text-xs font-medium flex items-center gap-1 transition-colors ${
                  theme === "dark" ? "bg-accent/15 text-accent" : "text-muted-fg hover:text-fg"
                }`}
                title="Studio Dark Preview"
              >
                <Moon className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setTheme("light")}
                className={`p-1.5 rounded-md text-xs font-medium flex items-center gap-1 transition-colors ${
                  theme === "light" ? "bg-accent/15 text-accent" : "text-muted-fg hover:text-fg"
                }`}
                title="Print Light Sheet"
              >
                <Sun className="w-3.5 h-3.5" />
              </button>
            </div>

            <button
              onClick={handleDownloadSvg}
              disabled={isExporting}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-border bg-surface hover:bg-surface/80 text-fg flex items-center gap-1.5 transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-accent" />
              Export SVG
            </button>

            <button
              onClick={handleDownloadDxf}
              disabled={isExporting}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              Export DXF (AutoCAD)
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-surface text-muted-fg hover:text-fg transition-colors ml-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Workspace Body */}
        <div className="flex-1 flex overflow-hidden">
          {/* Controls & Presets Sidebar */}
          <div className="w-72 border-r border-border bg-surface/30 p-4 flex flex-col gap-4 overflow-y-auto">
            {/* Drawing Presets */}
            <div>
              <label className="text-[11px] font-semibold text-muted-fg uppercase tracking-wider block mb-2">
                Cutting Planes & Facades
              </label>
              <div className="space-y-1.5">
                {presets.map((preset) => {
                  const meta = getDrawingTypeMeta(preset.type);
                  const isSelected = selectedType === preset.type;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => setSelectedType(preset.type)}
                      className={`w-full text-left p-2.5 rounded-lg border transition-all text-xs ${
                        isSelected
                          ? "border-accent bg-accent/10 text-fg shadow-sm"
                          : "border-border/60 bg-panel hover:bg-surface text-muted-fg hover:text-fg"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-fg">{preset.title}</span>
                        <span className={`text-[9px] px-1.5 py-0.2 rounded border font-medium ${meta.badgeClass}`}>
                          {meta.category.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-[10px] text-muted-fg line-clamp-1">{preset.axis}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Scale & Sheet Layout Settings */}
            <div className="space-y-3 pt-3 border-t border-border">
              <label className="text-[11px] font-semibold text-muted-fg uppercase tracking-wider block">
                Drawing Configuration
              </label>

              <div>
                <label className="text-xs text-muted-fg block mb-1">Architectural Scale</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {["1:50", "1:100", "1:200"].map((s) => (
                    <button
                      key={s}
                      onClick={() => setScale(s)}
                      className={`py-1.5 text-xs font-medium rounded-md border text-center transition-colors ${
                        scale === s
                          ? "border-accent bg-accent/10 text-accent"
                          : "border-border bg-panel text-muted-fg hover:text-fg"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs text-muted-fg block mb-1">Standard Sheet Size</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(["A1", "A2", "A3"] as SheetSize[]).map((ss) => (
                    <button
                      key={ss}
                      onClick={() => setSheetSize(ss)}
                      className={`py-1.5 text-xs font-medium rounded-md border text-center transition-colors ${
                        sheetSize === ss
                          ? "border-accent bg-accent/10 text-accent"
                          : "border-border bg-panel text-muted-fg hover:text-fg"
                      }`}
                    >
                      {ss}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Standards & Layer Legend */}
            <div className="mt-auto p-3 rounded-lg border border-border bg-panel text-[11px] space-y-1.5 text-muted-fg">
              <div className="font-semibold text-fg flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-accent" />
                AIA CAD Layer Standards
              </div>
              <div className="flex items-center justify-between">
                <span>Heavy Cut Geometry</span>
                <span className="font-mono text-[10px] text-fg">A-WALL-CUT (0.50mm)</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Background Projections</span>
                <span className="font-mono text-[10px] text-fg">A-WALL-PROJ (0.18mm)</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Datum Levels & Dims</span>
                <span className="font-mono text-[10px] text-fg">A-ANNO-DATM</span>
              </div>
            </div>
          </div>

          {/* Drawing Sheet Viewport */}
          <div className="flex-1 bg-surface flex items-center justify-center p-6 overflow-hidden">
            <div className="w-full h-full rounded-lg border border-border shadow-inner overflow-hidden flex items-center justify-center bg-panel relative">
              <iframe
                src={svgUrl}
                title="Architectural Section Drawing"
                className="w-full h-full border-0 select-none"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
