"use client";

import React, { useState, useEffect } from "react";
import {
  Calculator,
  Download,
  FileSpreadsheet,
  FileText,
  Layers,
  RefreshCw,
  Sliders,
  X,
  Home,
} from "lucide-react";
import {
  type TakeoffReport,
  type UnitRateOverride,
  formatCurrency,
} from "@/lib/studio/takeoff-model";
import {
  fetchMaterialTakeoff,
  calculateCustomEstimate,
  downloadTakeoffCsv,
  downloadTakeoffPdf,
} from "@/lib/api/takeoff";

export interface TakeoffScheduleModalProps {
  worldId: string;
  projectName?: string;
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

type ModalTab = "divisions" | "rooms" | "items" | "customizer";

export function TakeoffScheduleModal({
  worldId,
  projectName = "Architectural Reality Model",
  isOpen,
  onClose,
  className = "",
}: TakeoffScheduleModalProps) {
  const [report, setReport] = useState<TakeoffReport | null>(null);
  const [activeTab, setActiveTab] = useState<ModalTab>("divisions");
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Rate overrides state for customizer tab
  const [rateOverrides, setRateOverrides] = useState<Record<string, UnitRateOverride>>({
    "09 29 00": { material_cost: 16.5, labor_cost: 22.0, waste_percent: 10.0 }, // Gypsum Drywall
    "06 11 00": { material_cost: 28.0, labor_cost: 32.0, waste_percent: 8.0 },  // Wood Framing
    "09 64 00": { material_cost: 65.0, labor_cost: 40.0, waste_percent: 10.0 }, // Hardwood
    "09 30 13": { material_cost: 45.0, labor_cost: 55.0, waste_percent: 12.0 }, // Ceramic Tile
    "06 20 00": { material_cost: 14.0, labor_cost: 18.0, waste_percent: 7.0 },  // Baseboard Trim
  });

  useEffect(() => {
    if (!isOpen || !worldId) return;
    let active = true;

    fetchMaterialTakeoff(worldId)
      .then((data) => {
        if (active) {
          setReport(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        console.warn("Failed to load material takeoff:", err);
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [isOpen, worldId]);

  if (!isOpen) return null;

  const handleDownloadCsv = async () => {
    setIsExporting(true);
    setStatusMessage("Exporting CSI spreadsheet...");
    try {
      await downloadTakeoffCsv(worldId);
      setStatusMessage("CSV exported successfully");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to export CSV");
    } finally {
      setIsExporting(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  const handleDownloadPdf = async () => {
    setIsExporting(true);
    setStatusMessage("Compiling PDF-1.4 BoQ Certificate...");
    try {
      await downloadTakeoffPdf(worldId, report?.project_name || projectName);
      setStatusMessage("PDF certificate downloaded");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to export PDF");
    } finally {
      setIsExporting(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  const handleRecalculateEstimate = async () => {
    setIsLoading(true);
    setStatusMessage("Recalculating with custom rates...");
    try {
      const updated = await calculateCustomEstimate(worldId, {
        projectName: report?.project_name || projectName,
        rateOverrides,
      });
      setReport(updated);
      setStatusMessage("Cost estimate updated");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to calculate estimate");
    } finally {
      setIsLoading(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  const handleResetDefaults = async () => {
    setIsLoading(true);
    try {
      const def = await fetchMaterialTakeoff(worldId);
      setReport(def);
      setStatusMessage("Reset to standard CSI rates");
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  return (
    <div className={`fixed inset-0 z-50 bg-base/80 backdrop-blur-sm flex items-center justify-center p-4 ${className}`}>
      <div className="bg-panel border border-border rounded-xl shadow-2xl w-full max-w-5xl h-[88vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-accent/10 border border-accent/20">
              <Calculator className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h2 className="text-base font-bold text-fg tracking-tight">
                Bill of Quantities & Material Takeoff
              </h2>
              <p className="text-xs text-muted-fg">
                Conforming to CSI MasterFormat 2020 · Net surface deductions & waste analysis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {statusMessage && (
              <span className="text-xs text-accent px-2 py-1 rounded bg-accent/10 border border-accent/20">
                {statusMessage}
              </span>
            )}

            <button
              onClick={handleDownloadCsv}
              disabled={isExporting || isLoading || !report}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-border bg-surface hover:bg-surface/80 text-fg flex items-center gap-1.5 transition-colors disabled:opacity-50"
              title="Download tabular CSV spreadsheet for Excel / cost estimation software"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-accent" />
              Export CSV
            </button>

            <button
              onClick={handleDownloadPdf}
              disabled={isExporting || isLoading || !report}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity shadow-sm disabled:opacity-50"
              title="Download formal zero-dependency PDF-1.4 BoQ Schedule Certificate"
            >
              <Download className="w-3.5 h-3.5" />
              Export PDF
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-surface text-muted-fg hover:text-fg transition-colors ml-2"
              title="Close Takeoff Schedule"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Executive Summary Metrics Ribbon */}
        {report && (
          <div className="grid grid-cols-5 gap-3 px-6 py-3 border-b border-border bg-surface/20 text-xs">
            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Total Budget
              </span>
              <div className="text-sm font-bold text-fg">
                {formatCurrency(report.summary.grand_total_cost, report.currency)}
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Mat: {formatCurrency(report.summary.grand_total_material, report.currency)} · Lab: {formatCurrency(report.summary.grand_total_labor, report.currency)}
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Unit Cost Rate
              </span>
              <div className="text-sm font-bold text-accent">
                {formatCurrency(report.summary.cost_per_m2, report.currency)} / m²
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Based on measured floor area
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Net Floor Area
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_floor_area_m2.toFixed(1)} m²
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Across {report.rooms.length} room schedules
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Net Wall Surface
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_net_wall_area_m2.toFixed(1)} m²
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Deducted opening cutouts
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Openings & Assemblies
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_openings_count} Doors & Glazing
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                {report.items_count} line items tracked
              </div>
            </div>
          </div>
        )}

        {/* Tab Selection Header */}
        <div className="flex items-center gap-1 px-6 pt-3 border-b border-border bg-surface/10">
          <button
            onClick={() => setActiveTab("divisions")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "divisions"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            CSI Divisions Breakdown
          </button>

          <button
            onClick={() => setActiveTab("rooms")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "rooms"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Home className="w-3.5 h-3.5" />
            Room Takeoff Schedule
          </button>

          <button
            onClick={() => setActiveTab("items")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "items"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            Line Items ({report?.items_count ?? 0})
          </button>

          <button
            onClick={() => setActiveTab("customizer")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "customizer"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Unit Rates & Waste Override
          </button>
        </div>

        {/* Tab Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading && !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg gap-3">
              <RefreshCw className="w-6 h-6 animate-spin text-accent" />
              <p className="text-xs">Computing geometric surface deductions and BoQ...</p>
            </div>
          ) : !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg">
              <p className="text-xs">No takeoff data available for this world model.</p>
            </div>
          ) : (
            <>
              {/* TAB 1: CSI DIVISIONS BREAKDOWN */}
              {activeTab === "divisions" && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {report.divisions.map((div) => (
                      <div
                        key={div.division}
                        className="p-4 rounded-xl border border-border bg-surface/30 hover:border-accent/40 transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-xs text-fg">
                            {div.division_name}
                          </span>
                          <span className="text-xs font-bold text-accent">
                            {div.percentage_of_total.toFixed(1)}%
                          </span>
                        </div>

                        {/* Progress visual bar */}
                        <div className="w-full h-2 rounded-full bg-surface border border-border/50 overflow-hidden mb-3">
                          <div
                            className="h-full bg-accent rounded-full transition-all duration-300"
                            style={{ width: `${Math.min(100, Math.max(2, div.percentage_of_total))}%` }}
                          />
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-[11px] pt-1 border-t border-border/50">
                          <div>
                            <span className="text-muted-fg block">Items</span>
                            <span className="font-medium text-fg">{div.item_count}</span>
                          </div>
                          <div>
                            <span className="text-muted-fg block">Material</span>
                            <span className="font-medium text-fg">{formatCurrency(div.total_material_cost, report.currency)}</span>
                          </div>
                          <div>
                            <span className="text-muted-fg block">Labor</span>
                            <span className="font-medium text-fg">{formatCurrency(div.total_labor_cost, report.currency)}</span>
                          </div>
                        </div>

                        <div className="mt-2.5 pt-2 border-t border-border/40 flex items-center justify-between text-xs">
                          <span className="text-muted-fg">Subtotal:</span>
                          <span className="font-bold text-fg">{formatCurrency(div.total_cost, report.currency)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 2: ROOM SCHEDULE */}
              {activeTab === "rooms" && (
                <div className="overflow-x-auto border border-border rounded-xl bg-surface/20">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface/60 border-b border-border text-[11px] uppercase tracking-wider text-muted-fg font-semibold">
                      <tr>
                        <th className="py-2.5 px-3">Room Name</th>
                        <th className="py-2.5 px-3">Floor Area</th>
                        <th className="py-2.5 px-3">Perimeter</th>
                        <th className="py-2.5 px-3">Gross Wall</th>
                        <th className="py-2.5 px-3">Openings (D/W)</th>
                        <th className="py-2.5 px-3">Net Wall Area</th>
                        <th className="py-2.5 px-3">Baseboard Trim</th>
                        <th className="py-2.5 px-3">Finishes (F/W)</th>
                        <th className="py-2.5 px-3 text-right">Room Cost</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {report.rooms.map((rm) => (
                        <tr key={rm.room_id} className="hover:bg-surface/40 transition-colors">
                          <td className="py-2.5 px-3 font-semibold text-fg">{rm.room_name}</td>
                          <td className="py-2.5 px-3 font-medium text-fg">{rm.floor_area_m2.toFixed(2)} m²</td>
                          <td className="py-2.5 px-3 text-muted-fg">{rm.perimeter_m.toFixed(2)} m</td>
                          <td className="py-2.5 px-3 text-muted-fg">{rm.wall_gross_area_m2.toFixed(2)} m²</td>
                          <td className="py-2.5 px-3 text-muted-fg">{rm.doors_count} doors, {rm.windows_count} win</td>
                          <td className="py-2.5 px-3 font-medium text-accent">{rm.wall_net_area_m2.toFixed(2)} m²</td>
                          <td className="py-2.5 px-3 text-muted-fg">{rm.baseboard_length_m.toFixed(2)} m</td>
                          <td className="py-2.5 px-3 text-muted-fg capitalize">{rm.floor_finish} / {rm.wall_finish}</td>
                          <td className="py-2.5 px-3 font-bold text-fg text-right">{formatCurrency(rm.total_cost, report.currency)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* TAB 3: LINE ITEMS BILL OF QUANTITIES */}
              {activeTab === "items" && (
                <div className="overflow-x-auto border border-border rounded-xl bg-surface/20">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface/60 border-b border-border text-[11px] uppercase tracking-wider text-muted-fg font-semibold">
                      <tr>
                        <th className="py-2.5 px-3">CSI Code</th>
                        <th className="py-2.5 px-3">Description</th>
                        <th className="py-2.5 px-3">Room / Space</th>
                        <th className="py-2.5 px-3">Net Qty</th>
                        <th className="py-2.5 px-3">Waste %</th>
                        <th className="py-2.5 px-3">Gross Qty</th>
                        <th className="py-2.5 px-3">Unit</th>
                        <th className="py-2.5 px-3">Mat Rate</th>
                        <th className="py-2.5 px-3">Lab Rate</th>
                        <th className="py-2.5 px-3 text-right">Total Cost</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {report.rooms.flatMap((r) => r.items).map((item) => (
                        <tr key={item.item_id} className="hover:bg-surface/40 transition-colors">
                          <td className="py-2.5 px-3 font-mono font-medium text-accent">{item.csi_code}</td>
                          <td className="py-2.5 px-3 font-medium text-fg max-w-xs truncate">{item.description}</td>
                          <td className="py-2.5 px-3 text-muted-fg">{item.room_name || "General"}</td>
                          <td className="py-2.5 px-3 text-fg">{item.quantity_net.toFixed(2)}</td>
                          <td className="py-2.5 px-3 text-muted-fg">+{item.waste_percent.toFixed(0)}%</td>
                          <td className="py-2.5 px-3 font-medium text-fg">{item.quantity_gross.toFixed(2)}</td>
                          <td className="py-2.5 px-3 text-muted-fg">{item.unit}</td>
                          <td className="py-2.5 px-3 text-muted-fg">{formatCurrency(item.unit_cost_material, report.currency)}</td>
                          <td className="py-2.5 px-3 text-muted-fg">{formatCurrency(item.unit_cost_labor, report.currency)}</td>
                          <td className="py-2.5 px-3 font-bold text-fg text-right">{formatCurrency(item.total_cost, report.currency)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* TAB 4: UNIT RATES & WASTE OVERRIDE CUSTOMIZER */}
              {activeTab === "customizer" && (
                <div className="space-y-4 max-w-2xl">
                  <div className="p-4 rounded-xl border border-border bg-surface/30">
                    <h3 className="text-xs font-bold text-fg mb-1">Interactive Unit Cost Overrides</h3>
                    <p className="text-[11px] text-muted-fg mb-4">
                      Modify material unit costs, labor rates, and waste factors. Instant calculation applies net deductions with updated financial metrics.
                    </p>

                    <div className="space-y-3">
                      {Object.entries(rateOverrides).map(([code, rate]) => (
                        <div key={code} className="p-3 rounded-lg border border-border/70 bg-panel flex items-center justify-between gap-4">
                          <div>
                            <span className="font-mono text-xs font-bold text-accent">{code}</span>
                            <span className="text-xs text-fg ml-2 font-medium">
                              {code === "09 29 00" && "Gypsum Drywall Assemblies"}
                              {code === "06 11 00" && "Rough Wood / Metal Framing"}
                              {code === "09 64 00" && "Solid Hardwood Flooring"}
                              {code === "09 30 13" && "Ceramic & Porcelain Tile"}
                              {code === "06 20 00" && "Architectural Baseboard Millwork"}
                            </span>
                          </div>

                          <div className="flex items-center gap-3">
                            <div>
                              <label className="text-[10px] text-muted-fg block">Mat ($/u)</label>
                              <input
                                type="number"
                                value={rate.material_cost ?? 0}
                                onChange={(e) =>
                                  setRateOverrides((prev) => ({
                                    ...prev,
                                    [code]: { ...prev[code], material_cost: parseFloat(e.target.value) || 0 },
                                  }))
                                }
                                className="w-20 px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                              />
                            </div>

                            <div>
                              <label className="text-[10px] text-muted-fg block">Lab ($/u)</label>
                              <input
                                type="number"
                                value={rate.labor_cost ?? 0}
                                onChange={(e) =>
                                  setRateOverrides((prev) => ({
                                    ...prev,
                                    [code]: { ...prev[code], labor_cost: parseFloat(e.target.value) || 0 },
                                  }))
                                }
                                className="w-20 px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                              />
                            </div>

                            <div>
                              <label className="text-[10px] text-muted-fg block">Waste %</label>
                              <input
                                type="number"
                                value={rate.waste_percent ?? 0}
                                onChange={(e) =>
                                  setRateOverrides((prev) => ({
                                    ...prev,
                                    [code]: { ...prev[code], waste_percent: parseFloat(e.target.value) || 0 },
                                  }))
                                }
                                className="w-16 px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border">
                      <button
                        onClick={handleRecalculateEstimate}
                        disabled={isLoading}
                        className="px-4 py-2 text-xs font-semibold rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
                        Recalculate Estimate
                      </button>

                      <button
                        onClick={handleResetDefaults}
                        disabled={isLoading}
                        className="px-4 py-2 text-xs font-medium rounded-lg border border-border bg-surface hover:bg-surface/80 text-fg transition-colors"
                      >
                        Reset to Defaults
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
