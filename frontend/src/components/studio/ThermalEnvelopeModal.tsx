"use client";

import React, { useState, useEffect } from "react";
import {
  Flame,
  Download,
  Layers,
  Compass,
  Sliders,
  X,
  RefreshCw,
  Sun,
} from "lucide-react";
import {
  type ThermalEnvelopeReport,
  type ClimateZone,
  CLIMATE_ZONE_LABELS,
  formatThermalStatus,
  formatUValue,
  formatRValue,
} from "@/lib/studio/thermal-envelope-model";
import {
  fetchThermalEnvelopeReport,
  evaluateCustomThermalScenario,
  downloadThermalCertificate,
} from "@/lib/api/thermal-envelope";

export interface ThermalEnvelopeModalProps {
  worldId: string;
  projectName?: string;
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

type ModalTab = "assemblies" | "glazing" | "parameters";

export function ThermalEnvelopeModal({
  worldId,
  projectName = "Architectural Reality Model",
  isOpen,
  onClose,
  className = "",
}: ThermalEnvelopeModalProps) {
  const [report, setReport] = useState<ThermalEnvelopeReport | null>(null);
  const [activeTab, setActiveTab] = useState<ModalTab>("assemblies");
  const [climateZone, setClimateZone] = useState<ClimateZone>("5A_cold");
  const [designDeltaT, setDesignDeltaT] = useState<number>(25.0);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Custom U-factor overrides
  const [customUOverrides, setCustomUOverrides] = useState<Record<string, number>>({
    wall_u: 0.30,
    roof_u: 0.16,
    fenestration_u: 1.90,
  });

  useEffect(() => {
    if (!isOpen || !worldId) return;
    let active = true;

    fetchThermalEnvelopeReport(worldId, {
      climateZone,
      designDeltaTK: designDeltaT,
    })
      .then((data) => {
        if (active) {
          setReport(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        console.warn("Failed to load thermal envelope report:", err);
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [isOpen, worldId, climateZone, designDeltaT]);

  if (!isOpen) return null;

  const handleDownloadCertificate = async () => {
    setIsExporting(true);
    setStatusMessage("Compiling ASHRAE 90.1 Certificate PDF...");
    try {
      await downloadThermalCertificate(worldId, {
        climateZone,
      });
      setStatusMessage("Certificate downloaded successfully");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to download certificate");
    } finally {
      setIsExporting(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  const handleReEvaluate = async () => {
    setIsLoading(true);
    setStatusMessage("Re-evaluating building thermal envelope...");
    try {
      const updated = await evaluateCustomThermalScenario(worldId, {
        climateZone,
        designDeltaTK: designDeltaT,
        customUOverrides,
        projectName: report?.project_name || projectName,
      });
      setReport(updated);
      setStatusMessage("Thermal model updated");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to evaluate envelope");
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
              <Flame className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h2 className="text-base font-bold text-fg tracking-tight">
                Building Envelope Thermal Performance
              </h2>
              <p className="text-xs text-muted-fg">
                ANSI/ASHRAE/IES Standard 90.1-2022 & IECC Prescriptive + UA Tradeoff Verification
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
              onClick={handleDownloadCertificate}
              disabled={isExporting || isLoading || !report}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity shadow-sm disabled:opacity-50"
              title="Download formal zero-dependency PDF-1.4 ASHRAE 90.1 Certificate"
            >
              <Download className="w-3.5 h-3.5" />
              Download Certificate
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-surface text-muted-fg hover:text-fg transition-colors ml-2"
              title="Close Thermal Inspection"
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
                Window-to-Wall Ratio
              </span>
              <div className="text-sm font-bold text-accent">
                {report.summary.window_to_wall_ratio_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Cap: {report.summary.max_prescriptive_wwr_pct}% (ASHRAE §5.5.4)
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                UA Budget Tradeoff
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_ua_actual_w_k.toFixed(1)} W/K
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Target: {report.summary.total_ua_target_w_k.toFixed(1)} W/K ({report.summary.ua_tradeoff_margin_pct >= 0 ? "+" : ""}{report.summary.ua_tradeoff_margin_pct.toFixed(1)}%)
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Peak Heat Loss
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_design_heat_loss_kw.toFixed(2)} kW
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                At design ΔT {report.design_delta_t_k} K
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Glazing Area
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_glazing_area_m2.toFixed(1)} m²
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Of {report.summary.gross_exterior_wall_area_m2.toFixed(1)} m² gross wall
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                ASHRAE 90.1 Status
              </span>
              <div
                className={`text-sm font-bold ${
                  report.compliance_verdict === "COMPLIANT" ? "text-success" : "text-danger"
                }`}
              >
                {report.compliance_verdict}
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                {report.summary.surface_count} envelope assemblies
              </div>
            </div>
          </div>
        )}

        {/* Tab Selection Header */}
        <div className="flex items-center gap-1 px-6 pt-3 border-b border-border bg-surface/10">
          <button
            onClick={() => setActiveTab("assemblies")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "assemblies"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Envelope Assemblies ({report?.surfaces.length ?? 0})
          </button>

          <button
            onClick={() => setActiveTab("glazing")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "glazing"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Sun className="w-3.5 h-3.5" />
            Glazing & Façade Distribution
          </button>

          <button
            onClick={() => setActiveTab("parameters")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "parameters"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Climate Zone & U-Factors
          </button>
        </div>

        {/* Tab Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading && !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg gap-3">
              <RefreshCw className="w-6 h-6 animate-spin text-accent" />
              <p className="text-xs">Computing envelope thermal transmittance and UA budget...</p>
            </div>
          ) : !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg">
              <p className="text-xs">No thermal envelope data available for this world model.</p>
            </div>
          ) : (
            <>
              {/* TAB 1: ENVELOPE ASSEMBLIES SCHEDULE */}
              {activeTab === "assemblies" && (
                <div className="overflow-x-auto border border-border rounded-xl bg-surface/20">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-surface/60 border-b border-border text-[11px] uppercase tracking-wider text-muted-fg font-semibold">
                      <tr>
                        <th className="py-2.5 px-3">Assembly Name</th>
                        <th className="py-2.5 px-3">Type</th>
                        <th className="py-2.5 px-3">Orientation</th>
                        <th className="py-2.5 px-3">Surface Area</th>
                        <th className="py-2.5 px-3">Actual U-Factor</th>
                        <th className="py-2.5 px-3">Target U-Factor</th>
                        <th className="py-2.5 px-3">Insulation R-Value</th>
                        <th className="py-2.5 px-3">Design Heat Loss</th>
                        <th className="py-2.5 px-3">Verdict</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60">
                      {report.surfaces.map((s) => {
                        const statusFmt = formatThermalStatus(s.status);
                        return (
                          <tr key={s.surface_id} className="hover:bg-surface/40 transition-colors">
                            <td className="py-2.5 px-3 font-semibold text-fg">{s.name}</td>
                            <td className="py-2.5 px-3 capitalize text-muted-fg">{s.component_type.replace("_", " ")}</td>
                            <td className="py-2.5 px-3 uppercase text-muted-fg font-medium">{s.orientation}</td>
                            <td className="py-2.5 px-3 font-medium text-fg">{s.area_m2.toFixed(2)} m²</td>
                            <td className="py-2.5 px-3 font-mono text-accent">{formatUValue(s.u_factor_w_m2k)}</td>
                            <td className="py-2.5 px-3 font-mono text-muted-fg">{s.target_u_w_m2k.toFixed(3)} W/m²K</td>
                            <td className="py-2.5 px-3 font-mono text-fg">{formatRValue(s.r_value_si)}</td>
                            <td className="py-2.5 px-3 text-muted-fg">{s.heat_loss_watts_at_design_dt.toFixed(0)} W</td>
                            <td className="py-2.5 px-3">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusFmt.colorClass} ${statusFmt.bgClass}`}>
                                {statusFmt.label}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* TAB 2: GLAZING & FAÇADE DISTRIBUTION */}
              {activeTab === "glazing" && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl border border-border bg-surface/30">
                      <div className="flex items-center gap-2 mb-2">
                        <Sun className="w-4 h-4 text-accent" />
                        <h3 className="text-xs font-bold text-fg">Prescriptive Fenestration Cap (ASHRAE §5.5.4)</h3>
                      </div>
                      <p className="text-[11px] text-muted-fg mb-4">
                        ASHRAE 90.1 caps total vertical fenestration at 40.0% of gross wall area. Glazing above 40% requires whole-building performance modeling.
                      </p>

                      <div className="w-full h-3 rounded-full bg-surface border border-border/50 overflow-hidden mb-2">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            report.summary.wwr_prescriptive_compliant ? "bg-accent" : "bg-danger"
                          }`}
                          style={{ width: `${Math.min(100, (report.summary.window_to_wall_ratio_pct / 40.0) * 100)}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-xs pt-1">
                        <span className="text-muted-fg">Measured WWR: <strong className="text-fg">{report.summary.window_to_wall_ratio_pct.toFixed(1)}%</strong></span>
                        <span className="text-muted-fg">Prescriptive Cap: <strong className="text-fg">40.0%</strong></span>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl border border-border bg-surface/30">
                      <div className="flex items-center gap-2 mb-2">
                        <Compass className="w-4 h-4 text-accent" />
                        <h3 className="text-xs font-bold text-fg">Solar Heat Gain Coefficient (SHGC)</h3>
                      </div>
                      <p className="text-[11px] text-muted-fg mb-3">
                        Vertical fenestration SHGC limits minimize cooling loads while admitting daylight.
                      </p>

                      <div className="space-y-2 text-xs">
                        <div className="flex items-center justify-between p-2 rounded-lg border border-border/50 bg-panel">
                          <span className="text-muted-fg">Glazing Assembly SHGC</span>
                          <span className="font-mono font-bold text-accent">0.35</span>
                        </div>
                        <div className="flex items-center justify-between p-2 rounded-lg border border-border/50 bg-panel">
                          <span className="text-muted-fg">ASHRAE {report.climate_zone} Max SHGC</span>
                          <span className="font-mono text-fg">0.40</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: CLIMATE ZONE & PARAMETERS */}
              {activeTab === "parameters" && (
                <div className="space-y-4 max-w-xl">
                  <div className="p-4 rounded-xl border border-border bg-surface/30 space-y-4">
                    <h3 className="text-xs font-bold text-fg">Climate Zone & Thermal Inputs</h3>

                    <div>
                      <label className="text-xs text-muted-fg block mb-1">ASHRAE Climate Zone</label>
                      <select
                        value={climateZone}
                        onChange={(e) => setClimateZone(e.target.value as ClimateZone)}
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-border bg-panel text-fg"
                      >
                        {Object.entries(CLIMATE_ZONE_LABELS).map(([k, label]) => (
                          <option key={k} value={k}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-xs text-muted-fg">Design Temperature Difference (ΔT)</label>
                        <span className="text-xs font-mono font-bold text-accent">{designDeltaT} K</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="45"
                        step="1"
                        value={designDeltaT}
                        onChange={(e) => setDesignDeltaT(parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>

                    <div className="pt-2 border-t border-border space-y-3">
                      <span className="text-xs font-semibold text-fg block">Custom Assembly U-Factors (W/m²K)</span>

                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div>
                          <label className="text-[10px] text-muted-fg block mb-1">Opaque Wall U</label>
                          <input
                            type="number"
                            step="0.01"
                            value={customUOverrides.wall_u ?? 0.30}
                            onChange={(e) =>
                              setCustomUOverrides((prev) => ({ ...prev, wall_u: parseFloat(e.target.value) || 0 }))
                            }
                            className="w-full px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                          />
                        </div>

                        <div>
                          <label className="text-[10px] text-muted-fg block mb-1">Roof Assembly U</label>
                          <input
                            type="number"
                            step="0.01"
                            value={customUOverrides.roof_u ?? 0.16}
                            onChange={(e) =>
                              setCustomUOverrides((prev) => ({ ...prev, roof_u: parseFloat(e.target.value) || 0 }))
                            }
                            className="w-full px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                          />
                        </div>

                        <div>
                          <label className="text-[10px] text-muted-fg block mb-1">Fenestration U</label>
                          <input
                            type="number"
                            step="0.01"
                            value={customUOverrides.fenestration_u ?? 1.90}
                            onChange={(e) =>
                              setCustomUOverrides((prev) => ({ ...prev, fenestration_u: parseFloat(e.target.value) || 0 }))
                            }
                            className="w-full px-2 py-1 text-xs rounded border border-border bg-surface text-fg font-mono"
                          />
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={handleReEvaluate}
                      disabled={isLoading}
                      className="px-4 py-2 text-xs font-semibold rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
                      Re-Evaluate Energy Model
                    </button>
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
