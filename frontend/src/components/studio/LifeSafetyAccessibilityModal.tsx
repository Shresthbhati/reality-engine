"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldCheck,
  Download,
  Footprints,
  RotateCw,
  Sliders,
  X,
  RefreshCw,
} from "lucide-react";
import {
  type LifeSafetyReport,
  formatComplianceStatus,
  formatDistance,
} from "@/lib/studio/life-safety-model";
import {
  fetchLifeSafetyReport,
  runCustomLifeSafetyAudit,
  downloadLifeSafetyCertificate,
} from "@/lib/api/life-safety";

export interface LifeSafetyAccessibilityModalProps {
  worldId: string;
  projectName?: string;
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

type ModalTab = "checks" | "egress" | "turning" | "parameters";

export function LifeSafetyAccessibilityModal({
  worldId,
  projectName = "Architectural Reality Model",
  isOpen,
  onClose,
  className = "",
}: LifeSafetyAccessibilityModalProps) {
  const [report, setReport] = useState<LifeSafetyReport | null>(null);
  const [activeTab, setActiveTab] = useState<ModalTab>("checks");
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Audit parameters state
  const [occupancyGroup, setOccupancyGroup] = useState<string>("Business (Group B)");
  const [isSprinklered, setIsSprinklered] = useState<boolean>(true);

  useEffect(() => {
    if (!isOpen || !worldId) return;
    let active = true;

    fetchLifeSafetyReport(worldId, {
      occupancyGroup,
      isSprinklered,
    })
      .then((data) => {
        if (active) {
          setReport(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        console.warn("Failed to load life safety report:", err);
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [isOpen, worldId, occupancyGroup, isSprinklered]);

  if (!isOpen) return null;

  const handleDownloadCertificate = async () => {
    setIsExporting(true);
    setStatusMessage("Compiling Life Safety Certificate PDF...");
    try {
      await downloadLifeSafetyCertificate(worldId, {
        occupancyGroup,
        isSprinklered,
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

  const handleReAudit = async () => {
    setIsLoading(true);
    setStatusMessage("Re-evaluating code compliance...");
    try {
      const updated = await runCustomLifeSafetyAudit(worldId, {
        occupancyGroup,
        isSprinklered,
        projectName: report?.project_name || projectName,
      });
      setReport(updated);
      setStatusMessage("Audit metrics updated");
    } catch (err) {
      console.error(err);
      setStatusMessage("Failed to re-audit model");
    } finally {
      setIsLoading(false);
      setTimeout(() => setStatusMessage(null), 3000);
    }
  };

  const filteredChecks = (report?.checks ?? []).filter((c) => {
    if (filterCategory === "all") return true;
    if (filterCategory === "failed") return c.status === "fail";
    return c.category === filterCategory;
  });

  return (
    <div className={`fixed inset-0 z-50 bg-base/80 backdrop-blur-sm flex items-center justify-center p-4 ${className}`}>
      <div className="bg-panel border border-border rounded-xl shadow-2xl w-full max-w-5xl h-[88vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-accent/10 border border-accent/20">
              <ShieldCheck className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h2 className="text-base font-bold text-fg tracking-tight">
                Life Safety & ADA Accessibility Compliance
              </h2>
              <p className="text-xs text-muted-fg">
                ADA 2010 Standards (§404, §403, §304) & IBC Chapter 10 Means of Egress
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
              title="Download formal zero-dependency PDF-1.4 Life Safety Certificate"
            >
              <Download className="w-3.5 h-3.5" />
              Download Certificate
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-surface text-muted-fg hover:text-fg transition-colors ml-2"
              title="Close Compliance Audit"
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
                Compliance Score
              </span>
              <div className="text-sm font-bold text-accent">
                {report.summary.compliance_score_pct.toFixed(1)}%
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Overall: {report.summary.overall_status.toUpperCase()}
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Audit Findings
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.passed_checks} Pass / {report.summary.failed_checks} Fail
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                {report.summary.warning_checks} items flagged
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Max Travel Distance
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.max_egress_distance_m.toFixed(1)} m
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Limit: {report.is_sprinklered ? "76.2m" : "60.96m"}
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Occupant Load
              </span>
              <div className="text-sm font-bold text-fg">
                {report.total_occupant_load} Persons
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                IBC Table 1004.5 density
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-border/60 bg-panel/70">
              <span className="text-[10px] uppercase font-semibold text-muted-fg tracking-wider block mb-0.5">
                Egress Doors
              </span>
              <div className="text-sm font-bold text-fg">
                {report.summary.total_doors_audited} Doorways
              </div>
              <div className="text-[10px] text-muted-fg mt-0.5">
                Across {report.summary.total_rooms_audited} spaces
              </div>
            </div>
          </div>
        )}

        {/* Tab Selection Header */}
        <div className="flex items-center gap-1 px-6 pt-3 border-b border-border bg-surface/10">
          <button
            onClick={() => setActiveTab("checks")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "checks"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            Code Checks ({report?.checks.length ?? 0})
          </button>

          <button
            onClick={() => setActiveTab("egress")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "egress"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <Footprints className="w-3.5 h-3.5" />
            Means of Egress Paths ({report?.egress_routes.length ?? 0})
          </button>

          <button
            onClick={() => setActiveTab("turning")}
            className={`px-3 py-2 text-xs font-medium border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "turning"
                ? "border-accent text-accent font-semibold"
                : "border-transparent text-muted-fg hover:text-fg"
            }`}
          >
            <RotateCw className="w-3.5 h-3.5" />
            Wheelchair Turning Spaces (ADA 60-inch)
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
            Audit Parameters & Occupancy
          </button>
        </div>

        {/* Tab Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading && !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg gap-3">
              <RefreshCw className="w-6 h-6 animate-spin text-accent" />
              <p className="text-xs">Auditing ADA clearances and egress travel distance...</p>
            </div>
          ) : !report ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-fg">
              <p className="text-xs">No life safety audit available for this world model.</p>
            </div>
          ) : (
            <>
              {/* TAB 1: CODE CHECKS MATRIX */}
              {activeTab === "checks" && (
                <div className="space-y-4">
                  {/* Category Filter Pills */}
                  <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
                    {[
                      { id: "all", label: "All Checks" },
                      { id: "failed", label: "Failed Only" },
                      { id: "door_clearance", label: "Door Width (32\")" },
                      { id: "corridor_width", label: "Corridors" },
                      { id: "turning_space", label: "Turning (60\")" },
                      { id: "egress_distance", label: "Egress Distance" },
                      { id: "door_swing", label: "Door Swings" },
                    ].map((cat) => (
                      <button
                        key={cat.id}
                        onClick={() => setFilterCategory(cat.id)}
                        className={`px-2.5 py-1 rounded-md border text-[11px] transition-colors ${
                          filterCategory === cat.id
                            ? "bg-accent/10 border-accent/40 text-accent font-medium"
                            : "bg-surface border-border/70 text-muted-fg hover:text-fg"
                        }`}
                      >
                        {cat.label}
                      </button>
                    ))}
                  </div>

                  <div className="overflow-x-auto border border-border rounded-xl bg-surface/20">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-surface/60 border-b border-border text-[11px] uppercase tracking-wider text-muted-fg font-semibold">
                        <tr>
                          <th className="py-2.5 px-3">Description</th>
                          <th className="py-2.5 px-3">Standard</th>
                          <th className="py-2.5 px-3">Space / Target</th>
                          <th className="py-2.5 px-3">Measured</th>
                          <th className="py-2.5 px-3">Required</th>
                          <th className="py-2.5 px-3">Status</th>
                          <th className="py-2.5 px-3">Remediation Guidance</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/60">
                        {filteredChecks.map((chk) => {
                          const statusFmt = formatComplianceStatus(chk.status);
                          return (
                            <tr key={chk.item_id} className="hover:bg-surface/40 transition-colors">
                              <td className="py-2.5 px-3 font-semibold text-fg max-w-xs">{chk.title}</td>
                              <td className="py-2.5 px-3 font-mono text-muted-fg">{chk.standard_ref}</td>
                              <td className="py-2.5 px-3 text-muted-fg">{chk.room_name || "General"}</td>
                              <td className="py-2.5 px-3 font-medium text-fg">
                                {chk.measured_value.toFixed(2)} {chk.unit}
                              </td>
                              <td className="py-2.5 px-3 text-muted-fg">
                                {chk.required_value.toFixed(2)} {chk.unit}
                              </td>
                              <td className="py-2.5 px-3">
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusFmt.colorClass} ${statusFmt.bgClass}`}>
                                  {statusFmt.label}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-muted-fg text-[11px] max-w-xs">
                                {chk.remediation_notes || chk.message}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* TAB 2: MEANS OF EGRESS ROUTES */}
              {activeTab === "egress" && (
                <div className="space-y-4">
                  <div className="p-3 rounded-lg border border-border bg-surface/30 text-xs text-muted-fg">
                    IBC §1017: Maximum permitted exit travel distance for {report.occupancy_group} is{" "}
                    <span className="font-semibold text-fg">{report.is_sprinklered ? "76.2m (250 ft)" : "60.96m (200 ft)"}</span> with{" "}
                    {report.is_sprinklered ? "automatic sprinkler suppression" : "unsprinklered construction"}.
                  </div>

                  <div className="overflow-x-auto border border-border rounded-xl bg-surface/20">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-surface/60 border-b border-border text-[11px] uppercase tracking-wider text-muted-fg font-semibold">
                        <tr>
                          <th className="py-2.5 px-3">Space Origin</th>
                          <th className="py-2.5 px-3">Exit Discharge Target</th>
                          <th className="py-2.5 px-3">Egress Distance</th>
                          <th className="py-2.5 px-3">Permitted Limit</th>
                          <th className="py-2.5 px-3">Waypoints</th>
                          <th className="py-2.5 px-3">Status</th>
                          <th className="py-2.5 px-3">Bottlenecks</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/60">
                        {report.egress_routes.map((rt) => {
                          const statusFmt = formatComplianceStatus(rt.status);
                          return (
                            <tr key={rt.route_id} className="hover:bg-surface/40 transition-colors">
                              <td className="py-2.5 px-3 font-semibold text-fg">{rt.origin_room_name}</td>
                              <td className="py-2.5 px-3 font-mono text-muted-fg">{rt.destination_door_id}</td>
                              <td className="py-2.5 px-3 font-medium text-fg">{formatDistance(rt.total_distance_m)}</td>
                              <td className="py-2.5 px-3 text-muted-fg">{formatDistance(rt.max_permitted_distance_m)}</td>
                              <td className="py-2.5 px-3 text-muted-fg">{rt.polyline.length} pts</td>
                              <td className="py-2.5 px-3">
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusFmt.colorClass} ${statusFmt.bgClass}`}>
                                  {statusFmt.label}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-muted-fg text-[11px]">
                                {rt.bottlenecks.length > 0 ? rt.bottlenecks.join(", ") : "Unobstructed"}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* TAB 3: WHEELCHAIR TURNING SPACES */}
              {activeTab === "turning" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {report.turning_spaces.map((ts) => (
                    <div
                      key={ts.room_id}
                      className="p-4 rounded-xl border border-border bg-surface/30 hover:border-accent/40 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-xs text-fg">{ts.room_name}</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            ts.has_clearance
                              ? "text-success bg-success/10 border-success/20"
                              : "text-warn bg-warn/10 border-warn/20"
                          }`}
                        >
                          {ts.has_clearance ? "60\" TURNING OK" : "TIGHT CLEARANCE"}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-border/50">
                        <div>
                          <span className="text-muted-fg text-[11px] block">Inscribed Diameter</span>
                          <span className="font-medium text-fg">{ts.max_inscribed_diameter_m.toFixed(2)} m ({(ts.max_inscribed_diameter_m*39.37).toFixed(0)} in)</span>
                        </div>
                        <div>
                          <span className="text-muted-fg text-[11px] block">ADA Requirement</span>
                          <span className="font-medium text-fg">{ts.required_diameter_m.toFixed(2)} m (60 in)</span>
                        </div>
                      </div>

                      <p className="text-[11px] text-muted-fg mt-3">
                        {ts.has_clearance
                          ? "Accommodates full 360-degree wheelchair turning circle per ADA §304.3.1."
                          : "Does not accommodate circular turning space; verify whether a T-shaped 3-point turning space (ADA §304.3.2) can be provided."}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 4: AUDIT PARAMETERS */}
              {activeTab === "parameters" && (
                <div className="space-y-4 max-w-xl">
                  <div className="p-4 rounded-xl border border-border bg-surface/30 space-y-4">
                    <h3 className="text-xs font-bold text-fg">Code Parameters & Occupancy Assumptions</h3>

                    <div>
                      <label className="text-xs text-muted-fg block mb-1">IBC Occupancy Classification</label>
                      <select
                        value={occupancyGroup}
                        onChange={(e) => setOccupancyGroup(e.target.value)}
                        className="w-full px-3 py-1.5 text-xs rounded-lg border border-border bg-panel text-fg"
                      >
                        <option value="Business (Group B)">Business (Group B) — Offices, civic, banks</option>
                        <option value="Assembly (Group A)">Assembly (Group A) — Auditoriums, restaurants</option>
                        <option value="Mercantile (Group M)">Mercantile (Group M) — Retail, stores</option>
                        <option value="Residential (Group R)">Residential (Group R) — Apartments, dwellings</option>
                        <option value="Educational (Group E)">Educational (Group E) — Schools, daycares</option>
                        <option value="Storage (Group S)">Storage (Group S) — Warehouses</option>
                      </select>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-lg border border-border/60 bg-panel">
                      <div>
                        <span className="text-xs font-semibold text-fg block">Automatic Fire Sprinkler System</span>
                        <span className="text-[11px] text-muted-fg">IBC §903: Extends max travel distance from 200 ft to 250 ft</span>
                      </div>
                      <input
                        type="checkbox"
                        checked={isSprinklered}
                        onChange={(e) => setIsSprinklered(e.target.checked)}
                        className="w-4 h-4 rounded border-border text-accent focus:ring-accent"
                      >
                      </input>
                    </div>

                    <button
                      onClick={handleReAudit}
                      disabled={isLoading}
                      className="px-4 py-2 text-xs font-semibold rounded-lg bg-accent text-accent-fg hover:opacity-90 flex items-center gap-1.5 transition-opacity"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
                      Re-Audit Code Compliance
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
