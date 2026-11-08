"use client";

import { useState, useMemo } from "react";
import {
  X,
  HardHat,
  CheckCircle2,
  AlertTriangle,
  Clock,
  HelpCircle,
  Download,
  Layers,
  TrendingUp,
  RefreshCw,
} from "lucide-react";
import {
  verifyPlan,
  downloadConstructionPdf,
  type SiteComplianceReport,
  type Trade,
  type ComplianceStatus,
} from "@/lib/api/construction";
import { useWorlds } from "@/lib/api";
import type { WorldRow } from "@/lib/types";

interface ConstructionProgressModalProps {
  currentWorldId: string;
  isOpen: boolean;
  onClose: () => void;
}

const TRADE_COLORS: Record<Trade, string> = {
  structure: "text-[var(--info)] bg-[var(--info-subtle)] border-[var(--info)]/20",
  architecture: "text-[var(--warning)] bg-[var(--warning-subtle)] border-[var(--warning)]/20",
  mep: "text-[var(--success)] bg-[var(--success-subtle)] border-[var(--success)]/20",
  finishes: "text-[var(--purple)] bg-[var(--purple-subtle)] border-[var(--purple)]/20",
  other: "text-[var(--text-tertiary)] bg-[var(--bg-elevated)] border-[var(--border)]",
};

const STATUS_BADGES: Record<
  ComplianceStatus,
  { label: string; bg: string; text: string; icon: React.ComponentType<{ className?: string }> }
> = {
  completed: {
    label: "COMPLIANT",
    bg: "bg-[var(--success-subtle)] border-[var(--success)]/30",
    text: "text-[var(--success)]",
    icon: CheckCircle2,
  },
  misaligned: {
    label: "MISALIGNED",
    bg: "bg-[var(--warning-subtle)] border-[var(--warning)]/30",
    text: "text-[var(--warning)]",
    icon: AlertTriangle,
  },
  in_progress: {
    label: "IN PROGRESS",
    bg: "bg-[var(--info-subtle)] border-[var(--info)]/30",
    text: "text-[var(--info)]",
    icon: Clock,
  },
  missing: {
    label: "MISSING",
    bg: "bg-[var(--error-subtle)] border-[var(--error)]/30",
    text: "text-[var(--error)]",
    icon: X,
  },
  unplanned: {
    label: "UNPLANNED",
    bg: "bg-[var(--bg-elevated)] border-[var(--border)]",
    text: "text-[var(--text-tertiary)]",
    icon: HelpCircle,
  },
};

export default function ConstructionProgressModal({
  currentWorldId,
  isOpen,
  onClose,
}: ConstructionProgressModalProps) {
  const { data: worldsData } = useWorlds();
  const allWorlds: WorldRow[] = worldsData ?? [];

  const defaultDesignId = useMemo(() => {
    const other = allWorlds.find((w) => w.id !== currentWorldId);
    return other ? other.id : currentWorldId;
  }, [allWorlds, currentWorldId]);

  const [selectedDesignWorldId, setSelectedDesignWorldId] = useState<string | null>(null);
  const designWorldId = selectedDesignWorldId ?? defaultDesignId;
  const [toleranceCm, setToleranceCm] = useState<number>(5);
  const [autoAlign, setAutoAlign] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [downloading, setDownloading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<SiteComplianceReport | null>(null);
  const [tradeFilter, setTradeFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const handleVerify = async () => {
    if (!designWorldId) return;
    setLoading(true);
    setError(null);
    try {
      const rep = await verifyPlan(
        designWorldId,
        currentWorldId,
        toleranceCm / 100.0,
        autoAlign
      );
      setReport(rep);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to run construction verification");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!report) return;
    setDownloading(true);
    try {
      await downloadConstructionPdf(
        designWorldId,
        currentWorldId,
        "Site As-Built Verification",
        toleranceCm / 100.0
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to download PDF report");
    } finally {
      setDownloading(false);
    }
  };

  if (!isOpen) return null;

  const filteredElements = report
    ? report.elements.filter((elem) => {
        if (tradeFilter !== "all" && elem.trade !== tradeFilter) return false;
        if (statusFilter !== "all" && elem.status !== statusFilter) return false;
        return true;
      })
    : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 select-none">
      <div className="flex flex-col w-full max-w-5xl max-h-[92vh] bg-[var(--bg-surface)] border border-[var(--border)] rounded-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent-border)]">
              <HardHat className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-[var(--text-primary)]">
                BIM-vs-Reality Compliance & Progress
              </h2>
              <p className="text-xs text-[var(--text-secondary)]">
                Compare as-built site capture against design baseline with tolerance deviation gating
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[var(--text-tertiary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Configuration Bar */}
        <div className="px-6 py-3 border-b border-[var(--border-subtle)] bg-[var(--bg-base)] flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <label className="text-[var(--text-secondary)] font-medium">Design Baseline:</label>
            <select
              value={designWorldId}
              onChange={(e) => setSelectedDesignWorldId(e.target.value)}
              className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md px-2.5 py-1 text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)]"
            >
              {allWorlds.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name || w.id} {w.id === currentWorldId ? "(Current)" : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-[var(--text-secondary)] font-medium">Tolerance:</label>
            <div className="flex items-center gap-1 bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md px-2 py-1">
              <input
                type="number"
                min="1"
                max="50"
                value={toleranceCm}
                onChange={(e) => setToleranceCm(Math.max(1, Number(e.target.value)))}
                className="w-12 bg-transparent text-right text-[var(--text-primary)] focus:outline-none"
              />
              <span className="text-[var(--text-tertiary)]">cm</span>
            </div>
          </div>

          <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-secondary)] hover:text-[var(--text-primary)] select-none">
            <input
              type="checkbox"
              checked={autoAlign}
              onChange={(e) => setAutoAlign(e.target.checked)}
              className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
            />
            <span>Auto-align static anchors</span>
          </label>

          <button
            onClick={handleVerify}
            disabled={loading}
            className="ml-auto inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-[var(--primary)] text-[var(--on-primary)] font-medium rounded-lg hover:bg-[var(--primary-hover)] transition-colors disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            {loading ? "Verifying..." : "Run Verification"}
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {error && (
            <div className="p-3.5 bg-[var(--error-subtle)] border border-[var(--error)]/20 text-[var(--error)] text-xs rounded-lg flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!report && !loading && (
            <div className="py-16 text-center text-[var(--text-tertiary)]">
              <HardHat className="w-12 h-12 mx-auto mb-3 opacity-30 text-[var(--accent)]" />
              <p className="text-sm font-medium text-[var(--text-primary)]">No verification run yet</p>
              <p className="text-xs max-w-sm mx-auto mt-1 text-[var(--text-secondary)]">
                Select a design BIM baseline model and click &ldquo;Run Verification&rdquo; to analyze as-built compliance and progress.
              </p>
            </div>
          )}

          {report && (
            <>
              {/* KPI Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] flex flex-col justify-between">
                  <div className="flex items-center justify-between text-[var(--text-secondary)] text-xs">
                    <span>Overall Progress</span>
                    <TrendingUp className="w-3.5 h-3.5 text-[var(--success)]" />
                  </div>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-[var(--text-primary)]">
                      {report.summary.overall_completion_pct}%
                    </span>
                    <span className="text-xs text-[var(--text-tertiary)]">complete</span>
                  </div>
                  <div className="mt-2 w-full bg-[var(--bg-base)] rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-[var(--success)] h-full rounded-full transition-all duration-500"
                      style={{ width: `${report.summary.overall_completion_pct}%` }}
                    />
                  </div>
                </div>

                <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] flex flex-col justify-between">
                  <span className="text-[var(--text-secondary)] text-xs">Completed Elements</span>
                  <div className="mt-2 flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-[var(--success)]">
                      {report.summary.completed_count}
                    </span>
                    <span className="text-xs text-[var(--text-tertiary)]">
                      / {report.summary.total_planned} planned
                    </span>
                  </div>
                  <span className="text-[10px] text-[var(--text-tertiary)] mt-2">Within {toleranceCm} cm</span>
                </div>

                <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] flex flex-col justify-between">
                  <span className="text-[var(--text-secondary)] text-xs">Tolerance Deviations</span>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-[var(--warning)]">
                      {report.summary.misaligned_count}
                    </span>
                  </div>
                  <span className="text-[10px] text-[var(--warning)] mt-2">Exceeds {toleranceCm} cm offset</span>
                </div>

                <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] flex flex-col justify-between">
                  <span className="text-[var(--text-secondary)] text-xs">Missing Work</span>
                  <div className="mt-2">
                    <span className="text-2xl font-bold text-[var(--error)]">
                      {report.summary.missing_count}
                    </span>
                  </div>
                  <span className="text-[10px] text-[var(--error)] mt-2">Not observed on site</span>
                </div>
              </div>

              {/* Trade Readiness Grid */}
              <div>
                <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[var(--accent)]" />
                  Trade Breakdown & Readiness
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(report.summary.trades).map(([tradeKey, tSummary]) => (
                    <div
                      key={tradeKey}
                      className="p-3.5 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded border uppercase ${
                            TRADE_COLORS[tradeKey as Trade] || TRADE_COLORS.other
                          }`}
                        >
                          {tradeKey}
                        </span>
                        <span className="text-xs font-bold text-[var(--text-primary)]">
                          {tSummary.completion_pct}%
                        </span>
                      </div>
                      <div className="mt-2.5 text-[11px] text-[var(--text-secondary)] space-y-0.5">
                        <div className="flex justify-between">
                          <span>Completed:</span>
                          <span className="text-[var(--text-primary)] font-medium">{tSummary.completed_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Misaligned:</span>
                          <span className="text-[var(--warning)] font-medium">{tSummary.misaligned_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Missing:</span>
                          <span className="text-[var(--error)] font-medium">{tSummary.missing_count}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Element Compliance Breakdown Table */}
              <div>
                <div className="flex flex-wrap items-center justify-between gap-3 mb-2.5">
                  <h3 className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
                    Element Clashes & Deviation Schedule ({filteredElements.length})
                  </h3>
                  <div className="flex items-center gap-2 text-xs">
                    <select
                      value={tradeFilter}
                      onChange={(e) => setTradeFilter(e.target.value)}
                      className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)]"
                    >
                      <option value="all">All Trades</option>
                      <option value="structure">Structure</option>
                      <option value="architecture">Architecture</option>
                      <option value="mep">MEP</option>
                      <option value="finishes">Finishes</option>
                    </select>
                    <select
                      value={statusFilter}
                      onChange={(e) => setStatusFilter(e.target.value)}
                      className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)]"
                    >
                      <option value="all">All Statuses</option>
                      <option value="misaligned">Misaligned Only</option>
                      <option value="missing">Missing Only</option>
                      <option value="completed">Compliant Only</option>
                      <option value="in_progress">In Progress</option>
                      <option value="unplanned">Unplanned</option>
                    </select>
                  </div>
                </div>

                <div className="border border-[var(--border)] rounded-lg overflow-hidden bg-[var(--bg-surface)]">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[var(--bg-elevated)] border-b border-[var(--border)] text-[var(--text-secondary)] font-medium">
                        <tr>
                          <th className="py-2.5 px-3">Element</th>
                          <th className="py-2.5 px-3">Trade</th>
                          <th className="py-2.5 px-3">Status</th>
                          <th className="py-2.5 px-3 text-right">Deviation</th>
                          <th className="py-2.5 px-3">Notes</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[var(--border-subtle)]">
                        {filteredElements.length === 0 ? (
                          <tr>
                            <td colSpan={5} className="py-6 text-center text-[var(--text-tertiary)]">
                              No elements match the selected filters.
                            </td>
                          </tr>
                        ) : (
                          filteredElements.map((elem) => {
                            const badge = STATUS_BADGES[elem.status] || STATUS_BADGES.unplanned;
                            const IconComponent = badge.icon;
                            return (
                              <tr key={elem.element_id} className="hover:bg-[var(--bg-hover)] transition-colors">
                                <td className="py-2.5 px-3 font-medium text-[var(--text-primary)]">
                                  {elem.name}
                                  <div className="text-[10px] text-[var(--text-tertiary)] font-mono">
                                    {elem.element_id}
                                  </div>
                                </td>
                                <td className="py-2.5 px-3">
                                  <span
                                    className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border uppercase ${
                                      TRADE_COLORS[elem.trade] || TRADE_COLORS.other
                                    }`}
                                  >
                                    {elem.trade}
                                  </span>
                                </td>
                                <td className="py-2.5 px-3">
                                  <span
                                    className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full border ${badge.bg} ${badge.text}`}
                                  >
                                    <IconComponent className="w-3 h-3" />
                                    {badge.label}
                                  </span>
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono">
                                  {elem.observed_pos ? (
                                    <span
                                      className={
                                        elem.deviation_m > elem.tolerance_m
                                          ? "text-[var(--warning)] font-semibold"
                                          : "text-[var(--text-primary)]"
                                      }
                                    >
                                      {(elem.deviation_m * 100).toFixed(1)} cm
                                    </span>
                                  ) : (
                                    <span className="text-[var(--text-tertiary)]">—</span>
                                  )}
                                </td>
                                <td className="py-2.5 px-3 text-[var(--text-tertiary)] text-[11px]">
                                  {elem.notes}
                                </td>
                              </tr>
                            );
                          })
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-[var(--border)] bg-[var(--bg-elevated)] flex items-center justify-between text-xs">
          <div className="text-[var(--text-tertiary)]">
            {report
              ? `Inspected on ${report.summary.inspection_date.slice(0, 10)} | Residual: ${(
                  report.summary.alignment_residual_m * 100
                ).toFixed(1)} cm`
              : "Ready"}
          </div>
          <div className="flex items-center gap-2">
            {report && (
              <button
                onClick={handleDownloadPdf}
                disabled={downloading}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[var(--bg-elevated)] text-[var(--text-primary)] font-medium rounded-lg hover:bg-[var(--bg-hover)] border border-[var(--border)] transition-colors disabled:opacity-50 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-[var(--accent)]" />
                {downloading ? "Generating PDF..." : "Export Progress PDF"}
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-1.5 bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] font-medium rounded-lg hover:bg-[var(--bg-hover)] border border-[var(--border)] transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
