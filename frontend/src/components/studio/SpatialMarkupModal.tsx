"use client";

import React, { useState } from "react";
import {
  Download,
  Filter,
  Layers,
  MapPin,
  Navigation,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import {
  type MarkupStatus,
  type MarkupTrade,
  type SpatialMarkup,
  formatMarkupCoord,
  formatRelativeTimestamp,
  getStatusLabel,
  getTradeColorHex,
  getTradeLabel,
} from "@/lib/studio/collab-presence-model";
import { downloadCollabBcf } from "@/lib/api/collab";

export interface SpatialMarkupModalProps {
  worldId: string;
  markups: SpatialMarkup[];
  isOpen: boolean;
  onClose: () => void;
  onCreateMarkup: (payload: {
    title: string;
    description: string;
    trade: MarkupTrade;
    status: MarkupStatus;
    position: [number, number, number];
  }) => Promise<void>;
  onDeleteMarkup: (markupId: string) => Promise<void>;
  onFlyToMarkup?: (markup: SpatialMarkup) => void;
  defaultPosition?: [number, number, number];
}

export function SpatialMarkupModal({
  worldId,
  markups,
  isOpen,
  onClose,
  onCreateMarkup,
  onDeleteMarkup,
  onFlyToMarkup,
  defaultPosition = [0.0, 0.0, 1.5],
}: SpatialMarkupModalProps) {
  const [selectedTrade, setSelectedTrade] = useState<MarkupTrade | "all">("all");
  const [selectedStatus, setSelectedStatus] = useState<MarkupStatus | "all">("all");
  const [isExportingBcf, setIsExportingBcf] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);

  // New markup form state
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newTrade, setNewTrade] = useState<MarkupTrade>("architecture");
  const [newStatus, setNewStatus] = useState<MarkupStatus>("open");
  const [posX, setPosX] = useState(defaultPosition[0].toFixed(2));
  const [posY, setPosY] = useState(defaultPosition[1].toFixed(2));
  const [posZ, setPosZ] = useState(defaultPosition[2].toFixed(2));

  if (!isOpen) return null;

  const filtered = markups.filter((m) => {
    if (selectedTrade !== "all" && m.trade !== selectedTrade) return false;
    if (selectedStatus !== "all" && m.status !== selectedStatus) return false;
    return true;
  });

  const handleExportBcf = async () => {
    try {
      setIsExportingBcf(true);
      await downloadCollabBcf(worldId);
    } catch (err) {
      console.error("Failed to export BCF:", err);
    } finally {
      setIsExportingBcf(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;

    try {
      setIsCreating(true);
      await onCreateMarkup({
        title: newTitle.trim(),
        description: newDescription.trim(),
        trade: newTrade,
        status: newStatus,
        position: [parseFloat(posX) || 0, parseFloat(posY) || 0, parseFloat(posZ) || 0],
      });
      setNewTitle("");
      setNewDescription("");
      setShowAddForm(false);
    } catch (err) {
      console.error("Failed to create spatial markup:", err);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Spatial Markups & Telepresence Issues"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-studio-surface/80 backdrop-blur-sm pointer-events-auto"
    >
      <div className="flex flex-col w-full max-w-2xl max-h-[85vh] rounded-xl border border-studio-border bg-studio-surface text-studio-text shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-studio-border/80 bg-studio-panel/50">
          <div className="flex items-center gap-2.5">
            <MapPin className="w-5 h-5 text-studio-accent" />
            <div>
              <h2 className="text-sm font-semibold text-studio-text tracking-wide">
                Spatial Markups & BCF Issues
              </h2>
              <p className="text-xs text-studio-muted">
                {markups.length} recorded annotations across architectural disciplines
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleExportBcf}
              disabled={isExportingBcf || markups.length === 0}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-studio-border bg-studio-panel hover:bg-studio-border/30 text-studio-text transition-colors disabled:opacity-50"
              title="Download standard buildingSMART BCF 2.1 zip package"
            >
              {isExportingBcf ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Download className="w-3.5 h-3.5 text-studio-muted" />
              )}
              <span>Export BCF 2.1</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded-md text-studio-muted hover:text-studio-text hover:bg-studio-border/20 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Filters and Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-b border-studio-border/60 bg-studio-surface">
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 text-xs text-studio-muted">
              <Filter className="w-3.5 h-3.5" />
              <span>Trade:</span>
            </div>
            {(["all", "architecture", "structure", "mep", "safety", "general"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setSelectedTrade(t)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                  selectedTrade === t
                    ? "bg-studio-panel text-studio-text border-studio-accent"
                    : "border-transparent text-studio-muted hover:text-studio-text"
                }`}
              >
                {t === "all" ? "All" : getTradeLabel(t)}
              </button>
            ))}

            <div className="flex items-center gap-1 text-xs text-studio-muted ml-1 pl-2 border-l border-studio-border/60">
              <span>Status:</span>
            </div>
            {(["all", "open", "in_review", "resolved"] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSelectedStatus(s)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                  selectedStatus === s
                    ? "bg-studio-panel text-studio-text border-studio-accent"
                    : "border-transparent text-studio-muted hover:text-studio-text"
                }`}
              >
                {s === "all" ? "All" : getStatusLabel(s)}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setShowAddForm(!showAddForm)}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium bg-studio-accent text-studio-surface hover:opacity-90 transition-opacity"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>{showAddForm ? "Cancel" : "Add Issue / Note"}</span>
          </button>
        </div>

        {/* Add Form (Collapsible) */}
        {showAddForm && (
          <form
            onSubmit={handleCreate}
            className="p-4 border-b border-studio-border/80 bg-studio-panel/30 space-y-3"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-studio-text mb-1">
                  Title
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Beam Interference with HVAC Duct"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-md text-xs border border-studio-border bg-studio-surface text-studio-text focus:outline-none focus:border-studio-accent"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs font-medium text-studio-text mb-1">
                    Trade
                  </label>
                  <select
                    value={newTrade}
                    onChange={(e) => setNewTrade(e.target.value as MarkupTrade)}
                    className="w-full px-2 py-1.5 rounded-md text-xs border border-studio-border bg-studio-surface text-studio-text focus:outline-none focus:border-studio-accent"
                  >
                    <option value="architecture">Architecture</option>
                    <option value="structure">Structure</option>
                    <option value="mep">MEP / Services</option>
                    <option value="safety">Safety & Code</option>
                    <option value="general">General</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-studio-text mb-1">
                    Status
                  </label>
                  <select
                    value={newStatus}
                    onChange={(e) => setNewStatus(e.target.value as MarkupStatus)}
                    className="w-full px-2 py-1.5 rounded-md text-xs border border-studio-border bg-studio-surface text-studio-text focus:outline-none focus:border-studio-accent"
                  >
                    <option value="open">Open</option>
                    <option value="in_review">In Review</option>
                    <option value="resolved">Resolved</option>
                  </select>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-studio-text mb-1">
                Description / Notes
              </label>
              <textarea
                rows={2}
                placeholder="Specific clash details, elevation constraints, or code references..."
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded-md text-xs border border-studio-border bg-studio-surface text-studio-text focus:outline-none focus:border-studio-accent"
              />
            </div>

            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-2 text-xs text-studio-muted">
                <span>Position (X, Y, Z):</span>
                <input
                  type="number"
                  step="0.1"
                  value={posX}
                  onChange={(e) => setPosX(e.target.value)}
                  className="w-14 px-1.5 py-0.5 rounded border border-studio-border bg-studio-surface text-studio-text text-center text-xs font-mono"
                />
                <input
                  type="number"
                  step="0.1"
                  value={posY}
                  onChange={(e) => setPosY(e.target.value)}
                  className="w-14 px-1.5 py-0.5 rounded border border-studio-border bg-studio-surface text-studio-text text-center text-xs font-mono"
                />
                <input
                  type="number"
                  step="0.1"
                  value={posZ}
                  onChange={(e) => setPosZ(e.target.value)}
                  className="w-14 px-1.5 py-0.5 rounded border border-studio-border bg-studio-surface text-studio-text text-center text-xs font-mono"
                />
              </div>

              <button
                type="submit"
                disabled={isCreating || !newTitle.trim()}
                className="px-4 py-1.5 rounded-md text-xs font-medium bg-studio-accent text-studio-surface hover:opacity-90 disabled:opacity-50"
              >
                {isCreating ? "Saving..." : "Save Markup"}
              </button>
            </div>
          </form>
        )}

        {/* Markups List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {filtered.length === 0 ? (
            <div className="py-12 text-center text-studio-muted">
              <Layers className="w-8 h-8 mx-auto mb-2 opacity-40" />
              <p className="text-xs">No spatial markups found in this view.</p>
              <p className="text-[11px] mt-1 opacity-75">
                Drop a pin in the 3D model or add a note using the button above.
              </p>
            </div>
          ) : (
            filtered.map((item) => {
              const tradeHex = getTradeColorHex(item.trade);

              return (
                <div
                  key={item.markup_id}
                  className="flex items-start justify-between gap-3 p-3 rounded-lg border border-studio-border/70 bg-studio-panel/40 hover:bg-studio-panel/70 transition-colors"
                >
                  <div className="flex items-start gap-2.5">
                    {/* Trade Icon Pin */}
                    <div
                      className="mt-0.5 w-6 h-6 rounded-full flex items-center justify-center shrink-0"
                      style={{ backgroundColor: `${tradeHex}20` }}
                    >
                      <MapPin className="w-3.5 h-3.5" style={{ color: tradeHex }} />
                    </div>

                    {/* Details */}
                    <div>
                      <div className="flex flex-wrap items-center gap-1.5 mb-1">
                        <span className="text-xs font-semibold text-studio-text">
                          {item.title}
                        </span>
                        <span
                          className="px-1.5 py-0.2 rounded text-[10px] font-medium border"
                          style={{
                            borderColor: `${tradeHex}50`,
                            backgroundColor: `${tradeHex}15`,
                            color: tradeHex,
                          }}
                        >
                          {getTradeLabel(item.trade)}
                        </span>
                        <span className="px-1.5 py-0.2 rounded text-[10px] font-mono border border-studio-border bg-studio-surface text-studio-muted">
                          {getStatusLabel(item.status)}
                        </span>
                      </div>

                      {item.description && (
                        <p className="text-xs text-studio-muted leading-relaxed mb-1.5">
                          {item.description}
                        </p>
                      )}

                      <div className="flex flex-wrap items-center gap-3 text-[11px] text-studio-muted font-mono">
                        <span>{formatMarkupCoord(item.position)}</span>
                        <span>by {item.author_name}</span>
                        <span>{formatRelativeTimestamp(item.created_at)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 shrink-0">
                    {onFlyToMarkup && (
                      <button
                        type="button"
                        onClick={() => onFlyToMarkup(item)}
                        className="p-1.5 rounded-md text-studio-muted hover:text-studio-text hover:bg-studio-border/30 transition-colors"
                        title="Fly camera to markup location"
                      >
                        <Navigation className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => onDeleteMarkup(item.markup_id)}
                      className="p-1.5 rounded-md text-studio-muted hover:text-studio-accent hover:bg-studio-border/30 transition-colors"
                      title="Delete markup"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
