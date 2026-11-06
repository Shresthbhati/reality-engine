"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Ruler, Trash2, X } from "lucide-react";
import type { WorldSceneController } from "@/lib/viewport/three-scene";
import type { WorldIR } from "@/types/worldir";
import {
  KIND_LABEL,
  MEASURE_TOOLS,
  computePreview,
  formatUncertainty,
  formatValue,
  histogram,
  provenanceSummary,
  resolveUnitContext,
  stateFor,
  STATE_HELP,
  type DimensionState,
  unitFor,
  unitLabelFromRow,
  type MeasureKind,
  type Vec3,
} from "@/lib/viewport/measure-model";
import { setKnownLengthScale } from "@/lib/api/scale";
import {
  createMeasurement,
  deleteMeasurement,
  listMeasurements,
  type SavedMeasurement,
} from "@/lib/api/measurements";

interface Props {
  worldId: string | null;
  world: WorldIR | null;
  getController: () => WorldSceneController | null;
  onClose: () => void;
}

interface Live {
  tool: MeasureKind;
  points: Vec3[];
  done: boolean;
}

function label(m: SavedMeasurement): string {
  return `${KIND_LABEL[m.kind]} ${formatValue(m.value, unitLabelFromRow(m.unit), 2)}`;
}

const STATE_STYLE: Record<DimensionState, string> = {
  MEASURED: "bg-success-subtle border-success/40 text-success",
  RECONSTRUCTED: "bg-info-subtle border-info/40 text-info",
  ESTIMATED: "bg-warning-subtle border-warning/40 text-warning",
  VERIFIED: "bg-purple-subtle border-purple/40 text-purple",
};

function StateBadge({ state }: { state: DimensionState }) {
  return (
    <span
      data-testid="state-badge"
      title={STATE_HELP[state]}
      className={`px-1.5 py-0.5 rounded border text-[9px] font-mono ${STATE_STYLE[state]}`}
    >
      {state}
    </span>
  );
}

export default function MeasurementPanel({ worldId, world, getController, onClose }: Props) {
  const ctx = useMemo(() => resolveUnitContext(world), [world]);
  const [tool, setTool] = useState<MeasureKind>("distance");
  const [snap, setSnap] = useState(true);
  const [annotations, setAnnotations] = useState(true);
  const [live, setLive] = useState<Live | null>(null);
  const [saved, setSaved] = useState<SavedMeasurement[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showSetScale, setShowSetScale] = useState(false);
  const [knownDistM, setKnownDistM] = useState<string>("");
  const [scaleBusy, setScaleBusy] = useState(false);
  const [scaleSuccess, setScaleSuccess] = useState<string | null>(null);


  useEffect(() => {
    const c = getController();
    c?.setMeasurePointsCallback((s) => setLive(s));
    return () => c?.setMeasurePointsCallback(undefined);
  }, [getController]);

  useEffect(() => {
    getController()?.setMeasureTool(tool);
  }, [tool, getController]);

  useEffect(() => {
    getController()?.setMeasureSnap(snap);
  }, [snap, getController]);

  useEffect(() => {
    getController()?.setAnnotationsVisible(annotations);
  }, [annotations, getController]);

  useEffect(() => {
    if (!worldId) return;
    let cancelled = false;
    listMeasurements(worldId)
      .then((rows) => { if (!cancelled) { setSaved(rows); setError(null); } })
      .catch((e: unknown) => { if (!cancelled) setError(`Could not load saved measurements: ${e instanceof Error ? e.message : String(e)}`); });
    return () => { cancelled = true; };
  }, [worldId]);

  useEffect(() => {
    getController()?.setSavedMeasurements(
      saved.map((m) => ({ id: m.id, kind: m.kind, points: m.points, label: label(m) })),
      selectedId,
    );
  }, [saved, selectedId, getController]);

  useEffect(() => () => getController()?.setSavedMeasurements([], null), [getController]);

  const preview = live ? computePreview(live.tool, live.points) : null;
  const complete = live?.done === true && preview !== null;

  const save = useCallback(async () => {
    if (!worldId || !live || !complete || preview === null) return;
    setBusy(true);
    try {
      const row = await createMeasurement(worldId, {
        kind: live.tool,
        points: live.points,
        scale_state: ctx.basis,
        method: ctx.method,
        value: preview,
        uncertainty: ctx.uncertainty,
      });
      setSaved((s) => [...s, row]);
      setSelectedId(row.id);
      setError(null);
      getController()?.clearMeasurement();
    } catch (e) {
      setError(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }, [worldId, live, complete, preview, ctx, getController]);

  const remove = useCallback(async (id: string) => {
    if (!worldId) return;
    try {
      await deleteMeasurement(worldId, id);
      setSaved((s) => s.filter((m) => m.id !== id));
      setSelectedId((cur) => (cur === id ? null : cur));
    } catch (e) {
      setError(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [worldId]);

  const selected = saved.find((m) => m.id === selectedId) ?? null;
  const profile = useMemo(() => {
    const cls = selected ? saved.filter((m) => m.unit === selected.unit) : [];
    return { unit: selected ? unitLabelFromRow(selected.unit) : "", bins: histogram(cls.map((m) => m.value)) };
  }, [saved, selected]);
  const maxCount = Math.max(1, ...profile.bins.map((b) => b.count));

  const handleSetScale = useCallback(async () => {
    if (!worldId) return;
    const dist = parseFloat(knownDistM);
    if (!Number.isFinite(dist) || dist <= 0) {
      setError("Please enter a valid positive physical length in metres.");
      return;
    }
    setScaleBusy(true);
    setError(null);
    setScaleSuccess(null);
    try {
      const modelLength = (live && preview !== null) ? preview : (selected ? selected.value : undefined);
      const pointA = (live && live.points.length >= 2) ? live.points[0] : (selected && selected.points?.length >= 2) ? selected.points[0] as Vec3 : undefined;
      const pointB = (live && live.points.length >= 2) ? live.points[1] : (selected && selected.points?.length >= 2) ? selected.points[1] as Vec3 : undefined;

      const payload: { distance_m: number; model_length?: number; point_a?: [number, number, number]; point_b?: [number, number, number] } = {
        distance_m: dist,
        model_length: modelLength,
        point_a: pointA,
        point_b: pointB,
      };

      const res = await setKnownLengthScale(worldId, payload);
      setScaleSuccess(`Scale calibration job queued (${res.job_id}). Scene converting to metric scale.`);
      setShowSetScale(false);
      setKnownDistM("");
    } catch (e) {
      setError(`Scale calibration failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setScaleBusy(false);
    }
  }, [worldId, knownDistM, live, preview, selected]);


  return (
    <div
      className="absolute top-14 left-3 z-20 w-[310px] max-h-[calc(100%-5rem)] overflow-y-auto rounded-lg text-xs pointer-events-auto"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
      aria-label="Measurement mode"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
        <div className="flex items-center gap-2 font-semibold text-[var(--accent)]">
          <Ruler className="w-3.5 h-3.5" />
          <span>Measurement Mode</span>
          <span
            data-testid="scale-calibration-badge"
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
              ctx.scaleCalibrationState === "METRIC_CALIBRATED"
                ? "bg-success-subtle border-success/40 text-success"
                : "bg-warning-subtle border-warning/40 text-warning"
            }`}
            title={
              ctx.scaleCalibrationState === "METRIC_CALIBRATED"
                ? "Calibrated metric scale. Measurements are in physical metres."
                : "Unscaled relative scene. Measurements are in model units."
            }
          >
            {ctx.scaleCalibrationState === "METRIC_CALIBRATED" ? "Metric Calibrated" : "Unscaled Relative"}
          </span>
        </div>
        <button type="button" onClick={onClose} aria-label="Close measurement mode" className="text-fg-muted hover:text-fg cursor-pointer">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-2 grid grid-cols-1 gap-1">
        {MEASURE_TOOLS.map((t) => {
          const disabled = t.unavailable !== null;
          const active = !disabled && t.kind === tool;
          return (
            <button
              key={t.kind}
              type="button"
              disabled={disabled}
              title={disabled ? (t.unavailable ?? undefined) : t.hint}
              onClick={() => !disabled && setTool(t.kind as MeasureKind)}
              className={`text-left px-2 h-7 rounded border transition-colors ${
                active
                  ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-subtle)] cursor-pointer"
                  : disabled
                    ? "border-transparent text-fg-subtle cursor-not-allowed"
                    : "border-transparent hover:text-fg hover:bg-elevated cursor-pointer"
              }`}
            >
              {t.label}
            </button>
          );
        })}
        <label className="flex items-center gap-2 px-2 pt-1 cursor-pointer" title="Snap to the nearest corner or edge of an entity's axis-aligned bounding box">
          <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
          <span>Snap to entity box corners/edges</span>
        </label>
        <label className="flex items-center gap-2 px-2 cursor-pointer">
          <input type="checkbox" checked={annotations} onChange={(e) => setAnnotations(e.target.checked)} />
          <span>Show 3D annotations</span>
        </label>
      </div>

      <div className="px-3 py-2 border-t border-[var(--border)] space-y-1.5">
        <div className="text-[10px] uppercase tracking-wider text-fg-subtle">Current</div>
        {live && live.points.length > 0 ? (
          <>
            <div className="flex items-center gap-2">
              <div className="text-lg font-bold text-fg font-mono">
                {preview !== null ? formatValue(preview, unitFor(live.tool, ctx)) : `${live.points.length} point(s)…`}
              </div>
              {preview !== null && <StateBadge state={ctx.state} />}
            </div>
            <div className="text-[11px]">{formatUncertainty(ctx.uncertainty, ctx.lengthUnit)}</div>
            <div className="flex gap-1.5 pt-1">
              {live.tool === "area" && !live.done && (
                <button type="button" disabled={live.points.length < 3} onClick={() => getController()?.finishMeasurement()} className="px-2 h-6 rounded border border-[var(--border)] hover:text-fg disabled:opacity-40 cursor-pointer">
                  Finish
                </button>
              )}
              {!live.done && (
                <button type="button" onClick={() => getController()?.undoMeasurementPoint()} className="px-2 h-6 rounded border border-[var(--border)] hover:text-fg cursor-pointer">
                  Undo
                </button>
              )}
              <button type="button" disabled={!complete || busy || !worldId} onClick={save} className="px-2 h-6 rounded bg-[var(--accent)] text-on-accent font-semibold disabled:opacity-40 cursor-pointer">
                Save
              </button>
              <button type="button" onClick={() => getController()?.clearMeasurement()} className="px-2 h-6 rounded border border-[var(--border)] hover:text-fg cursor-pointer">
                Reset
              </button>
            </div>
          </>
        ) : (
          <div className="text-fg-subtle">{MEASURE_TOOLS.find((t) => t.kind === tool)?.hint}</div>
        )}
        {error && <div className="text-error" role="alert">{error}</div>}
      </div>

      <div className="px-3 py-2 border-t border-[var(--border)] space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-wider text-fg-subtle">Scale Calibration</span>
          <button
            type="button"
            data-testid="set-scale-toggle"
            onClick={() => { setShowSetScale((v) => !v); setError(null); }}
            className="text-[10px] font-medium text-[var(--accent)] hover:underline cursor-pointer"
          >
            {showSetScale ? "Cancel" : "Set scale"}
          </button>
        </div>
        {showSetScale && (
          <div className="rounded border border-[var(--border)] p-2 space-y-2 bg-[var(--bg-base)]" data-testid="set-scale-panel">
            <div className="text-[11px] text-fg-muted">
              Calibrate relative scale to metric by entering the physical ground-truth length for the current or selected distance:
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-fg-muted">Length (m):</span>
              <input
                type="number"
                step="0.01"
                min="0.01"
                value={knownDistM}
                onChange={(e) => setKnownDistM(e.target.value)}
                placeholder="e.g. 2.05"
                data-testid="known-distance-input"
                className="w-24 px-2 py-0.5 rounded border border-[var(--border)] bg-[var(--bg-surface)] text-fg text-xs font-mono"
              />
              <button
                type="button"
                disabled={scaleBusy || !knownDistM || !worldId}
                onClick={handleSetScale}
                data-testid="apply-scale-button"
                className="px-2 py-1 rounded bg-[var(--accent)] text-on-accent font-semibold text-xs disabled:opacity-40 cursor-pointer"
              >
                {scaleBusy ? "Applying..." : "Calibrate"}
              </button>
            </div>
          </div>
        )}
        {scaleSuccess && (
          <div className="text-success text-[11px]" role="status">{scaleSuccess}</div>
        )}
      </div>

      {selected && (
        <div className="px-3 py-2 border-t border-[var(--border)] space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-fg-subtle">Measurement Details</div>
          <div className="text-fg font-semibold">{KIND_LABEL[selected.kind]}</div>
          <div className="flex justify-between gap-3 items-center">
            <span className="text-fg-subtle shrink-0">State</span>
            <StateBadge state={stateFor(selected)} />
          </div>
          <Row k="Estimated Value" v={formatValue(selected.value, unitLabelFromRow(selected.unit))} />
          {selected.verified_value != null && (
            <Row k="Tape-verified" v={formatValue(selected.verified_value, unitLabelFromRow(selected.unit))} />
          )}
          <Row k="Uncertainty (±)" v={selected.uncertainty === null ? "not estimated" : selected.uncertainty === 0 ? "exact (authored)" : `± ${selected.uncertainty.toFixed(3)} ${unitLabelFromRow(selected.unit)}`} />
          <Row k="Confidence" v="not estimated" />
          <Row k="Method" v={selected.method} />
          <Row k="Supporting Evidence" v={`world entities: ${provenanceSummary(world)}`} />
          <Row k="Scale" v={selected.scale_state} />
        </div>
      )}

      <div className="px-3 py-2 border-t border-[var(--border)]">
        <div className="text-[10px] uppercase tracking-wider text-fg-subtle mb-1">Measurements ({saved.length})</div>
        {saved.length === 0 ? (
          <div className="text-fg-subtle">None saved yet.</div>
        ) : (
          <table className="w-full text-[11px]">
            <thead className="text-fg-subtle text-left">
              <tr><th>Type</th><th>Value</th><th>State</th><th>±</th><th>Created</th><th /></tr>
            </thead>
            <tbody>
              {saved.map((m) => (
                <tr
                  key={m.id}
                  onClick={() => { setSelectedId(m.id); getController()?.focusMeasurement(m.id); }}
                  className={`cursor-pointer hover:bg-elevated ${m.id === selectedId ? "text-fg bg-[var(--accent-subtle)]" : ""}`}
                  title={m.method}
                >
                  <td>{KIND_LABEL[m.kind]}</td>
                  <td className="font-mono">{formatValue(m.value, unitLabelFromRow(m.unit), 2)}</td>
                  <td><StateBadge state={stateFor(m)} /></td>
                  <td className="font-mono">{m.uncertainty === null ? "n/e" : m.uncertainty === 0 ? "exact" : m.uncertainty.toFixed(2)}</td>
                  <td>{m.created_at ? new Date(m.created_at).toLocaleTimeString() : "—"}</td>
                  <td>
                    <button type="button" aria-label={`Delete ${m.id}`} onClick={(e) => { e.stopPropagation(); void remove(m.id); }} className="text-fg-subtle hover:text-error cursor-pointer">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && profile.bins.length > 0 && (
        <div className="px-3 py-2 border-t border-[var(--border)]">
          <div className="text-[10px] uppercase tracking-wider text-fg-subtle mb-1">Measurement Profile (saved values, {profile.unit})</div>
          <div className="flex items-end gap-0.5 h-12">
            {profile.bins.map((b, i) => (
              <div key={i} className="flex-1 bg-[var(--accent)]/60" style={{ height: `${(b.count / maxCount) * 100}%` }} title={`${b.lo.toFixed(2)}–${b.hi.toFixed(2)}: ${b.count}`} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-fg-subtle shrink-0">{k}</span>
      <span className="text-right text-fg break-words min-w-0">{v}</span>
    </div>
  );
}
