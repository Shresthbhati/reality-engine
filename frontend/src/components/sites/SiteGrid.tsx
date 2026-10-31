"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import {
  centerM,
  contentBounds,
  fitView,
  footprintRect,
  gridLines,
  nudge,
  nudgeStepM,
  overlappingIds,
  positionFromPoint,
  snapPoint,
  toMetres,
  toScreen,
  type Placement,
  type PlacementPosition,
  type ViewTransform,
  type Vec2,
} from "@/lib/sites/site-grid-model";
import { isMapContextPlacement as isMap } from "@/lib/sites/compose-model";

interface Props {
  gridCellM: number;
  placements: Placement[];
  overlaps: [string, string][];
  selectedId: string | null;
  /** Metres to snap drags to; 0 = free. */
  snapM: number;
  onSelect: (id: string | null) => void;
  /** Called once when a drag or keyboard nudge ends. */
  onMove: (id: string, pos: PlacementPosition) => void;
  onRemoveRequest: (id: string) => void;
}

interface DragState {
  id: string;
  pointerId: number;
  startPt: Vec2; // pointer at start, metres
  startCenter: Vec2;
  center: Vec2; // live
  moved: boolean;
}

const MIN_SIZE = { w: 320, h: 280 };

export default function SiteGrid({ gridCellM, placements, overlaps, selectedId, snapM, onSelect, onMove, onRemoveRequest }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState(MIN_SIZE);
  const [view, setView] = useState<ViewTransform | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const panRef = useRef<{ x: number; y: number; v: ViewTransform; pointerId: number } | null>(null);

  const overlapSet = useMemo(() => overlappingIds(overlaps), [overlaps]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(MIN_SIZE.w, Math.round(r.width)), h: Math.max(MIN_SIZE.h, Math.round(r.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fit = useCallback(() => {
    setView(fitView(contentBounds(placements, gridCellM), size.w, size.h));
  }, [placements, gridCellM, size.w, size.h]);

  // Auto-fit when the viewport, grid size or the NUMBER of placements changes - not on every move,
  // so a drag or nudge never makes the canvas jump under the pointer.
  const fitKey = `${size.w}x${size.h}|${gridCellM}|${placements.length}`;
  const lastFit = useRef("");
  useEffect(() => {
    if (lastFit.current !== fitKey) {
      lastFit.current = fitKey;
      fit();
    }
  }, [fitKey, fit]);

  const v = view ?? fitView(contentBounds(placements, gridCellM), size.w, size.h);

  const pointerMetres = useCallback(
    (e: { clientX: number; clientY: number }): Vec2 => {
      const r = svgRef.current!.getBoundingClientRect();
      return toMetres({ x: e.clientX - r.left, y: e.clientY - r.top }, v);
    },
    [v],
  );

  const zoomAt = (factor: number, px: { x: number; y: number }) => {
    const m = toMetres(px, v);
    const pxPerM = Math.min(Math.max(v.pxPerM * factor, 0.02), 200);
    setView({ pxPerM, originX: m.x - px.x / pxPerM, originZ: m.z - px.y / pxPerM });
  };

  const lines = gridLines(v, size.w, size.h, gridCellM);
  const emptyCenterPx = { x: size.w / 2, y: size.h / 2 };

  const onKeyDown = (e: React.KeyboardEvent, p: Placement) => {
    const fine = e.shiftKey;
    const step = nudgeStepM(gridCellM, fine);
    const delta: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step], // toward the back
      ArrowDown: [0, step], // toward the viewer
    };
    if (delta[e.key]) {
      e.preventDefault();
      const [dx, dz] = delta[e.key];
      onMove(p.id, nudge(p, dx, dz, gridCellM));
    } else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      onRemoveRequest(p.id);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(p.id);
    } else if (e.key === "Escape") {
      onSelect(null);
    }
  };

  return (
    <div ref={wrapRef} className="relative w-full h-full min-h-[320px] overflow-hidden rounded-lg border" style={{ background: "var(--bg-base)", borderColor: "var(--border)" }}>
      <svg
        ref={svgRef}
        width={size.w}
        height={size.h}
        role="group"
        aria-label="Top-down site grid. Back is at the top, front at the bottom. Select a placement and use arrow keys to move it."
        className="absolute inset-0 block touch-none select-none"
        onWheel={(e) => {
          const r = svgRef.current!.getBoundingClientRect();
          zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, { x: e.clientX - r.left, y: e.clientY - r.top });
        }}
        onPointerDown={(e) => {
          if (e.target !== e.currentTarget && (e.target as Element).getAttribute("data-bg") !== "1") return;
          onSelect(null);
          panRef.current = { x: e.clientX, y: e.clientY, v, pointerId: e.pointerId };
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const pan = panRef.current;
          if (!pan || pan.pointerId !== e.pointerId) return;
          setView({
            pxPerM: pan.v.pxPerM,
            originX: pan.v.originX - (e.clientX - pan.x) / pan.v.pxPerM,
            originZ: pan.v.originZ - (e.clientY - pan.y) / pan.v.pxPerM,
          });
        }}
        onPointerUp={() => {
          panRef.current = null;
        }}
      >
        <rect data-bg="1" x={0} y={0} width={size.w} height={size.h} fill="transparent" />
        {lines.xs.map((x) => {
          const sx = toScreen({ x, z: 0 }, v).x;
          return <line key={`x${x}`} x1={sx} x2={sx} y1={0} y2={size.h} style={{ stroke: "var(--border)" }} strokeWidth={1} pointerEvents="none" />;
        })}
        {lines.zs.map((z) => {
          const sy = toScreen({ x: 0, z }, v).y;
          return <line key={`z${z}`} x1={0} x2={size.w} y1={sy} y2={sy} style={{ stroke: "var(--border)" }} strokeWidth={1} pointerEvents="none" />;
        })}
        {v.pxPerM * gridCellM > 48 &&
          lines.xs.slice(0, -1).flatMap((x) =>
            lines.zs.slice(0, -1).map((z) => {
              const s = toScreen({ x, z }, v);
              return (
                <text key={`c${x}_${z}`} x={s.x + 4} y={s.y + 12} fontSize={10} style={{ fill: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }} pointerEvents="none">
                  {Math.round(x / gridCellM)},{Math.round(z / gridCellM)}
                </text>
              );
            }),
          )}

        {placements.map((p) => {
          const live = drag?.id === p.id ? drag.center : p.center_m;
          const c = toScreen(live, v);
          const r = footprintRect(p, gridCellM);
          const w = r.width_m * v.pxPerM;
          const d = r.depth_m * v.pxPerM;
          const selected = p.id === selectedId;
          const overlapped = overlapSet.has(p.id);
          const stroke = overlapped ? "var(--warning)" : selected ? "var(--accent)" : "var(--text-secondary)";
          return (
            <g
              key={p.id}
              tabIndex={0}
              role="button"
              aria-pressed={selected}
              aria-label={`${p.label}, ${p.world_name}, cell ${p.cell_x}, ${p.cell_z}${r.isFallback ? ", size unknown" : ""}${overlapped ? ", overlaps another placement" : ""}`}
              className="outline-none cursor-grab focus-visible:[&>g>rect]:stroke-[3]"
              onKeyDown={(e) => onKeyDown(e, p)}
              onFocus={() => onSelect(p.id)}
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelect(p.id);
                const pt = pointerMetres(e);
                const center = centerM(p, gridCellM);
                setDrag({ id: p.id, pointerId: e.pointerId, startPt: pt, startCenter: center, center, moved: false });
                (e.currentTarget as Element).setPointerCapture(e.pointerId);
              }}
              onPointerMove={(e) => {
                if (!drag || drag.id !== p.id || drag.pointerId !== e.pointerId) return;
                const pt = pointerMetres(e);
                let next = { x: drag.startCenter.x + pt.x - drag.startPt.x, z: drag.startCenter.z + pt.z - drag.startPt.z };
                if (snapM > 0) next = snapPoint(next, snapM);
                setDrag({ ...drag, center: next, moved: drag.moved || next.x !== drag.startCenter.x || next.z !== drag.startCenter.z });
              }}
              onPointerUp={(e) => {
                if (!drag || drag.id !== p.id) return;
                (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
                if (drag.moved) onMove(p.id, positionFromPoint(drag.center, gridCellM));
                setDrag(null);
              }}
              onPointerCancel={() => setDrag(null)}
            >
              <g transform={`translate(${c.x} ${c.y}) rotate(${p.rotation_deg})`}>
                <rect
                  x={-w / 2}
                  y={-d / 2}
                  width={w}
                  height={d}
                  rx={3}
                  strokeWidth={selected ? 2.5 : 1.5}
                  strokeDasharray={r.isFallback || isMap(p) ? "6 4" : undefined}
                  style={{
                    stroke,
                    fill: overlapped ? "var(--warning-subtle)" : selected ? "var(--accent-subtle)" : "var(--bg-elevated)",
                  }}
                />
                {/* front edge tick so rotation is visible */}
                <line x1={-w / 2} x2={w / 2} y1={d / 2} y2={d / 2} strokeWidth={3} style={{ stroke }} />
              </g>
              {/* a real footprint can be a few metres on a 50 m grid; the marker keeps it findable and draggable */}
              <circle cx={c.x} cy={c.y} r={6} strokeWidth={2} style={{ stroke, fill: "var(--bg-base)" }} />
              <text
                x={c.x}
                y={c.y - 12}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                pointerEvents="none"
                style={{ fill: "var(--text-primary)", paintOrder: "stroke", stroke: "var(--bg-base)", strokeWidth: 3, strokeLinejoin: "round" }}
              >
                {p.label}
              </text>
              {isMap(p) ? (
                <text x={c.x} y={c.y + 18} textAnchor="middle" fontSize={10} pointerEvents="none" style={{ fill: "var(--text-tertiary)" }}>
                  Map data
                </text>
              ) : (
                r.isFallback && (
                  <text x={c.x} y={c.y + 18} textAnchor="middle" fontSize={10} pointerEvents="none" style={{ fill: "var(--text-tertiary)" }}>
                    size unknown
                  </text>
                )
              )}
            </g>
          );
        })}

        <text x={size.w / 2} y={16} textAnchor="middle" fontSize={10} letterSpacing={2} pointerEvents="none" style={{ fill: "var(--text-tertiary)" }}>
          BACK
        </text>
        <text x={size.w / 2} y={size.h - 8} textAnchor="middle" fontSize={10} letterSpacing={2} pointerEvents="none" style={{ fill: "var(--text-tertiary)" }}>
          FRONT
        </text>
      </svg>

      {placements.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none px-6 text-center">
          <p className="text-sm max-w-xs" style={{ color: "var(--text-secondary)" }}>
            Nothing placed yet. Use “Add world” to put a reconstructed world on this grid.
          </p>
        </div>
      )}

      <div className="absolute right-2 top-2 flex flex-col gap-1">
        {[
          { label: "Fit to content", icon: Maximize2, run: fit },
          { label: "Zoom in", icon: ZoomIn, run: () => zoomAt(1.25, emptyCenterPx) },
          { label: "Zoom out", icon: ZoomOut, run: () => zoomAt(1 / 1.25, emptyCenterPx) },
        ].map(({ label, icon: Icon, run }) => (
          <button
            key={label}
            type="button"
            onClick={run}
            title={label}
            aria-label={label}
            className="w-8 h-8 flex items-center justify-center rounded-md border hover:bg-[var(--bg-hover)]"
            style={{ background: "var(--bg-surface)", borderColor: "var(--border)", color: "var(--text-secondary)" }}
          >
            <Icon className="w-4 h-4" aria-hidden="true" />
          </button>
        ))}
      </div>
      <p className="absolute left-2 bottom-2 text-[11px]" style={{ color: "var(--text-tertiary)" }}>
        Grid cell {gridCellM} m
      </p>
    </div>
  );
}
