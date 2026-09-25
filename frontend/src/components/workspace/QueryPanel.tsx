"use client";

import { useState } from "react";
import { X, Loader2, Search } from "lucide-react";
import { queryNearest, queryWithinRadius, queryContents, queryContainer, type QueryResult } from "@/lib/api/query";
import { isApiError } from "@/lib/api/client";
import type { Entity } from "@/types/worldir";

type Mode = "nearest" | "within_radius" | "contents" | "container";

/** Real spatial/scene-graph queries -- POST/GET the actual sdk.reality
 * SpatialIndex/SceneGraph via apps/api/routes_query.py. Results are
 * selectable back into the viewport via the same onSelectEntity/
 * onFrameEntity callbacks the rest of the workstation uses. */
export default function QueryPanel({
  worldId,
  selectedEntityId,
  onClose,
  onSelectEntity,
  onFrameEntity,
}: {
  worldId: string;
  selectedEntityId: string | null;
  onClose: () => void;
  onSelectEntity: (id: string) => void;
  onFrameEntity: (id: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("nearest");
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const [z, setZ] = useState(0);
  const [k, setK] = useState(5);
  const [radius, setRadius] = useState(2);
  const [entityId, setEntityId] = useState(selectedEntityId ?? "");
  const [results, setResults] = useState<(QueryResult | Entity)[] | null>(null);
  const [container, setContainer] = useState<Entity | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function handleRun() {
    setRunning(true);
    setError(null);
    setResults(null);
    setContainer(undefined);
    try {
      if (mode === "nearest") {
        setResults(await queryNearest(worldId, { x, y, z }, k));
      } else if (mode === "within_radius") {
        setResults(await queryWithinRadius(worldId, { x, y, z }, radius));
      } else if (mode === "contents") {
        setResults(await queryContents(worldId, entityId));
      } else {
        setContainer(await queryContainer(worldId, entityId));
      }
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Query failed.");
    } finally {
      setRunning(false);
    }
  }

  const needsEntityId = mode === "contents" || mode === "container";

  return (
    <div
      className="absolute top-4 left-4 z-20 w-80 rounded-lg p-4 flex flex-col gap-3 shadow-xl max-h-[calc(100%-2rem)] overflow-y-auto"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-1.5" style={{ color: "var(--text-primary)" }}>
          <Search className="w-3.5 h-3.5" />
          Spatial Query
        </h3>
        <button type="button" onClick={onClose} aria-label="Close">
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5 text-xs">
        {(["nearest", "within_radius", "contents", "container"] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className="px-2 py-1 rounded transition-colors"
            style={{
              background: mode === m ? "var(--accent-subtle)" : "var(--bg-elevated)",
              color: mode === m ? "var(--accent)" : "var(--text-secondary)",
              border: `1px solid ${mode === m ? "var(--accent-border)" : "var(--border-subtle)"}`,
            }}
          >
            {m === "nearest" ? "Nearest" : m === "within_radius" ? "Within Radius" : m === "contents" ? "Contents Of" : "Container Of"}
          </button>
        ))}
      </div>

      {needsEntityId ? (
        <label className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
          Entity ID
          <input
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            placeholder="e.g. living-room-room"
            className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
            style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
          />
        </label>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            {(["x", "y", "z"] as const).map((axis) => (
              <label key={axis} className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
                {axis.toUpperCase()}
                <input
                  type="number"
                  step="0.1"
                  value={axis === "x" ? x : axis === "y" ? y : z}
                  onChange={(e) => (axis === "x" ? setX : axis === "y" ? setY : setZ)(Number(e.target.value))}
                  className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
                  style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
                />
              </label>
            ))}
          </div>
          {mode === "nearest" ? (
            <label className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
              k (result count)
              <input
                type="number"
                min={1}
                max={100}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
                style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
              />
            </label>
          ) : (
            <label className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
              Radius (m)
              <input
                type="number"
                min={0}
                step="0.1"
                value={radius}
                onChange={(e) => setRadius(Number(e.target.value))}
                className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
                style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
              />
            </label>
          )}
        </>
      )}

      <button
        type="button"
        onClick={handleRun}
        disabled={running || (needsEntityId && !entityId.trim())}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-medium disabled:opacity-40 transition-colors"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        {running && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {running ? "Running…" : "Run Query"}
      </button>

      {error && (
        <p className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {error}
        </p>
      )}

      {results && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {results.length} result{results.length === 1 ? "" : "s"}
          </p>
          {results.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => {
                onSelectEntity(r.id);
                onFrameEntity(r.id);
              }}
              className="flex items-center justify-between px-2.5 py-1.5 rounded text-xs text-left transition-colors"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)" }}
            >
              <span style={{ color: "var(--text-primary)" }}>{r.name || r.id}</span>
              <span style={{ color: "var(--text-tertiary)" }}>
                {"distance_m" in r && r.distance_m != null ? `${r.distance_m.toFixed(2)}m` : r.type}
              </span>
            </button>
          ))}
        </div>
      )}

      {container !== undefined && (
        container ? (
          <button
            type="button"
            onClick={() => {
              onSelectEntity(container.id);
              onFrameEntity(container.id);
            }}
            className="flex items-center justify-between px-2.5 py-1.5 rounded text-xs text-left transition-colors"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)" }}
          >
            <span style={{ color: "var(--text-primary)" }}>{container.name || container.id}</span>
            <span style={{ color: "var(--text-tertiary)" }}>{container.type}</span>
          </button>
        ) : (
          <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            Not contained by anything.
          </p>
        )
      )}
    </div>
  );
}
