"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Plus } from "lucide-react";
import { isApiError, listWorlds, type WorldDto } from "@/lib/api";
import { HOME } from "@/lib/copy";
import NewPlaceFlow from "./NewPlaceFlow";
import PlaceCard from "./PlaceCard";

const POLL_MS = 3000;

/** Home: the user's places. Polls only while a build is running, so an idle page makes no requests. */
export default function PlaceLibrary() {
  const [worlds, setWorlds] = useState<WorldDto[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [tick, setTick] = useState(0);

  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let alive = true;
    listWorlds()
      .then(({ dtos }) => {
        if (!alive) return;
        setWorlds(dtos);
        setError(null);
        setNow(Date.now());
      })
      .catch((err) => {
        if (alive) setError(isApiError(err) ? err.describe() : String(err));
      });
    return () => {
      alive = false;
    };
  }, [tick]);

  const building = worlds?.some((w) => w.active_job) ?? false;
  useEffect(() => {
    if (!building) return;
    const t = setTimeout(reload, POLL_MS);
    return () => clearTimeout(t);
  }, [building, worlds, reload]);

  if (error && !worlds) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
        <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          {HOME.loadError}
        </p>
        <p className="max-w-md text-xs" style={{ color: "var(--text-tertiary)" }}>
          {HOME.engineHint}
        </p>
        <button
          type="button"
          onClick={reload}
          className="h-9 rounded-md px-3 text-sm font-medium"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {HOME.retry}
        </button>
      </div>
    );
  }

  if (worlds === null) {
    return (
      <div className="h-full overflow-y-auto p-6" aria-busy="true">
        <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(240px,1fr))]">
          {[0, 1, 2].map((i) => (
            <div key={i} className="aspect-[4/3] animate-pulse rounded-lg" style={{ background: "var(--bg-elevated)" }} />
          ))}
        </div>
      </div>
    );
  }

  if (worlds.length === 0) return <NewPlaceFlow hero />;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
            {HOME.title}
          </h1>
          <Link
            href="/reconstruct"
            className="btn-press inline-flex h-9 items-center gap-1.5 rounded-md px-3 text-sm font-semibold"
            style={{ background: "var(--accent)", color: "var(--bg-base)" }}
          >
            <Plus className="h-4 w-4" aria-hidden /> {HOME.newPlace}
          </Link>
        </div>
        <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(240px,1fr))]">
          {worlds.map((w) => (
            <PlaceCard key={w.id} world={w} now={now} onChanged={reload} />
          ))}
        </div>
      </div>
    </div>
  );
}
