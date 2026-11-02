"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Globe, Loader2, MoreHorizontal } from "lucide-react";
import { apiBaseUrl, isApiError, renameWorld, type WorldDto } from "@/lib/api";
import { HOME, SCALE } from "@/lib/copy";
import { cardView } from "@/lib/places";
import { BAND_COLOR, BAND_MARK } from "./bandStyle";
import DeletePlaceDialog from "./DeletePlaceDialog";

export default function PlaceCard({ world, now, onChanged }: { world: WorldDto; now: number; onChanged: () => void }) {
  const view = cardView(world, now);
  const [menuOpen, setMenuOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(world.name);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [thumbFailed, setThumbFailed] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setMenuOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [menuOpen]);

  const saveName = async () => {
    const next = name.trim();
    if (!next || next === world.name) {
      setName(world.name);
      setRenaming(false);
      return;
    }
    try {
      await renameWorld(world.id, next);
      setRenaming(false);
      setError(null);
      onChanged();
    } catch (err) {
      setError(isApiError(err) ? err.message : String(err));
    }
  };

  const thumb = world.thumbnail_url && !thumbFailed ? `${apiBaseUrl()}${world.thumbnail_url}` : null;

  return (
    <article className="relative flex flex-col overflow-hidden rounded-lg" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}>
      {!renaming && <Link href={`/worlds/${world.id}`} className="absolute inset-0 z-0" aria-label={`Open ${world.name}`} />}

      <div className="relative flex aspect-[4/3] items-center justify-center" style={{ background: "var(--bg-elevated)" }}>
        {thumb ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumb} alt="" className="photo-outline h-full w-full object-cover" onError={() => setThumbFailed(true)} />
        ) : (
          <Globe className="h-8 w-8" style={{ color: "var(--text-tertiary)" }} aria-hidden />
        )}
        {view.building && (
          <div
            className="absolute inset-x-0 bottom-0 px-3 py-2 text-xs"
            style={{ background: "color-mix(in srgb, var(--bg-base) 85%, transparent)", color: "var(--text-primary)" }}
            role="status"
          >
            <div className="flex items-center gap-1.5">
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> {HOME.building}: {view.building.step}
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded" style={{ background: "var(--border)" }}>
              <div className="h-full" style={{ width: `${Math.max(4, view.building.percent)}%`, background: "var(--accent)" }} />
            </div>
          </div>
        )}
      </div>

      <div className="pointer-events-none relative z-10 flex flex-col gap-1.5 p-3">
        {renaming ? (
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={() => void saveName()}
            onKeyDown={(e) => {
              if (e.key === "Enter") void saveName();
              if (e.key === "Escape") {
                setName(world.name);
                setRenaming(false);
              }
            }}
            aria-label="Place name"
            className="pointer-events-auto h-8 rounded-md px-2 text-sm font-semibold"
            style={{ background: "var(--bg-base)", border: "1px solid var(--accent-border)", color: "var(--text-primary)" }}
          />
        ) : (
          <h3 className="truncate pr-8 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {world.name}
          </h3>
        )}
        {error && (
          <p role="alert" className="text-xs" style={{ color: "var(--error)" }}>
            {error}
          </p>
        )}
        <p className="text-xs tabular-nums" style={{ color: "var(--text-secondary)" }}>
          {view.photoLine}
          {view.updated ? ` · ${view.updated}` : ""}
        </p>
        <div className="flex flex-wrap gap-1.5">
          {view.quality && (
            <span
              className="rounded px-1.5 py-0.5 text-[11px] font-medium"
              style={{ color: BAND_COLOR[view.quality.band], border: `1px solid ${BAND_COLOR[view.quality.band]}` }}
            >
              <span aria-hidden>{BAND_MARK[view.quality.band]} </span>
              {view.quality.label}
            </span>
          )}
          {view.notRealSize && (
            <span className="rounded px-1.5 py-0.5 text-[11px]" style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
              {SCALE.relative}
            </span>
          )}
        </div>
      </div>

      <div ref={menuRef} className="absolute bottom-2 right-2 z-20">
        <button
          type="button"
          aria-label={`More options for ${world.name}`}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
          className="flex h-8 w-8 items-center justify-center rounded-md pointer-coarse:h-11 pointer-coarse:w-11"
          style={{ color: "var(--text-secondary)" }}
        >
          <MoreHorizontal className="h-4 w-4" aria-hidden />
        </button>
        {menuOpen && (
          <div
            role="menu"
            className="absolute bottom-9 right-0 w-32 overflow-hidden rounded-md py-1 text-sm shadow-lg"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
          >
            <button
              role="menuitem"
              type="button"
              className="block w-full px-3 py-1.5 text-left"
              style={{ color: "var(--text-primary)" }}
              onClick={() => {
                setMenuOpen(false);
                setRenaming(true);
              }}
            >
              {HOME.rename}
            </button>
            <button
              role="menuitem"
              type="button"
              className="block w-full px-3 py-1.5 text-left"
              style={{ color: "var(--error)" }}
              onClick={() => {
                setMenuOpen(false);
                setDeleting(true);
              }}
            >
              {HOME.delete}
            </button>
          </div>
        )}
      </div>

      {deleting && (
        <DeletePlaceDialog
          id={world.id}
          name={world.name}
          onClose={() => setDeleting(false)}
          onDeleted={() => {
            setDeleting(false);
            onChanged();
          }}
        />
      )}
    </article>
  );
}
