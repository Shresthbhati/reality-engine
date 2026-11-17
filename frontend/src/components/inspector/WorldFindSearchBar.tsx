"use client";

import { useState, useEffect, useRef } from "react";
import { Search, X, Loader2, MapPin, ChevronRight, CheckCircle2, User, HelpCircle } from "lucide-react";
import { queryFind, type FindResult } from "@/lib/api/query";

interface WorldFindSearchBarProps {
  worldId: string;
  version?: string;
  onSelectEntity: (id: string, position?: [number, number, number] | null) => void;
  onFrameEntity?: (id: string) => void;
}

export function WorldFindSearchBar({
  worldId,
  version,
  onSelectEntity,
  onFrameEntity,
}: WorldFindSearchBarProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<FindResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setLoading(false);
      setError(null);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await queryFind(worldId, query.trim(), version);
        setResults(res.results || []);
        setOpen(true);
      } catch (err: unknown) {
        setError("Search query failed");
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query, worldId, version]);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (item: FindResult) => {
    onSelectEntity(item.id, item.position);
    if (onFrameEntity) {
      onFrameEntity(item.id);
    }
    setOpen(false);
  };

  return (
    <div ref={containerRef} className="relative w-full">
      <div className="relative flex items-center">
        <Search
          className="absolute left-2.5 h-3.5 w-3.5 text-[var(--text-tertiary)] pointer-events-none"
          aria-hidden
        />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (!open) setOpen(true);
          }}
          onFocus={() => {
            if (results.length > 0) setOpen(true);
          }}
          placeholder="Find in world (e.g. snack drawer, socket)..."
          className="w-full rounded border border-[var(--border)] bg-[var(--bg-elevated)] py-1.5 pl-8 pr-8 text-[12px] text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none"
        />
        {loading && (
          <Loader2
            className="absolute right-2.5 h-3.5 w-3.5 animate-spin text-[var(--text-tertiary)]"
            aria-hidden
          />
        )}
        {!loading && query && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setResults([]);
              setOpen(false);
            }}
            className="absolute right-2.5 p-0.5 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] cursor-pointer"
            aria-label="Clear search"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </button>
        )}
      </div>

      {open && query.trim() && (
        <div className="absolute left-0 right-0 top-full z-40 mt-1 max-h-72 overflow-y-auto rounded border border-[var(--border)] bg-[var(--bg-elevated)] p-1 shadow-lg">
          {error && (
            <div className="p-2.5 text-[11px] text-[var(--error)] text-center">
              {error}
            </div>
          )}

          {!loading && !error && results.length === 0 && (
            <div className="p-3 text-[11px] text-[var(--text-tertiary)] text-center">
              No matching entities or parts found in this world.
            </div>
          )}

          {results.map((r) => {
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => handleSelect(r)}
                className="flex w-full flex-col gap-1 rounded p-2 text-left hover:bg-[var(--accent-subtle)] focus:bg-[var(--accent-subtle)] focus:outline-none cursor-pointer"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-[12px] text-[var(--text-primary)]">
                    {r.type.charAt(0).toUpperCase() + r.type.slice(1)} ({r.id})
                  </span>
                  <span
                    className={`inline-flex items-center gap-1 rounded px-1.5 py-0.2 text-[10px] font-medium ${
                      r.basis === "user_stated"
                        ? "bg-[var(--info-subtle)] text-[var(--info)]"
                        : "bg-[var(--accent-subtle)] text-[var(--accent)]"
                    }`}
                  >
                    {r.basis === "user_stated" ? (
                      <>
                        <User className="h-2.5 w-2.5" /> User stated
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="h-2.5 w-2.5" /> Observed
                      </>
                    )}
                  </span>
                </div>

                {r.chain && r.chain.length > 0 && (
                  <div className="flex items-center flex-wrap gap-1 text-[11px] text-[var(--text-secondary)]">
                    {r.chain.map((link, idx) => (
                      <span key={link.id} className="flex items-center gap-1">
                        <span>{link.type}</span>
                        {idx < r.chain.length - 1 && (
                          <ChevronRight className="h-2.5 w-2.5 text-[var(--text-tertiary)]" />
                        )}
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex items-center justify-between text-[10px] text-[var(--text-tertiary)]">
                  {r.position && (
                    <span className="flex items-center gap-1 font-mono-num">
                      <MapPin className="h-2.5 w-2.5" />
                      [{r.position.map((n) => n.toFixed(2)).join(", ")}]
                    </span>
                  )}
                  {r.interior === "not_observed" && (
                    <span className="italic text-[var(--warning)]">
                      Closed — contents unobserved
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default WorldFindSearchBar;
