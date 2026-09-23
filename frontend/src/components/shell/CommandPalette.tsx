"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  Camera,
  Globe,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { searchAll, type SearchResult, type SearchResultType } from "@/lib/search";

const TYPE_LABEL: Record<SearchResultType, string> = {
  SESSION: "Session",
  WORLD: "World",
  EVIDENCE: "Evidence",
  ANALYSIS: "Analysis",
  RESULT: "Result",
  REPORT: "Report",
};

interface Command {
  label: string;
  href: string;
  icon: typeof Camera;
}

const COMMANDS: Command[] = [
  { label: "Create Session", href: "/sessions/new", icon: Camera },
  { label: "Create World", href: "/worlds/new", icon: Globe },
  { label: "Add Evidence", href: "/evidence", icon: ShieldCheck },
  { label: "Open World", href: "/worlds", icon: Globe },
  { label: "Find Session", href: "/sessions", icon: Camera },
  { label: "Show Running Sessions", href: "/sessions?state=PROCESSING", icon: Zap },
  { label: "Show Recent Evidence", href: "/evidence", icon: ShieldCheck },
];

export default function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [searchState, setSearchState] = useState<{ q: string; rows: SearchResult[] }>({ q: "", rows: [] });
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<Element | null>(null);
  const isOpenRef = useRef(open);

  // Capture the trigger element on open, restore focus to it on close
  // (Escape, backdrop click, item selection, or the parent flipping
  // `open` to false for any other reason all funnel through this).
  useEffect(() => {
    isOpenRef.current = open;
    if (open) {
      previousFocusRef.current = document.activeElement;
      // Don't setQuery here - let the parent control query via key if needed
      requestAnimationFrame(() => inputRef.current?.focus());
    } else {
      const el = previousFocusRef.current;
      if (el instanceof HTMLElement && document.contains(el)) {
        el.focus();
      }
      previousFocusRef.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- cleanup on close
      setQuery("");
      setHighlightedIndex(0);
    }
  }, [open]);

  // Reset query + keyboard highlight when the palette opens — the react.dev
  // "adjust state during render" pattern (guarded, so nothing stateful runs
  // inside an effect body).
  const [prevOpen, setPrevOpen] = useState(open);
  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) {
      setQuery("");
      setHighlightedIndex(0);
    }
  }

  // Reset the keyboard highlight whenever the query changes.
  const [prevQuery, setPrevQuery] = useState(query);
  if (query !== prevQuery) {
    setPrevQuery(query);
    setHighlightedIndex(0);
  }

  // `searchAll` fetches real collections over the API, so it resolves into
  // state — only promise callbacks touch state, never the effect body. Rows
  // are tagged with the query they answer; a superseded query's late response
  // is discarded via `cancelled`.
  useEffect(() => {
    if (query.trim() === "") return;
    let cancelled = false;
    searchAll(query)
      .then((rows) => {
        if (!cancelled) setSearchState({ q: query, rows });
      })
      .catch(() => {
        if (!cancelled) setSearchState({ q: query, rows: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [query]);

  // Only rows that answer the current query are shown; anything stale (or
  // from a previous palette session) reads as an empty result set.
  const results = searchState.q === query ? searchState.rows : [];
  const flatList = query.trim() === "" ? COMMANDS : results;

  const go = useCallback(
    (href: string) => {
      onClose();
      router.push(href);
    },
    [onClose, router]
  );

  useEffect(() => {
    if (!open) return;
    const handler = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        if (flatList.length === 0) return;
        e.preventDefault();
        setHighlightedIndex((i) => (i + 1) % flatList.length);
        return;
      }
      if (e.key === "ArrowUp") {
        if (flatList.length === 0) return;
        e.preventDefault();
        setHighlightedIndex((i) => (i - 1 + flatList.length) % flatList.length);
        return;
      }
      if (e.key === "Enter") {
        const item = flatList[highlightedIndex];
        if (item) {
          e.preventDefault();
          go(item.href);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, flatList, highlightedIndex, go]);

  const handleDialogKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab") return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>("input, button, [href]"));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    const atBoundary = e.shiftKey
      ? active === first || !dialog.contains(active)
      : active === last || !dialog.contains(active);
    if (atBoundary) {
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      style={{ background: "rgba(6,7,9,0.6)" }}
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Search and commands"
        className="w-full max-w-lg rounded-lg overflow-hidden shadow-2xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div className="flex items-center gap-2 px-3 h-12 border-b" style={{ borderColor: "var(--border)" }}>
          <Search className="w-4 h-4 shrink-0" style={{ color: "var(--text-tertiary)" }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Reality Engine…"
            aria-label="Search Reality Engine"
            className="flex-1 bg-transparent outline-none text-sm"
            style={{ color: "var(--text-primary)" }}
          />
          <kbd
            className="text-xs font-mono-num px-1.5 py-0.5 rounded shrink-0"
            style={{ background: "var(--bg-base)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
          >
            Esc
          </kbd>
        </div>

        <div className="max-h-96 overflow-y-auto py-1.5">
          {query.trim() === "" ? (
            <>
              <div className="px-3 pt-1 pb-1.5 text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
                Commands
              </div>
              {COMMANDS.map((c, index) => (
                <button
                  key={c.label}
                  type="button"
                  onClick={() => go(c.href)}
                  className="w-full flex items-center gap-2.5 px-3 h-9 text-sm text-left transition-colors"
                  style={{
                    color: "var(--text-primary)",
                    background: index === highlightedIndex ? "var(--bg-hover)" : "transparent",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <c.icon className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
                  {c.label}
                </button>
              ))}
            </>
          ) : results.length === 0 ? (
            <p className="px-3 py-6 text-sm text-center" style={{ color: "var(--text-tertiary)" }}>
              No results for &quot;{query}&quot;.
            </p>
          ) : (
            results.map((r, index) => (
              <button
                key={`${r.type}-${r.id}`}
                type="button"
                onClick={() => go(r.href)}
                className="w-full flex flex-col items-start gap-0.5 px-3 py-2 text-left transition-colors"
                style={{ background: index === highlightedIndex ? "var(--bg-hover)" : "transparent" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--accent)" }}>
                  {TYPE_LABEL[r.type]}
                </span>
                <span className="text-sm" style={{ color: "var(--text-primary)" }}>
                  {r.title}
                </span>
                {r.subtitle && (
                  <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                    {r.subtitle}
                  </span>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
