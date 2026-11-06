"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Camera,
  Layers,
  Sparkles,
  ArrowRight,
  RefreshCw,
  CheckCircle2,
  Clock,
  History,
} from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import PhotoDropZone from "@/components/reconstruct/PhotoDropZone";
import { CaptureRequestCard } from "@/components/capture/CaptureRequestCard";
import {
  listCaptureRequests,
  updateCaptureRequestStatus,
  type CaptureRequestItem,
} from "@/lib/api/capture-requests";
import {
  filterRequestsByStatus,
  sortRequestsByGain,
} from "@/lib/studio/capture-requests-model";
import { listSessions, type SessionRow } from "@/lib/api";

export default function CapturePage() {
  return (
    <Suspense>
      <CapturePageInner />
    </Suspense>
  );
}

function CapturePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const worldIdParam = searchParams.get("world") || "";

  const [worldId, setWorldId] = useState<string>(worldIdParam);
  const [statusFilter, setStatusFilter] = useState<"open" | "satisfied" | "dismissed">("open");
  const [activeTab, setActiveTab] = useState<"requests" | "upload" | "sessions">("requests");
  // Bumped by handlers to re-run the loaders; loaded data is tagged with the
  // key it was fetched for, so "loading" is derived (key mismatch) rather than
  // set synchronously inside the effect.
  const [reloadTick, setReloadTick] = useState(0);
  const loadKey = `${worldId}:${reloadTick}`;
  const [requestsLoaded, setRequestsLoaded] = useState<{ key: string; items: CaptureRequestItem[] }>({
    key: "",
    items: [],
  });
  const [sessionsLoaded, setSessionsLoaded] = useState<{ key: string; items: SessionRow[] }>({
    key: "",
    items: [],
  });
  const requests = worldId ? requestsLoaded.items : [];
  const sessions = sessionsLoaded.items;
  const loadingRequests = !!worldId && requestsLoaded.key !== loadKey;
  const loadingSessions = sessionsLoaded.key !== loadKey;

  useEffect(() => {
    let cancelled = false;
    if (worldId) {
      listCaptureRequests(worldId)
        .then((res) => res.items || [])
        .catch(() => [] as CaptureRequestItem[])
        .then((items) => {
          if (!cancelled) setRequestsLoaded({ key: loadKey, items });
        });
    }
    listSessions({ worldId: worldId || undefined })
      .then((res) => res.rows || [])
      .catch(() => [] as SessionRow[])
      .then((items) => {
        if (!cancelled) setSessionsLoaded({ key: loadKey, items });
      });
    return () => {
      cancelled = true;
    };
  }, [worldId, loadKey]);
  const handleDismiss = async (requestId: string) => {
    if (!worldId) return;
    try {
      await updateCaptureRequestStatus(worldId, requestId, "dismissed");
      setReloadTick((t) => t + 1);
    } catch (err) {
      console.error("Failed to dismiss request", err);
    }
  };

  const handleSatisfy = async (requestId: string) => {
    if (!worldId) return;
    try {
      await updateCaptureRequestStatus(worldId, requestId, "satisfied");
      setReloadTick((t) => t + 1);
    } catch (err) {
      console.error("Failed to mark satisfied", err);
    }
  };

  const filteredRequests = sortRequestsByGain(filterRequestsByStatus(requests, statusFilter));

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto bg-base p-6 text-fg">
      <PageHeader title="Capture & Guidance" />
      <p className="mt-1 text-xs text-fg-muted">
        Manage photo capture sessions, fulfill intelligent capture requests, and update existing world models incrementally.
      </p>

      {/* World Context Bar */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line-subtle bg-surface/60 p-4">
        <div className="flex items-center gap-3">
          <Layers className="h-5 w-5 text-info" />
          <div>
            <div className="text-xs font-semibold text-fg-muted">Active Target World</div>
            <div className="text-sm font-mono text-fg">
              {worldId ? worldId : "No specific world selected (Creating new world)"}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {worldId ? (
            <button
              onClick={() => router.push(`/worlds/${worldId}`)}
              className="flex items-center gap-1.5 rounded-lg border border-line bg-elevated px-3 py-1.5 text-xs font-medium text-fg hover:bg-line transition-colors"
            >
              <span>Open in 3D Studio</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              onClick={() => router.push("/worlds")}
              className="rounded-lg border border-line bg-elevated px-3 py-1.5 text-xs font-medium text-fg hover:bg-line transition-colors"
            >
              Choose Existing World
            </button>
          )}
        </div>
      </div>

      {/* Primary Navigation Tabs */}
      <div className="mt-6 flex border-b border-line-subtle">
        <button
          onClick={() => setActiveTab("requests")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "requests"
              ? "border-info text-info"
              : "border-transparent text-fg-muted hover:text-fg"
          }`}
        >
          <Sparkles className="h-4 w-4" />
          <span>Capture Requests ({requests.filter((r) => r.status === "open").length})</span>
        </button>

        <button
          onClick={() => setActiveTab("upload")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "upload"
              ? "border-info text-info"
              : "border-transparent text-fg-muted hover:text-fg"
          }`}
        >
          <Camera className="h-4 w-4" />
          <span>Add Photographs</span>
        </button>

        <button
          onClick={() => setActiveTab("sessions")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "sessions"
              ? "border-info text-info"
              : "border-transparent text-fg-muted hover:text-fg"
          }`}
        >
          <History className="h-4 w-4" />
          <span>Recent Sessions ({sessions.length})</span>
        </button>
      </div>

      {/* Tab 1: Capture Requests */}
      {activeTab === "requests" && (
        <div className="mt-6">
          <div className="flex items-center justify-between pb-4">
            <div className="flex items-center gap-2">
              <span className="text-xs text-fg-muted">Status filter:</span>
              {(["open", "satisfied", "dismissed"] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setStatusFilter(s)}
                  className={`rounded-lg px-2.5 py-1 text-xs capitalize transition-colors ${
                    statusFilter === s
                      ? "bg-info-subtle text-info font-semibold border border-info/40"
                      : "bg-elevated/60 text-fg-muted hover:bg-elevated hover:text-fg"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>

            {worldId && (
              <button
                onClick={() => setReloadTick((t) => t + 1)}
                className="flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loadingRequests ? "animate-spin" : ""}`} />
                <span>Refresh</span>
              </button>
            )}
          </div>

          {!worldId ? (
            <div className="rounded-xl border border-dashed border-line-subtle p-8 text-center">
              <Sparkles className="mx-auto h-8 w-8 text-fg-subtle" />
              <h3 className="mt-2 text-sm font-semibold text-fg">No world selected</h3>
              <p className="mt-1 text-xs text-fg-muted">
                Select a reconstructed world to inspect its unobserved areas and guidance requests.
              </p>
              <Link
                href="/worlds"
                className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-on-primary hover:bg-primary-hover"
              >
                Browse Worlds
              </Link>
            </div>
          ) : filteredRequests.length === 0 ? (
            <div className="rounded-xl border border-dashed border-line-subtle p-8 text-center">
              <CheckCircle2 className="mx-auto h-8 w-8 text-success" />
              <h3 className="mt-2 text-sm font-semibold text-fg">No {statusFilter} capture requests</h3>
              <p className="mt-1 text-xs text-fg-muted">
                {statusFilter === "open"
                  ? "All structural surfaces currently have sufficient coverage, or new requests have not yet been generated."
                  : `No requests with status ${statusFilter}.`}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredRequests.map((req) => (
                <CaptureRequestCard
                  key={req.id}
                  request={req}
                  onAddPhoto={() => setActiveTab("upload")}
                  onDismiss={handleDismiss}
                  onSatisfy={handleSatisfy}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Upload Photographs */}
      {activeTab === "upload" && (
        <div className="mt-6 flex flex-col items-center">
          <div className="w-full max-w-2xl">
            <div className="mb-4 text-center">
              <h3 className="text-base font-semibold text-fg">
                {worldId ? `Add photographs to ${worldId}` : "Create a new spatial reconstruction"}
              </h3>
              <p className="mt-1 text-xs text-fg-muted">
                Newly uploaded photographs register against existing geometry via incremental PnP. Only
                affected regional entities will recompute.
              </p>
            </div>

            <PhotoDropZone
              worldId={worldId || undefined}
              onCreated={(res) => router.push(`/worlds/${res.world_id}`)}
            />
          </div>
        </div>
      )}

      {/* Tab 3: Recent Sessions */}
      {activeTab === "sessions" && (
        <div className="mt-6">
          {sessions.length === 0 ? (
            <div className="rounded-xl border border-dashed border-line-subtle p-8 text-center">
              <Clock className="mx-auto h-8 w-8 text-fg-subtle" />
              <h3 className="mt-2 text-sm font-semibold text-fg">No capture sessions yet</h3>
              <p className="mt-1 text-xs text-fg-muted">
                Upload photos to launch an ingestion session.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-line-subtle rounded-xl border border-line-subtle bg-surface/60 overflow-hidden">
              {sessions.map((sess) => (
                <div
                  key={sess.id}
                  className="flex items-center justify-between p-4 hover:bg-elevated/40 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Camera className="h-5 w-5 text-fg-muted" />
                    <div>
                      <div className="text-sm font-medium text-fg">{sess.name || sess.id}</div>
                      <div className="text-xs text-fg-muted">
                        {sess.location ? `${sess.location} · ` : ""}{sess.state}
                      </div>
                    </div>
                  </div>

                  <Link
                    href={`/sessions`}
                    className="flex items-center gap-1 text-xs font-semibold text-info hover:text-info"
                  >
                    <span>View Session</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
