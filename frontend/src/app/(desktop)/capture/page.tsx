"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
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
  const [requests, setRequests] = useState<CaptureRequestItem[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [statusFilter, setStatusFilter] = useState<"open" | "satisfied" | "dismissed">("open");
  const [loadingRequests, setLoadingRequests] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [activeTab, setActiveTab] = useState<"requests" | "upload" | "sessions">("requests");

  const loadRequests = useCallback(async (wid: string) => {
    if (!wid) {
      setRequests([]);
      return;
    }
    setLoadingRequests(true);
    try {
      const res = await listCaptureRequests(wid);
      setRequests(res.items || []);
    } catch {
      setRequests([]);
    } finally {
      setLoadingRequests(false);
    }
  }, []);

  const loadSessions = useCallback(async (wid?: string) => {
    setLoadingSessions(true);
    try {
      const res = await listSessions({ worldId: wid || undefined });
      setSessions(res.rows || []);
    } catch {
      setSessions([]);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    if (worldId) {
      void loadRequests(worldId);
      void loadSessions(worldId);
    } else {
      void loadSessions();
    }
  }, [worldId, loadRequests, loadSessions]);

  const handleDismiss = async (requestId: string) => {
    if (!worldId) return;
    try {
      await updateCaptureRequestStatus(worldId, requestId, "dismissed");
      void loadRequests(worldId);
    } catch (err) {
      console.error("Failed to dismiss request", err);
    }
  };

  const handleSatisfy = async (requestId: string) => {
    if (!worldId) return;
    try {
      await updateCaptureRequestStatus(worldId, requestId, "satisfied");
      void loadRequests(worldId);
    } catch (err) {
      console.error("Failed to mark satisfied", err);
    }
  };

  const filteredRequests = sortRequestsByGain(filterRequestsByStatus(requests, statusFilter));

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto bg-zinc-950 p-6 text-zinc-100">
      <PageHeader title="Capture & Guidance" />
      <p className="mt-1 text-xs text-zinc-400">
        Manage photo capture sessions, fulfill intelligent capture requests, and update existing world models incrementally.
      </p>

      {/* World Context Bar */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
        <div className="flex items-center gap-3">
          <Layers className="h-5 w-5 text-sky-400" />
          <div>
            <div className="text-xs font-semibold text-zinc-300">Active Target World</div>
            <div className="text-sm font-mono text-zinc-100">
              {worldId ? worldId : "No specific world selected (Creating new world)"}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {worldId ? (
            <button
              onClick={() => router.push(`/worlds/${worldId}`)}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-700 transition-colors"
            >
              <span>Open in 3D Studio</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              onClick={() => router.push("/worlds")}
              className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-700 transition-colors"
            >
              Choose Existing World
            </button>
          )}
        </div>
      </div>

      {/* Primary Navigation Tabs */}
      <div className="mt-6 flex border-b border-zinc-800">
        <button
          onClick={() => setActiveTab("requests")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "requests"
              ? "border-sky-500 text-sky-400"
              : "border-transparent text-zinc-400 hover:text-zinc-200"
          }`}
        >
          <Sparkles className="h-4 w-4" />
          <span>Capture Requests ({requests.filter((r) => r.status === "open").length})</span>
        </button>

        <button
          onClick={() => setActiveTab("upload")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "upload"
              ? "border-sky-500 text-sky-400"
              : "border-transparent text-zinc-400 hover:text-zinc-200"
          }`}
        >
          <Camera className="h-4 w-4" />
          <span>Add Photographs</span>
        </button>

        <button
          onClick={() => setActiveTab("sessions")}
          className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${
            activeTab === "sessions"
              ? "border-sky-500 text-sky-400"
              : "border-transparent text-zinc-400 hover:text-zinc-200"
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
              <span className="text-xs text-zinc-400">Status filter:</span>
              {(["open", "satisfied", "dismissed"] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setStatusFilter(s)}
                  className={`rounded-lg px-2.5 py-1 text-xs capitalize transition-colors ${
                    statusFilter === s
                      ? "bg-sky-500/20 text-sky-300 font-semibold border border-sky-500/40"
                      : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>

            {worldId && (
              <button
                onClick={() => loadRequests(worldId)}
                className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loadingRequests ? "animate-spin" : ""}`} />
                <span>Refresh</span>
              </button>
            )}
          </div>

          {!worldId ? (
            <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center">
              <Sparkles className="mx-auto h-8 w-8 text-zinc-500" />
              <h3 className="mt-2 text-sm font-semibold text-zinc-200">No world selected</h3>
              <p className="mt-1 text-xs text-zinc-400">
                Select a reconstructed world to inspect its unobserved areas and guidance requests.
              </p>
              <Link
                href="/worlds"
                className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-sky-400"
              >
                Browse Worlds
              </Link>
            </div>
          ) : filteredRequests.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center">
              <CheckCircle2 className="mx-auto h-8 w-8 text-emerald-400" />
              <h3 className="mt-2 text-sm font-semibold text-zinc-200">No {statusFilter} capture requests</h3>
              <p className="mt-1 text-xs text-zinc-400">
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
              <h3 className="text-base font-semibold text-zinc-100">
                {worldId ? `Add photographs to ${worldId}` : "Create a new spatial reconstruction"}
              </h3>
              <p className="mt-1 text-xs text-zinc-400">
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
            <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center">
              <Clock className="mx-auto h-8 w-8 text-zinc-500" />
              <h3 className="mt-2 text-sm font-semibold text-zinc-200">No capture sessions yet</h3>
              <p className="mt-1 text-xs text-zinc-400">
                Upload photos to launch an ingestion session.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-zinc-800 rounded-xl border border-zinc-800 bg-zinc-900/60 overflow-hidden">
              {sessions.map((sess) => (
                <div
                  key={sess.id}
                  className="flex items-center justify-between p-4 hover:bg-zinc-800/40 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Camera className="h-5 w-5 text-zinc-400" />
                    <div>
                      <div className="text-sm font-medium text-zinc-100">{sess.name || sess.id}</div>
                      <div className="text-xs text-zinc-400">
                        {sess.location ? `${sess.location} · ` : ""}{sess.state}
                      </div>
                    </div>
                  </div>

                  <Link
                    href={`/sessions`}
                    className="flex items-center gap-1 text-xs font-semibold text-sky-400 hover:text-sky-300"
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
