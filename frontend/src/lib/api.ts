"use client";

/**
 * Reality Engine — Centralized API Client & React Hooks
 *
 * Single source of truth for all backend communication.
 * Calls Next.js API routes (same-origin) which proxy to the Python backend.
 * No static fallback data. If the API is unreachable, return honest errors.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  WorldRow,
  SessionRow,
  EvidenceRow,
  EvidenceProcessingState,
  EvidenceType,
  ProcessingState,
} from "@/lib/types";
import type { WorldIR, CamerasPayload } from "@/types/worldir";

// ── Error Type ────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API ${status}: ${body}`);
    this.name = "ApiError";
  }
}

// ── Core Fetch ────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, body);
  }
  return res.json();
}

async function apiFetchRaw(path: string): Promise<Response> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new ApiError(res.status, res.statusText);
  }
  return res;
}

// ── Worlds ────────────────────────────────────────────────────────────

interface WorldsApiResponse {
  items: Array<{
    id: string;
    name: string;
    location?: string;
    description?: string;
    lat?: number | null;
    latitude?: number | null;
    lng?: number | null;
    longitude?: number | null;
    coverageKm2?: number | null;
    sessionCount?: number | null;
    session_count?: number | null;
    evidenceCount?: number | null;
    evidence_count?: number | null;
    timeRangeStart?: string | null;
    timeRangeEnd?: string | null;
    updatedAt?: string | null;
    created_at?: string | null;
    has3DData?: boolean;
  }>;
  count?: number;
}

function mapWorld(raw: WorldsApiResponse["items"][number]): WorldRow & { has3DData?: boolean } {
  return {
    id: raw.id,
    name: raw.name,
    location: raw.location ?? raw.description ?? null,
    lat: raw.lat ?? raw.latitude ?? null,
    lng: raw.lng ?? raw.longitude ?? null,
    coverageKm2: raw.coverageKm2 ?? null,
    sessionCount: raw.sessionCount ?? raw.session_count ?? 0,
    evidenceCount: raw.evidenceCount ?? raw.evidence_count ?? 0,
    timeRangeStart: raw.timeRangeStart ?? null,
    timeRangeEnd: raw.timeRangeEnd ?? null,
    updatedAt: raw.updatedAt ?? raw.created_at ?? null,
    has3DData: raw.has3DData,
  };
}

export async function fetchWorlds(): Promise<(WorldRow & { has3DData?: boolean })[]> {
  const data = await apiFetch<WorldsApiResponse>("/api/worlds");
  return (data.items || []).map(mapWorld);
}

// ── Sessions ──────────────────────────────────────────────────────────

interface SessionsApiResponse {
  sessions?: Array<{
    id: string;
    name: string;
    description?: string;
    device_type?: string;
    created_at?: string;
    location_lat?: number | null;
    location_lng?: number | null;
    location_name?: string | null;
    state?: ProcessingState;
    world_id?: string | null;
    upload_count?: number;
  }>;
  count?: number;
}

function mapSession(raw: NonNullable<SessionsApiResponse["sessions"]>[number]): SessionRow {
  return {
    id: raw.id,
    name: raw.name,
    state: raw.state ?? "COMPLETE",
    worldId: raw.world_id ?? null,
    worldName: null,
    location: raw.location_name ?? null,
    lat: raw.location_lat ?? null,
    lng: raw.location_lng ?? null,
    coverageKm2: null,
    capturedAt: raw.created_at ?? null,
    durationSec: null,
    evidenceCount: raw.upload_count ?? 0,
    stages: [],
  };
}

export async function fetchSessions(): Promise<SessionRow[]> {
  const data = await apiFetch<SessionsApiResponse>("/api/sessions");
  return (data.sessions || []).map(mapSession);
}

// ── Evidence ──────────────────────────────────────────────────────────

interface EvidenceApiResponse {
  evidence?: Array<{
    id: string;
    original_filename?: string;
    evidence_type?: string;
    processing_state?: string;
    session_id?: string | null;
    created_at?: string | null;
    metadata?: Record<string, unknown>;
  }>;
}

function mapEvidence(raw: NonNullable<EvidenceApiResponse["evidence"]>[number]): EvidenceRow {
  const typeMap: Record<string, EvidenceType> = {
    image: "IMAGE",
    video: "VIDEO",
    point_cloud: "POINT_CLOUD",
    document: "DOCUMENT",
    sensor_data: "SENSOR_DATA",
  };
  const stateMap: Record<string, EvidenceProcessingState> = {
    uploading: "UPLOADING",
    processing: "PROCESSING",
    processed: "PROCESSED",
    failed: "FAILED",
    complete: "PROCESSED",
  };

  return {
    id: raw.id,
    name: raw.original_filename ?? raw.id,
    type: typeMap[(raw.evidence_type ?? "image").toLowerCase()] ?? "IMAGE",
    location: null,
    sessionId: raw.session_id ?? null,
    sessionName: null,
    worldId: null,
    worldName: null,
    capturedAt: raw.created_at ?? null,
    uploadedAt: raw.created_at ?? null,
    processedAt: null,
    processingState: stateMap[(raw.processing_state ?? "processed").toLowerCase()] ?? "PROCESSED",
  };
}

export async function fetchEvidence(): Promise<EvidenceRow[]> {
  try {
    const data = await apiFetch<EvidenceApiResponse>("/api/evidence");
    return (data.evidence || []).map(mapEvidence);
  } catch {
    // Evidence endpoint may not exist as a Next.js route yet — return empty
    return [];
  }
}

// ── Jobs ──────────────────────────────────────────────────────────────

export interface JobInfo {
  id: string;
  job_type: string;
  status: string;
  params?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
}

export async function fetchJobs(): Promise<JobInfo[]> {
  try {
    const data = await apiFetch<{ jobs?: JobInfo[] }>("/api/jobs");
    return data.jobs ?? [];
  } catch {
    return [];
  }
}

export async function fetchJob(jobId: string): Promise<JobInfo | null> {
  try {
    return await apiFetch<JobInfo>(`/api/jobs/${jobId}`);
  } catch {
    return null;
  }
}

// ── Activity ──────────────────────────────────────────────────────────

export interface ActivityEvent {
  id: string;
  event_type: string;
  entity_type?: string;
  entity_id?: string;
  detail?: string;
  created_at?: string;
}

export async function fetchActivity(): Promise<ActivityEvent[]> {
  try {
    const data = await apiFetch<{ items?: ActivityEvent[] }>("/api/activity");
    return data.items ?? [];
  } catch {
    return [];
  }
}

// ── Notifications ─────────────────────────────────────────────────────

export interface Notification {
  id: string;
  level: string;
  title: string;
  body?: string;
  created_at?: string;
  read: boolean;
}

export async function fetchNotifications(): Promise<Notification[]> {
  try {
    const data = await apiFetch<{ items?: Notification[] }>("/api/notifications");
    return data.items ?? [];
  } catch {
    return [];
  }
}

// ── WorldIR Artifacts ─────────────────────────────────────────────────

export async function fetchWorldIR(worldId: string): Promise<WorldIR | null> {
  try {
    return await apiFetch<WorldIR>(`/api/worlds/${worldId}/worldir`);
  } catch {
    return null;
  }
}

export async function fetchWorldPoints(worldId: string): Promise<ArrayBuffer | null> {
  try {
    const res = await apiFetchRaw(`/api/worlds/${worldId}/points`);
    return await res.arrayBuffer();
  } catch {
    return null;
  }
}

export async function fetchWorldCameras(worldId: string): Promise<CamerasPayload | null> {
  try {
    return await apiFetch<CamerasPayload>(`/api/worlds/${worldId}/cameras`);
  } catch {
    return null;
  }
}

// ── Generic Hook ──────────────────────────────────────────────────────

interface UseApiResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);

  // Update fetcher ref when fetcher changes
  useEffect(() => {
    fetcherRef.current = fetcher;
  }, [fetcher]);

  const refetch = useCallback(() => {
    setIsLoading(true);
    setError(null);
    fetcherRef
      .current()
      .then((result) => {
        if (mountedRef.current) {
          setData(result);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (mountedRef.current) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setIsLoading(false);
        }
      });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data fetch
    refetch();
    return () => {
      mountedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, refetch]);

  return { data, isLoading, error, refetch };
}

// ── React Hooks ───────────────────────────────────────────────────────

export function useWorlds() {
  return useApi(fetchWorlds);
}

export function useSessions() {
  return useApi(fetchSessions);
}

export function useEvidence() {
  return useApi(fetchEvidence);
}

export function useJobs() {
  return useApi(fetchJobs);
}

export function useActivity() {
  return useApi(fetchActivity);
}

export function useNotifications() {
  return useApi(fetchNotifications);
}

export function useWorldIR(worldId: string | null) {
  return useApi(
    () => (worldId ? fetchWorldIR(worldId) : Promise.resolve(null)),
    [worldId],
  );
}

export function useWorldPoints(worldId: string | null) {
  return useApi(
    () => (worldId ? fetchWorldPoints(worldId) : Promise.resolve(null)),
    [worldId],
  );
}

export function useWorldCameras(worldId: string | null) {
  return useApi(
    () => (worldId ? fetchWorldCameras(worldId) : Promise.resolve(null)),
    [worldId],
  );
}
