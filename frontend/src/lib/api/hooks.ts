"use client";

import { useState, useEffect, useCallback } from "react";
import type { WorldRow, SessionRow, EvidenceRow } from "@/lib/types";
import type { WorldIR, CamerasPayload } from "@/types/worldir";
import { listWorlds } from "./worlds";
import { listSessions } from "./sessions";
import { listEvidence } from "./evidence";
import { fetchWorldIR, fetchWorldPoints, fetchWorldCameras } from "./worldir";

export interface UseApiResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [trigger, setTrigger] = useState(0);

  const refetch = useCallback(() => {
    setIsLoading(true);
    setError(null);
    setTrigger((t) => t + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetcher()
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setIsLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, trigger]);

  return { data, isLoading, error, refetch };
}

export function useWorlds() {
  return useApi(async () => {
    try {
      // First check local /api/worlds proxy which provides 3D metadata flags
      const res = await fetch("/api/worlds");
      if (res.ok) {
        const json = await res.json();
        if (json.items && Array.isArray(json.items)) {
          return json.items as (WorldRow & { has3DData?: boolean })[];
        }
      }
    } catch {
      // fallback to canonical listWorlds
    }
    const res = await listWorlds();
    return res.rows as (WorldRow & { has3DData?: boolean })[];
  });
}

export function useSessions(worldId?: string | null) {
  return useApi(async () => {
    const res = await listSessions({ worldId });
    return res.rows;
  }, [worldId]);
}

export function useEvidence(sessionId?: string | null) {
  return useApi(async () => {
    const res = await listEvidence({ sessionId: sessionId || undefined });
    return res.rows;
  }, [sessionId]);
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
