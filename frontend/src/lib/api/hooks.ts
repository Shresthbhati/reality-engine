"use client";

import { useState, useEffect, useCallback } from "react";
import type { WorldRow, SessionRow, EvidenceRow, WorldVersionRow } from "@/lib/types";
import type { WorldIR, CamerasPayload } from "@/types/worldir";
import { listWorlds, listWorldVersions } from "./worlds";
import { listSessions } from "./sessions";
import { listEvidence } from "./evidence";
import { fetchWorldIR, fetchWorldPoints, fetchWorldCameras, fetchWorldReport } from "./worldir";

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
    const res = await listWorlds();
    return res.rows;
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

export function useWorldReport(worldId: string | null) {
  return useApi(
    () => (worldId ? fetchWorldReport(worldId) : Promise.resolve(null)),
    [worldId],
  );
}

export function useWorldVersions(worldId: string | null) {
  return useApi(async () => {
    if (!worldId) return [] as WorldVersionRow[];
    const dtos = await listWorldVersions(worldId);
    return dtos.map((v) => ({
      id: v.id,
      worldId: v.world_id,
      label: v.created_at ?? v.id,
      createdAt: v.created_at ?? v.id,
      changeSummary: `${v.changed_entity_ids?.length ?? 0} entities changed`,
      isCurrent: v.is_current ?? false,
    })) satisfies WorldVersionRow[];
  }, [worldId]);
}
