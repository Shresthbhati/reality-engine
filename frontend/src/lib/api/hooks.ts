"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { WorldRow, SessionRow, EvidenceRow, WorldVersionRow } from "@/lib/types";
import type { WorldIR, CamerasPayload } from "@/types/worldir";
import { listWorlds, listWorldVersions } from "./worlds";
import { listSessions } from "./sessions";
import { listEvidence } from "./evidence";
import { fetchWorldIR, fetchWorldPoints, fetchWorldCameras, fetchWorldReport } from "./worldir";
import { isApiError } from "./client";

/**
 * `absent` is true only when the backend answered successfully and the resource
 * genuinely does not exist yet (e.g. a world with no reconstruction version).
 *
 * It is deliberately NOT set on failure: an outage leaves `data === null` and
 * `absent === false`, with `error` set, so a page can render an error+retry
 * instead of a plausible empty state. Collapsing those two is exactly the bug
 * this hook exists to prevent.
 */
export interface UseApiResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  /** Backend confirmed the resource does not exist (404), as opposed to a failure. */
  absent: boolean;
  refetch: () => void;
}

/**
 * Sentinel for "this hook has no target to ask about" (e.g. no world selected).
 *
 * It is a distinct object identity so the hook can tell it apart from a real
 * ApiError and skip the alarming error UI, while still never pretending a
 * request succeeded with empty data.
 */
export const NOT_REQUESTED = Symbol("not_requested");

export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[] = [],
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [absent, setAbsent] = useState(false);
  const [trigger, setTrigger] = useState(0);

  // Generation counter + AbortController, so an obsolete response can never
  // write state. See the effect below for the scenario this defends against.
  const generationRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const refetch = useCallback(() => {
    setIsLoading(true);
    setError(null);
    setAbsent(false);
    setTrigger((t) => t + 1);
  }, []);

  useEffect(() => {
    // Generation counter + AbortController, so an obsolete response can never
    // write state.
    //
    // Scenario C (A loaded, request B, then request C, B answers late): the
    // `cancelled` boolean alone is not enough, because a response that
    // resolved BEFORE the cleanup ran can still be queued to apply. The
    // generation check rejects any response whose generation is no longer
    // current, even if its promise already resolved.
    const generation = ++generationRef.current;

    // Stop the obsolete network work too, not just its state write.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const isCurrent = () => generationRef.current === generation;

    // A dependency change (e.g. switching worlds) must clear the previous
    // result immediately. Without this, `data` still holds the PREVIOUS
    // world's WorldIR/points/versions while the new world loads, and the user
    // sees world A's geometry labelled as world B -- or, if the new fetch
    // fails, world A's data displayed alongside an error for world B.
    setData(null);
    setError(null);
    setAbsent(false);
    setIsLoading(true);

    Promise.resolve()
      .then(() => fetcher(controller.signal))
      .then((res) => {
        // Late response from a superseded request: drop it silently.
        if (!isCurrent()) return;
        setData(res);
        setAbsent(false);
        setIsLoading(false);
      })
      .catch((err) => {
        // A superseded request's failure must not overwrite current state.
        if (!isCurrent()) return;
        // No target selected: not a failure, and explicitly not "empty data".
        if (err === NOT_REQUESTED) {
          setData(null);
          setError(null);
          setAbsent(false);
          setIsLoading(false);
          return;
        }
        // An abort is our own doing (a newer request superseded this one), so
        // it is not a user-visible error.
        if (controller.signal.aborted || (err as Error)?.name === "AbortError") {
          return;
        }
        setData(null);
        setError(err instanceof Error ? err : new Error(String(err)));
        // A 404 is the backend saying "this does not exist", which is an
        // honest empty state. Every other failure is an error the user must
        // be able to see and retry.
        setAbsent(isApiError(err) && err.code === "not_found");
        setIsLoading(false);
      });
    return () => {
      // Mark this generation obsolete first, then abort the transport.
      if (generationRef.current === generation) generationRef.current++;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, trigger]);

  return { data, isLoading, error, absent, refetch };
}

export function useWorlds() {
  return useApi(async (signal) => {
    const res = await listWorlds({ signal });
    return res.rows;
  });
}

export function useSessions(worldId?: string | null) {
  return useApi(async (signal) => {
    const res = await listSessions({ worldId, signal });
    return res.rows;
  }, [worldId]);
}

export function useEvidence(sessionId?: string | null) {
  return useApi(async (signal) => {
    const res = await listEvidence({ sessionId: sessionId || undefined, signal });
    return res.rows;
  }, [sessionId]);
}

/**
 * The world's current WorldStore WorldIR.
 *
 * `absent` distinguishes "never reconstructed" (404 → honest empty) from
 * "could not reach the API" (error → retry). With no world selected the hook
 * stays idle rather than resolving `null`, so callers never confuse
 * "nothing selected" with "backend said there is nothing".
 */
export function useWorldIR(worldId: string | null) {
  return useApi(
    (signal) => (worldId ? fetchWorldIR(worldId, signal) : Promise.reject(NOT_REQUESTED)),
    [worldId],
  );
}

export function useWorldPoints(worldId: string | null) {
  return useApi(
    (signal) => (worldId ? fetchWorldPoints(worldId, signal) : Promise.reject(NOT_REQUESTED)),
    [worldId],
  );
}

export function useWorldCameras(worldId: string | null) {
  return useApi(
    (signal) => (worldId ? fetchWorldCameras(worldId, signal) : Promise.reject(NOT_REQUESTED)),
    [worldId],
  );
}

export function useWorldReport(worldId: string | null) {
  return useApi(
    (signal) => (worldId ? fetchWorldReport(worldId, signal) : Promise.reject(NOT_REQUESTED)),
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
      parentVersionId: v.parent_version_id ?? null,
      label: v.created_at ?? v.id,
      createdAt: v.created_at ?? v.id,
      changeSummary: `${v.changed_entity_ids?.length ?? 0} entities changed`,
      isCurrent: v.is_current ?? false,
    })) satisfies WorldVersionRow[];
  }, [worldId]);
}
