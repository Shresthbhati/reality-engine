/**
 * The one calm hint shown after a batch of photos (plan T-5.4). Pure, so it is testable under `node --test`.
 *
 * It comes from the real capture-coverage API (never from a guess), is shown at most once, is silent after the person dismissed it, and
 * is never shown before a model exists. "No nagging" is structural: `shown` holds every key already displayed this session.
 */
import type { CaptureCoverageDto } from "@/lib/api/capture-coverage-model";

export interface HintState {
  /** keys of hints already shown this session */
  shown: string[];
  /** key of the hint the person dismissed most recently */
  dismissedKey: string | null;
}

export interface CaptureHint {
  key: string;
  text: string;
  tone: "info" | "good";
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

export function pickHint(dto: CaptureCoverageDto | null | undefined, state: HintState): CaptureHint | null {
  if (!dto || !dto.available) return null;
  const ok = (key: string) => !state.shown.includes(key) && state.dismissedKey !== key;
  const early = dto.provisional ? " (early estimate)" : "";

  for (const h of dto.hints ?? []) {
    if (ok(h)) return { key: h, text: `${h}${early}`, tone: "info" };
  }
  if ((dto.hints ?? []).length > 0) return null;   // the backend had advice and it was already given or dismissed: no second, generic one

  const weak = dto.summary?.weak ?? dto.walls.filter((w) => w.status === "weak").length;
  const missing = dto.summary?.not_seen ?? dto.walls.filter((w) => w.status === "not_seen").length;
  if (dto.walls.length > 0 && weak === 0 && missing === 0) {
    const key = "complete";
    return ok(key) ? { key, text: `Every wall has been seen${early}. You can stop here, or add more photos for detail.`, tone: "good" } : null;
  }
  const n = weak + missing;
  if (n > 0) {
    const key = `status:${weak}:${missing}`;
    const what = missing > 0 && weak > 0 ? "seen weakly or not at all" : missing > 0 ? "not seen yet" : "only seen weakly";
    return ok(key)
      ? { key, text: `${n} ${plural(n, "wall is", "walls are")} ${what}. A few more photos from a different position would help${early}.`, tone: "info" }
      : null;
  }
  return null;
}
