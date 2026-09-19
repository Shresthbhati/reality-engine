'use client';

/**
 * Deterministic capture-quality assessment from actual pixel data.
 *
 * Every verdict ships with machine-checkable reasons. When we cannot know
 * something (no GPS, no prior frame), the result is null/UNKNOWN — it is
 * never silently promoted to a positive value.
 */
import type { FrameQuality, FrameVerdict, QualityReason } from './types';

export const QUALITY_THRESHOLDS = {
  /** Laplacian variance below this = blurry handheld frame. */
  sharpnessMin: 120,
  /** Mean luma bounds for usable exposure. */
  exposureMin: 0.08,
  exposureMax: 0.92,
  /** Fraction of clipped pixels above which the frame is unusable. */
  clippedMax: 0.08,
  /** Structural difference below this (vs previous accepted) = redundant. */
  redundantDiffMax: 0.02,
  /** Difference above this vs previous = likely new viewpoint (useful). */
  novelDiffMin: 0.12,
} as const;

const ASSESS_SIZE = 160;

/** Decode a blob to a small grayscale buffer; throws on undecodable input. */
async function blobToGrayPixels(
  blob: Blob,
): Promise<{ data: Uint8ClampedArray; w: number; h: number }> {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, ASSESS_SIZE / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('OffscreenCanvas 2d context unavailable');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    gray[j] = (d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000;
  }
  return { data: gray, w, h };
}

/** Laplacian variance (sharpness) over grayscale pixels. */
function laplacianVariance(g: Uint8ClampedArray, w: number, h: number): number {
  const vals: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      vals.push(4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]);
    }
  }
  if (vals.length === 0) return 0;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  return vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
}

/** Mean absolute difference 0..1 between two same-size gray buffers. */
function grayDiff(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  if (a.length !== b.length) return 1;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / (a.length * 255);
}

export interface AssessInput {
  blob: Blob;
  /** Gray pixel buffer of the previous frame (or null for first). */
  previousGray: { data: Uint8ClampedArray; w: number; h: number } | null;
}

export interface AssessResult {
  quality: FrameQuality;
  verdict: FrameVerdict;
  reasons: QualityReason[];
  /** Gray buffer (for the next frame's redundancy check). */
  gray: { data: Uint8ClampedArray; w: number; h: number };
}


const T = QUALITY_THRESHOLDS;

/**
 * The triage decision, isolated from pixel decoding so it is deterministic and
 * directly testable: metrics in, verdict + machine-checkable reasons out.
 * Blur or exposure failure dominates; a frame that duplicates the previous one
 * is REDUNDANT rather than REJECTED (duplicates are not bad evidence, they are
 * unnecessary evidence — the distinction the field user acts on).
 */
export function classifyQuality(metrics: FrameQuality): { verdict: FrameVerdict; reasons: QualityReason[] } {
  const { sharpness, exposure, clippedRatio, diffVsPrevious } = metrics;
  const reasons: QualityReason[] = [];
  const blurry = sharpness < T.sharpnessMin;
  const badExposure =
    exposure < T.exposureMin || exposure > T.exposureMax || clippedRatio > T.clippedMax;

  if (diffVsPrevious !== null && diffVsPrevious < T.redundantDiffMax) {
    reasons.push('duplicate_of_previous');
    if (blurry) reasons.push('sharpness_low');
    if (badExposure) reasons.push(exposure > 0.5 ? 'overexposed' : 'underexposed');
    return { verdict: 'REDUNDANT', reasons };
  }
  if (blurry || badExposure) {
    if (blurry) reasons.push('sharpness_low');
    if (badExposure) reasons.push(exposure > 0.5 ? 'overexposed' : 'underexposed');
    return { verdict: 'REJECTED', reasons };
  }
  reasons.push('sharpness_ok', 'exposure_ok');
  if (diffVsPrevious !== null && diffVsPrevious < T.novelDiffMin) {
    reasons.push('similar_to_previous');
  } else if (diffVsPrevious === null) {
    reasons.push('no_reference_available');
  }
  return { verdict: 'USEFUL', reasons };
}

export async function assessFrame(input: AssessInput): Promise<AssessResult> {
  const { data: gray, w, h } = await blobToGrayPixels(input.blob);

  // Exposure stats from real pixels.
  let sum = 0;
  let clipped = 0;
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i] / 255;
    sum += v;
    if (v >= 0.99 || v <= 0.01) clipped++;
  }
  const quality: FrameQuality = {
    sharpness: laplacianVariance(gray, w, h),
    exposure: sum / gray.length,
    clippedRatio: clipped / gray.length,
    diffVsPrevious: input.previousGray !== null ? grayDiff(gray, input.previousGray.data) : null,
  };

  const { verdict, reasons } = classifyQuality(quality);

  return { quality, verdict, reasons, gray: { data: gray, w, h } };
}

/** Human-glanceable verdict line, derived from computed state only. */
export function verdictMessage(verdict: FrameVerdict, reasons: QualityReason[]): string {
  switch (verdict) {
    case 'USEFUL':
      return reasons.includes('similar_to_previous')
        ? 'Useful — small change from previous frame; vary your angle more.'
        : 'Useful evidence. Keep moving to a new viewpoint.';
    case 'REDUNDANT':
      return 'You don’t need more footage here — this frame duplicates the previous one.';
    case 'REJECTED':
      if (reasons.includes('sharpness_low') && reasons.includes('overexposed'))
        return 'Rejected: blurry and overexposed. Hold steady, reduce exposure.';
      if (reasons.includes('sharpness_low') && reasons.includes('underexposed'))
        return 'Rejected: blurry and too dark. Hold steady, add light.';
      if (reasons.includes('sharpness_low'))
        return 'Rejected: motion blur. Hold steady and tap to refocus.';
      return 'Rejected: exposure unusable. Adjust light before capturing.';
    case 'UNASSESSED':
      return 'Not yet assessed.';
  }
}
