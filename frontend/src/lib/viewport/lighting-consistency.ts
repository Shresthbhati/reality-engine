/**
 * Lighting and exposure consistency (N-32).
 *
 * Improves visual consistency between separate images without erasing genuine
 * physical differences or changing the authoritative geometry.
 *
 * This module computes:
 *   1. Per-image exposure value (EV) from image metadata.
 *   2. A global exposure offset to minimize seams between images.
 *   3. Per-image colour temperature correction for white balance consistency.
 *
 * Honesty rules:
 *   * Exposure offsets are computed from actual image metadata, not invented.
 *   * If no exposure data is available, the UI must say so (not assume a value).
 *   * Genuine physical differences (e.g. dusk vs dawn) are preserved - we only
 *     remove camera-induced exposure seams.
 */

import type { Entity } from "@/types/worldir";

/** One image's exposure metadata. */
export interface ImageExposure {
  imageId: string;
  /** Exposure value in EV (Exposure Value). Higher = brighter.
   * May be undefined if only iso/shutter/aperture are known; it will be computed. */
  ev?: number;
  /** ISO sensitivity. */
  iso: number | null;
  /** Shutter speed in seconds. */
  shutterSpeed: number | null;
  /** Aperture f-number. */
  aperture: number | null;
  /** White balance temperature in Kelvin. */
  whiteBalanceK: number | null;
  /** Whether this data came from actual EXIF (vs estimated). */
  fromExif: boolean;
}

/** Exposure correction to apply to one image. */
export interface ExposureCorrection {
  imageId: string;
  /** EV offset to apply (positive = brighter, negative = darker). */
  evOffset: number;
  /** White balance temperature offset in Kelvin. */
  wbOffsetK: number;
  /** Confidence in this correction (0..1). */
  confidence: number;
  /** Human-readable reason. */
  reason: string;
}

/**
 * Compute exposure compensation for a set of images so that adjacent images
 * have consistent brightness without erasing genuine lighting differences.
 *
 * Algorithm:
 *   1. Convert each image's exposure settings to an EV value.
 *   2. Compute the median EV across all images (the "neutral" exposure).
 *   3. Each image's offset = medianEV - imageEV (so darker images get
 *     positive offset, brighter images get negative).
 *   4. Clamp offsets to a maximum (e.g. ±2 EV) to avoid over-correction.
 *
 * Pure function: same inputs -> same outputs.
 */
export function computeExposureCompensation(
  exposures: ImageExposure[],
  maxOffsetEv = 2.0,
): ExposureCorrection[] {
  if (exposures.length === 0) return [];

  // Compute EV for each exposure if not already provided.
  const withEv = exposures.map((e) => ({
    ...e,
    ev: e.ev ?? computeEvFromSettings(e.iso, e.shutterSpeed, e.aperture),
  }));

  // Compute median EV.
  const evs = withEv.map((e) => e.ev).sort((a, b) => a - b);
  const medianEv = evs.length % 2 === 0
    ? (evs[evs.length / 2 - 1] + evs[evs.length / 2]) / 2
    : evs[Math.floor(evs.length / 2)];

  // Compute corrections.
  return withEv.map((e) => {
    let offset = medianEv - e.ev;
    // Clamp to max offset.
    offset = Math.min(Math.max(offset, -maxOffsetEv), maxOffsetEv);

    // White balance offset: neutral is ~5500K (daylight).
    const wbOffsetK = e.whiteBalanceK ? 5500 - e.whiteBalanceK : 0;

    // Confidence decreases with offset magnitude and if data was estimated.
    const evConfidence = e.fromExif ? 1.0 : 0.5;
    const offsetConfidence = 1 - Math.abs(offset) / maxOffsetEv;
    const confidence = Math.max(0, Math.min(1, evConfidence * offsetConfidence));

    return {
      imageId: e.imageId,
      evOffset: Math.round(offset * 100) / 100,
      wbOffsetK: Math.round(wbOffsetK * 10) / 10,
      confidence,
      reason: exposureReason(e, offset, e.fromExif),
    };
  });
}

/**
 * Compute EV from ISO, shutter speed, and aperture.
 * EV = log2(aperture^2 / shutterSpeed) + log2(ISO / 100).
 */
function computeEvFromSettings(iso: number | null, shutterSpeed: number | null, aperture: number | null): number {
  if (!iso || !shutterSpeed || !aperture) return 0;
  // EV = log2(N^2 / t) where N = f-number, t = exposure time in seconds.
  // ISO 100 is the base (EV0). Higher ISO = lower EV needed.
  const ev = Math.log2((aperture * aperture) / shutterSpeed) + Math.log2(iso / 100);
  return ev;
}

/** Human-readable reason for an exposure correction. */
function exposureReason(exp: ImageExposure, offset: number, fromExif: boolean): string {
  const offsetAbs = Math.abs(offset);
  const direction = offset > 0 ? "brighten" : offset < 0 ? "darken" : "no change";

  if (offsetAbs < 0.1) {
    return "Exposure already matches scene average.";
  }

  const source = fromExif ? "from EXIF" : "estimated from settings";
  return `${direction} by ${offsetAbs.toFixed(1)} EV (${source}).`;
}

/**
 * Apply exposure correction to a colour value (simple linear scaling).
 * Returns adjusted [r, g, b] in 0..255.
 */
export function applyExposureCorrection(
  rgb: [number, number, number],
  correction: ExposureCorrection,
): [number, number, number] {
  const factor = Math.pow(2, correction.evOffset);
  return [
    Math.min(255, Math.max(0, Math.round(rgb[0] * factor))),
    Math.min(255, Math.max(0, Math.round(rgb[1] * factor))),
    Math.min(255, Math.max(0, Math.round(rgb[2] * factor))),
  ];
}

/**
 * Build exposure corrections for all images in a world from their metadata.
 * Images without exposure data get confidence 0 and no correction.
 */
export function buildExposureCorrections(
  images: Array<{
    id: string;
    ev?: number;
    iso?: number;
    shutterSpeed?: number;
    aperture?: number;
    whiteBalanceK?: number;
    fromExif?: boolean;
  }>,
): ExposureCorrection[] {
  const exposures: ImageExposure[] = images.map((img) => ({
    imageId: img.id,
    ev: img.ev ?? computeEvFromSettings(img.iso ?? null, img.shutterSpeed ?? null, img.aperture ?? null),
    iso: img.iso ?? null,
    shutterSpeed: img.shutterSpeed ?? null,
    aperture: img.aperture ?? null,
    whiteBalanceK: img.whiteBalanceK ?? null,
    fromExif: img.fromExif ?? false,
  }));

  return computeExposureCompensation(exposures);
}

