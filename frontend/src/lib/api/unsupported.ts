/**
 * Analysis / Results / Reports have no backend service in this build.
 *
 * Rather than a mock array, each of these exposes an explicit `unsupported`
 * result. Pages render the reason; nothing pretends data exists.
 */

export interface UnsupportedResource {
  available: false;
  reason: string;
  items: never[];
}

export function unsupportedResource(what: string): UnsupportedResource {
  return {
    available: false,
    reason: `${what} is not served by the Reality Engine API yet — no mock data is shown.`,
    items: [],
  };
}

/** Explicit unsupported resource for the Analysis section (no backend yet). */
export function unsupportedForAnalysis(): UnsupportedResource {
  return unsupportedResource("Analysis");
}
