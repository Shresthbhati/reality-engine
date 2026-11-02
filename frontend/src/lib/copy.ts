/**
 * Consumer wording. Simple-mode screens take every engine term from here; a contract test fails if a consumer string
 * contains an engine term. API, route and type names are unchanged -- only what the user reads is renamed.
 * Pro mode may show the raw engine terms.
 */

/** Engine terms that must not appear in Simple-mode copy. */
export const BANNED_IN_SIMPLE = [
  "WorldIR",
  "WorldStore",
  "COLMAP",
  "ENU",
  "HEAD",
  "coordinate frame",
  "registration",
  "provenance",
  "session",
  "evidence",
  "entity",
  "backend",
] as const;

export const NAV = {
  home: "Home",
  places: "All places",
  newPlace: "New place",
  settings: "Settings",
  sessions: "Sessions",
  evidence: "Evidence",
} as const;

/** The Home library and place cards. */
export const HOME = {
  title: "Your places",
  newPlace: "New place",
  emptyTitle: "Turn photos into a 3D place",
  loadError: "Couldn't load your places.",
  engineHint: "Is the Reality Engine running? Start it with: uvicorn apps.api.main:app --port 8100",
  retry: "Try again",
  rename: "Rename",
  delete: "Delete",
  deleteTitle: "Delete this place?",
  deleteBody: "Its model and version history are removed from the app. Your original photos' files are kept on disk.",
  deleteType: "Type the place name to confirm",
  deleteBusy: "A build is running for this place. Wait for it to finish or cancel it, then delete.",
  cancel: "Cancel",
  building: "Building",
  noPhotosYet: "No photos yet",
} as const;

export const HOME_STEPS = ["Walk around and take photos", "Drop them here", "Explore it, then add more photos to improve it"] as const;

/** Steps of a build, as far as the engine reports them. "Waiting" is queued; the three working steps are real stage
 *  groups (see apps/api/jobs.py STAGE_PROGRESS). The UI never shows a step the engine does not report. */
export const BUILD_STEPS = {
  waiting: "Waiting to start",
  reading: "Reading your photos",
  building: "Building the 3D place",
  saving: "Saving this version",
} as const;

export const BUILD = {
  title: "Building your place",
  leave: "You can leave this page. We'll keep building, and your place will be on Home.",
  cancel: "Stop building",
  elapsed: "Elapsed",
  doneTitle: "Your place is ready",
  partialTitle: "Ready, but some photos weren't used",
  failedTitle: "Couldn't build this place",
  failedSafe: "Nothing was lost. Your photos are kept, and you can try again.",
  tryAgain: "Try again",
  open: "Open place",
  home: "Back to Home",
  coarse: "Progress moves in steps because the engine reports stages, not a smooth percentage.",
} as const;

/** The place viewer (Simple mode). */
export const VIEWER = {
  back: "Home",
  openPro: "Pro mode",
  resetView: "Reset view",
  quality: "Quality",
  qualityHide: "Hide quality colours",
  qualityShow: "Show quality colours",
  legend: "Quality colours",
  loading: "Opening your place…",
  noModelTitle: "No 3D model yet",
  noModelBody: "This place has photos but no model. Add more photos from different positions, or try building again.",
  loadError: "Couldn't open this place.",
  selected: "Selected",
  builtFrom: "Built from",
  photosShown: "Photos from this place",
  traceNone: "This part was generated, not built from photos.",
  traceSession: "Built from this place's photos. The engine did not record which ones.",
  traceObserved: "Seen in these photos",
  closeDrawer: "Close details",
  statusOk: "Looks good",
  statusBuilding: "Building a new version…",
  cardTitle: "Place quality",
  cardPhotos: "Photos placed",
  cardSize: "Size",
  cardSizeNo: "Not real size yet, so measurements are not in metres.",
  cardSizeYes: "Real size is set.",
  cardWeak: "Weak parts",
  cardWeakNone: "None found",
  cardUnknown: "Not enough photos to judge",
  cardConflicts: "Parts where photos disagree",
  cardShow: "Show on model",
  cardHide: "Hide colours",
} as const;

/** Why a place has its overall quality band; keys match band_reason from the quality endpoint. */
export const BAND_REASON: Record<string, string> = {
  few_placed: "Fewer than half of your photos could be placed.",
  rough_model: "Built from a single view, so the model is rough.",
  partial_build: "The last build finished with warnings.",
  some_unplaced: "Some of your photos could not be placed.",
};

/** Result of the pre-flight photo check. */
export const CHECK = {
  title: "Check your photos",
  checking: "Checking your photos…",
  nameLabel: "Name this place",
  namePlaceholder: "e.g. Front of the library",
  build: "Build place",
  back: "Choose different photos",
  continueAnyway: "Build anyway",
  remove: "Remove",
  verdict: {
    good: "Good set. These photos should build well.",
    has_problems: "Some photos may cause trouble. You can remove them or build anyway.",
    add_angles: "Add a few more angles if you can. 8 or more photos from different positions build much better.",
    no_usable_photos: "None of these can be used. Choose different photos.",
  },
  problems: {
    soft: "Looks blurry compared with the rest",
    duplicate: "Same photo as",
    tooSmall: "Too small to use",
    notPhoto: "Looks like a drawing or screenshot, not a photo",
    unreadable: "Can't be opened as a photo",
    ok: "Looks fine",
    hasLocation: "Has a location",
  },
  unavailable: "Couldn't check these photos, so they are uploaded as they are.",
} as const;

export type ProvenanceKey = "OBSERVED" | "RECONSTRUCTED" | "ESTIMATED" | "INFERRED" | "GENERATED" | "UNKNOWN" | "CONFLICT";

export const PROVENANCE: Record<ProvenanceKey, string> = {
  OBSERVED: "Seen in photos",
  RECONSTRUCTED: "Built from several photos",
  ESTIMATED: "Measured estimate",
  INFERRED: "Best guess",
  GENERATED: "Added, not from photos",
  UNKNOWN: "Not enough photos",
  CONFLICT: "Photos disagree",
};

export type QualityBand = "good" | "fair" | "weak" | "unknown";

/** A model score is not a probability (master doc 14.2), so the UI shows a band, never the number. */
export const QUALITY: Record<QualityBand, string> = {
  good: "Good",
  fair: "Fair",
  weak: "Weak",
  unknown: "Unknown",
};

export const SCALE = {
  relative: "Not real size yet",
  metric: "Real size",
  setCta: "Set real size",
} as const;

export const JOB = {
  partial: "Ready, but some photos weren't used",
  failed: "Couldn't build this version. Your previous version is safe.",
} as const;

export function photosPlaced(placed: number, total: number): string {
  return `${placed} of ${total} photo${total === 1 ? "" : "s"} placed`;
}

export function versionLabel(n: number, current: boolean): string {
  return current ? `Version ${n} · Current` : `Version ${n}`;
}

/** Every consumer string above, flattened, for the banned-term test. */
export function allConsumerStrings(): string[] {
  return [
    ...Object.values(NAV),
    ...Object.values(PROVENANCE),
    ...Object.values(QUALITY),
    ...Object.values(SCALE),
    ...Object.values(JOB),
    ...Object.values(HOME),
    ...Object.values(VIEWER),
    ...Object.values(BAND_REASON),
    ...HOME_STEPS,
    ...Object.values(BUILD_STEPS),
    ...Object.values(BUILD),
    ...Object.values(CHECK).flatMap((v) => (typeof v === "string" ? [v] : Object.values(v))),
    photosPlaced(7, 10),
    versionLabel(3, true),
  ];
}
