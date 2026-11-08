/**
 * Reality Engine 3D WorldIR Viewport Controller.
 * Production Three.js WebGL engine consuming authentic pipeline artifacts.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type {
  CamerasPayload,
  WorldIR,
} from "@/types/worldir";
import { pointsBounds, robustPointsBounds } from "./loaders";
import {
  parseSplatBuffer,
  parsePlySplatBuffer,
  type ParsedSplatCloud,
  type HybridRenderMode,
} from "./splat-model";
import { isOversizedEntity } from "./oversized";
import { atlasUrl, buildRoomShellParts, confidenceRamp, partMaterial, DEFAULT_CONF_PALETTE, type ConfidencePalette, type ShellKind, type ShellPart, type ShellRenderMode } from "./room-shell-model";
import { centroid, cutawayHides, dollyForFov, nearestCenter, parseHexColor, upAxisOf, ORTHO_FOV, PERSPECTIVE_FOV } from "./view-model";
import { shouldAutoFrame } from "./autoframe";
import { buildOutdoor, derivePalette, emptyOutdoorResult, meshBounds, partMesh, type OutdoorPart, type OutdoorResult } from "./outdoor-model";
import type { MeasureKind, Vec3 } from "./measure-model";
import { STRUCTURE_COLORS, choosePointDisplay, decimatePoints, type PointDisplayMode, type StructurePoint } from "./structure-points-model";
import { TransformGizmoController } from "../studio/transform-gizmo.ts";
import { extractRoomBoundariesFromWorld } from "../studio/room-boundary-snap.ts";
import { DEFAULT_SNAP_CONFIG, type TransformGizmoMode, type SnapConfig } from "../studio/transform-types.ts";
import {
  DEFAULT_BODY_RADIUS,
  DEFAULT_EYE_HEIGHT,
  DEFAULT_WALK_SPEED,
  clampPitch,
  classifyTouchRegion,
  computeTouchJoystick,
  computeWalkDelta,
  resolveWallCollision,
  type Obstacle,
} from "./walk-model";

export interface ViewportLayers {
  points: boolean;
  /** show the whole dense cloud instead of the structure points (or the decimated cloud when there are none); off by default */
  fullCloud: boolean;
  entities: boolean;
  cameras: boolean;
  mesh: boolean;
  oversized: boolean;
  uncertainty: boolean;
  walls: boolean;
  topology: boolean;
  /** the real room shell (floor, walls with window openings, ceiling layers, furniture) built from measured WorldIR */
  shell: boolean;
  /** 3D Gaussian Splats photorealistic capture layer */
  splats?: boolean;
}

export type { HybridRenderMode };

export type RenderMode = ShellRenderMode;

export type ViewPreset = "isometric" | "top" | "front" | "side";

export interface MeasurementResult {
  p1: [number, number, number];
  p2: [number, number, number];
  distance: number;
  delta: [number, number, number];
}

/** A saved measurement drawn as a labelled 3D annotation. */
export interface SavedMeasurementOverlay {
  id: string;
  kind: MeasureKind;
  points: Vec3[];
  label: string;
}

export interface ViewportStatistics {
  points: number;
  canonicalPoints: number;
  renderedPoints: number;
  isLODSampled: boolean;
  /** what the point layer shows: structure points, the full cloud, a decimated cloud, or nothing */
  pointMode: PointDisplayMode;
  cameras: number;
  entities: number;
  isContextLost: boolean;
}

export const TYPE_COLORS: Record<string, number> = {
  building: 0x64748b, // slate-500
  level: 0x38bdf8,    // sky-400
  room: 0x3b82f6,     // blue-500
  corridor: 0x06b6d4, // cyan-500
  wall: 0xf59e0b,     // amber-500
  floor: 0x3b82f6,    // blue-500
  ceiling: 0x8b5cf6,  // purple-500
  roof: 0x6366f1,     // indigo-500
  door: 0x10b981,     // emerald-500
  window: 0x06b6d4,   // cyan-400
  stairs: 0xec4899,   // pink-500
  stair: 0xec4899,    // pink-500
  column: 0xeab308,   // yellow-500
  beam: 0xd97706,     // amber-600
  object: 0x14b8a6,   // teal-500
  furniture: 0x84cc16,// lime-500
  default: 0x94a3b8,  // slate-400
};

/**
 * Visual weight for entities whose confidence was never recorded.
 *
 * This is a rendering constant, not a measurement: it decides how solid a
 * box looks, and is never displayed, colour-coded as uncertainty, or written
 * back into WorldIR. Entities with a recorded confidence use that number.
 */
const UNRECORDED_RENDER_WEIGHT = 0.5;

function renderWeightFor(confidence: unknown): number {
  return typeof confidence === "number" && Number.isFinite(confidence)
    ? confidence
    : UNRECORDED_RENDER_WEIGHT;
}

type RGB = [number, number, number];

/** Scene colours come from the :root CSS tokens (globals.css); the fallbacks are the same values. */
interface ScenePalette { bg: RGB; gridMajor: RGB; gridMinor: RGB; text: RGB; textSecondary: RGB; accent: RGB; conf: ConfidencePalette }

function readPalette(): ScenePalette {
  const css = typeof document !== "undefined" ? getComputedStyle(document.documentElement) : null;
  const get = (name: string, fb: RGB): RGB => parseHexColor(css?.getPropertyValue(name)) ?? fb;
  return {
    bg: get("--bg-base", [23, 25, 24]),
    gridMajor: get("--border", [59, 65, 59]),
    gridMinor: get("--border-subtle", [46, 51, 46]),
    text: get("--text-primary", [236, 237, 230]),
    textSecondary: get("--text-secondary", [162, 166, 154]),
    accent: get("--accent", [197, 138, 96]),
    conf: {
      low: get("--conf-low", DEFAULT_CONF_PALETTE.low),
      mid: get("--warning", DEFAULT_CONF_PALETTE.mid),
      high: get("--conf-high", DEFAULT_CONF_PALETTE.high),
      unknown: get("--conf-unknown", DEFAULT_CONF_PALETTE.unknown),
    },
  };
}
const hex = (c: RGB) => (c[0] << 16) | (c[1] << 8) | c[2];

/** One merged mesh + edge set for outdoor parts that share an opacity tier (imported map data is slightly muted, generated is translucent). */
interface OutdoorBatch {
  tier: "normal" | "imported" | "generated";
  solid: THREE.Mesh;
  wire: THREE.LineSegments;
  /** global part indexes, and per local part: index range, wire-float range, vertex range */
  partIdx: number[];
  idxStart: number[]; idxCount: number[];
  wStart: number[]; wCount: number[];
  vStart: number[]; vCount: number[];
  fullIndex: Uint32Array;
  wirePos: Float32Array;
  wireCol: Float32Array;
  realistic: Float32Array;
  confidence: Float32Array;
  /** after the hidden-entity pass: cumulative triangle ends and the global part each active range belongs to (picking) */
  activeTriEnd: number[];
  activePart: number[];
}

export type PointerTool = "select" | "pan" | "orbit" | "walk";
export interface CameraInfo {
  azimuthDeg: number;
  elevationDeg: number;
  /** camera right / up in world space, for the axis gizmo */
  right: RGB;
  up: RGB;
  /** world metres per CSS pixel at the orbit target (scale bar) */
  metersPerPx: number;
}

export class WorldSceneController {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;

  // Scene layers
  private layers = {
    points: null as THREE.Points | null,
    entities: null as THREE.Group | null,
    cameras: null as THREE.Group | null,
    mesh: null as THREE.Mesh | null,
    selectionOutline: null as THREE.LineSegments | null,
    topology: null as THREE.Group | null,
    shell: null as THREE.Group | null,
    splats: null as THREE.InstancedMesh | null,
  };

  private outdoor: OutdoorResult = emptyOutdoorResult();
  private outdoorGroup: THREE.Group | null = null;
  private outdoorBatches: OutdoorBatch[] = [];
  private outdoorBoxes = new Map<string, THREE.Box3>();
  private outdoorBox: THREE.Box3 | null = null;
  private outdoorOutline: THREE.LineSegments | null = null;
  private distanceCap = 200;
  private gridCustom = false;
  private shellParts: ShellPart[] = [];
  /** atlas uri -> loaded texture, "loading", or "failed" (a failed atlas falls back to the flat/striped material) */
  private atlasCache = new Map<string, THREE.Texture | "loading" | "failed">();
  private shellSolid: THREE.Object3D[] = [];
  private shellWire: THREE.Object3D[] = [];
  private renderMode: RenderMode = "solid";
  // Gaussian Splats & Hybrid Viewport
  private splatCloud: ParsedSplatCloud | null = null;
  private splatOpacity = 1.0;
  private splatScaleMultiplier = 1.0;
  private hybridMode: HybridRenderMode = "hybrid";
  /**
   * All world content lives under `root`. Scene space is Y-up; a world compiled Z-up (metadata.compiled_from.up) is
   * stood upright by rotating this group, so data coordinates (measurements, saved points) stay in the data frame.
   */
  private root = new THREE.Group();
  private palette: ScenePalette = readPalette();
  private shellMeta: Array<{ center: RGB; roomCenter: RGB }> = [];
  private cutaway = false;
  private hiddenEntityIds = new Set<string>();
  /** entities whose every shell part is currently hidden (cutaway / explicit hide): their picking proxy is skipped */
  private shellHiddenEntities = new Set<string>();
  private selectionEnabled = true;
  private pointerTool: PointerTool = "select";
  private walkYaw = 0;
  private walkPitch = 0;
  private isPointerDraggingWalk = false;
  private lastWalkPointer = { x: 0, y: 0 };
  private walkKeysDown = { forward: false, backward: false, left: false, right: false };
  private walkTouchMove: { forward: number; strafe: number } | null = null;
  private walkTouchState = {
    moveTouchId: null as number | null,
    lookTouchId: null as number | null,
    startMoveX: 0,
    startMoveY: 0,
    lastLookX: 0,
    lastLookY: 0,
  };
  private lastWalkFrameTime = 0;
  private dimGroup: THREE.Group | null = null;
  private showDimensions = false;

  private entityMeshes = new Map<string, THREE.Mesh>();
  private selectedEntityId: string | null = null;
  private isolatedSpaceId: string | null = null;
  private activeLevelIndex: number | null = null;
  private previewCorrection: { entityId: string; type?: string; confidence?: number; originalPosition?: THREE.Vector3 } | null = null;
  private robustExtent = 1.0;
  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private abortController = new AbortController();

  // WebGL Context Loss & LOD State
  private isContextLost = false;
  private renderedPointsCount = 0;
  private pointMode: PointDisplayMode = "none";
  private structurePoints: StructurePoint[] = [];
  /** the photos could not be placed in one space: per-photo boxes and shell are not drawn as a room */
  private suppressLegacy = false;

  // Data cache
  private world: WorldIR | null = null;
  private framedWorldId: string | null = null;
  private pointsData: Float32Array | null = null;
  private camerasData: CamerasPayload | null = null;
  private meshData: { positions: Float32Array; indices: Uint32Array } | null = null;

  private layerVisibility: ViewportLayers = {
    points: true,
    fullCloud: false,
    entities: true,
    cameras: true,
    mesh: true,
    oversized: false,
    uncertainty: false,
    walls: true,
    topology: true,
    shell: true,
  };

  private transformGizmo: TransformGizmoController | null = null;
  private transformGizmoEnabled = true;
  private transformMode: TransformGizmoMode = "translate";
  private gizmoSnapConfig: SnapConfig = DEFAULT_SNAP_CONFIG;

  private onSelectCallback?: (id: string | null) => void;
  private onHoverCallback?: (id: string | null) => void;
  private onMeasureCallback?: (res: MeasurementResult | null) => void;
  private onCameraChangeCallback?: (info: CameraInfo) => void;
  private onContextLossChangeCallback?: (isLost: boolean) => void;

  private cameraAnimation: {
    startPos: THREE.Vector3;
    endPos: THREE.Vector3;
    startTarget: THREE.Vector3;
    endTarget: THREE.Vector3;
    startTime: number;
    duration: number;
  } | null = null;
  private lastAzimuthDeg = -1;
  private lastElevationDeg = -1;
  private lastMppKey = -1;

  // Spatial context & measurement tools
  private gridHelper: THREE.GridHelper | null = null;
  private axesHelper: THREE.AxesHelper | null = null;
  private isMeasurementMode = false;
  private measureTool: MeasureKind = "distance";
  private measureSnap = true;
  private measurePts: THREE.Vector3[] = [];
  private measureDone = false;
  private onMeasurePointsCallback?: (s: { tool: MeasureKind; points: Vec3[]; done: boolean } | null) => void;
  private measurementGroup: THREE.Group | null = null;
  private annotationGroup: THREE.Group | null = null;
  private annotationsVisible = true;
  private savedMeasurements: SavedMeasurementOverlay[] = [];
  private selectedMeasurementId: string | null = null;

  constructor(container: HTMLElement) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(hex(this.palette.bg));

    // Grid & Coordinate Frame
    this.gridHelper = new THREE.GridHelper(20, 20, hex(this.palette.gridMajor), hex(this.palette.gridMinor));
    this.gridHelper.position.y = -0.01;
    this.scene.add(this.gridHelper);

    // Subtle axes indicator
    this.axesHelper = new THREE.AxesHelper(1.5);
    (this.axesHelper.material as THREE.Material).depthTest = false;
    this.axesHelper.renderOrder = 1;
    this.axesHelper.visible = false; // the HUD axis gizmo replaces it
    this.scene.add(this.axesHelper);

    this.scene.add(this.root);

    // Measurement group
    this.measurementGroup = new THREE.Group();
    this.root.add(this.measurementGroup);
    this.annotationGroup = new THREE.Group();
    this.root.add(this.annotationGroup);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(6, 14, 5);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(hex(this.palette.textSecondary), 0.4);
    dirLight2.position.set(-6, -8, -5);
    this.scene.add(dirLight2);

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;

    this.camera = new THREE.PerspectiveCamera(50, w / Math.max(1, h), 0.05, 5000);
    this.camera.position.set(4, 3, 6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.screenSpacePanning = true;
    this.controls.maxDistance = 200;
    this.controls.minDistance = 0.2;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.setupEvents();
    this.initTransformGizmo();
    this.startLoop();
  }

  public setCallbacks(callbacks: {
    onSelect?: (id: string | null) => void;
    onHover?: (id: string | null) => void;
    onMeasure?: (res: MeasurementResult | null) => void;
    onCameraChange?: (info: CameraInfo) => void;
    onContextLossChange?: (isLost: boolean) => void;
  }) {
    this.onSelectCallback = callbacks.onSelect;
    this.onHoverCallback = callbacks.onHover;
    this.onMeasureCallback = callbacks.onMeasure;
    this.onCameraChangeCallback = callbacks.onCameraChange;
    this.onContextLossChangeCallback = callbacks.onContextLossChange;
  }

  private initTransformGizmo() {
    try {
      this.transformGizmo = new TransformGizmoController({
        scene: this.scene,
        camera: this.camera,
        domElement: this.renderer.domElement,
        orbitControls: this.controls,
        snapConfig: this.gizmoSnapConfig,
        onTransformChange: (entityId, state) => {
          if (this.layers.selectionOutline && this.selectedEntityId === entityId) {
            this.layers.selectionOutline.position.set(...state.position);
            this.layers.selectionOutline.rotation.set(...state.rotation);
            this.layers.selectionOutline.scale.set(...state.scale);
          }
          if (this.world && this.world.entities && this.world.entities[entityId]) {
            const ent = this.world.entities[entityId];
            if (!ent.transform) {
              ent.transform = {};
            }
            ent.transform.position = { x: state.position[0], y: state.position[1], z: state.position[2] };
            ent.transform.scale = { x: state.scale[0], y: state.scale[1], z: state.scale[2] };
          }
        },
      });
      this.transformGizmo.setMode(this.transformMode);
      if (this.world) {
        this.transformGizmo.setRoomBoundaries(extractRoomBoundariesFromWorld(this.world));
      }
    } catch (e) {
      console.warn("Failed to initialize TransformGizmoController:", e);
    }
  }

  public setTransformMode(mode: TransformGizmoMode) {
    this.transformMode = mode;
    this.transformGizmo?.setMode(mode);
  }

  public getTransformMode(): TransformGizmoMode {
    return this.transformMode;
  }

  public setGizmoSnapConfig(config: Partial<SnapConfig>) {
    this.gizmoSnapConfig = { ...this.gizmoSnapConfig, ...config };
    this.transformGizmo?.setSnapConfig(config);
  }

  public getGizmoSnapConfig(): SnapConfig {
    return { ...this.gizmoSnapConfig };
  }

  public setTransformGizmoEnabled(enabled: boolean) {
    this.transformGizmoEnabled = enabled;
    if (!enabled) {
      this.transformGizmo?.detach();
    } else if (this.selectedEntityId && this.entityMeshes.has(this.selectedEntityId)) {
      this.transformGizmo?.attach(this.selectedEntityId, this.entityMeshes.get(this.selectedEntityId)!);
    }
  }

  public getTransformGizmo(): TransformGizmoController | null {
    return this.transformGizmo;
  }

  public setGridVisible(visible: boolean) {
    if (this.gridHelper) this.gridHelper.visible = visible;
  }

  public setAxesVisible(visible: boolean) {
    if (this.axesHelper) this.axesHelper.visible = visible;
  }

  public setMeasurementMode(active: boolean) {
    this.isMeasurementMode = active;
    if (!active) {
      this.clearMeasurement();
    }
  }

  public setMeasureTool(tool: MeasureKind) {
    if (tool === this.measureTool) return;
    this.measureTool = tool;
    this.clearMeasurement();
  }

  public setMeasureSnap(on: boolean) {
    this.measureSnap = on;
  }

  public setMeasurePointsCallback(cb?: (s: { tool: MeasureKind; points: Vec3[]; done: boolean } | null) => void) {
    this.onMeasurePointsCallback = cb;
  }

  private disposeGroupChildren(group: THREE.Group | null) {
    if (!group) return;
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Sprite) {
        if (!(obj instanceof THREE.Sprite)) obj.geometry.dispose();
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((m) => {
          (m as THREE.SpriteMaterial).map?.dispose();
          m.dispose();
        });
      }
    }
  }

  public clearMeasurement() {
    this.measurePts = [];
    this.measureDone = false;
    this.disposeGroupChildren(this.measurementGroup);
    this.onMeasureCallback?.(null);
    this.onMeasurePointsCallback?.(null);
  }

  /** Close the area polygon (needs >= 3 points). */
  public finishMeasurement() {
    if (this.measureTool === "area" && this.measurePts.length >= 3 && !this.measureDone) {
      this.measureDone = true;
      this.redrawMeasurement();
    }
  }

  /** Remove the last picked point of an unfinished measurement. */
  public undoMeasurementPoint() {
    if (this.measureDone || this.measurePts.length === 0) return;
    this.measurePts.pop();
    this.redrawMeasurement();
  }

  public setAnnotationsVisible(visible: boolean) {
    this.annotationsVisible = visible;
    if (this.annotationGroup) this.annotationGroup.visible = visible;
  }

  /** Draw saved measurements as labelled 3D annotations (line/polygon + label sprite). */
  public setSavedMeasurements(items: SavedMeasurementOverlay[], selectedId: string | null) {
    this.savedMeasurements = items;
    this.selectedMeasurementId = selectedId;
    const group = this.annotationGroup;
    if (!group) return;
    this.disposeGroupChildren(group);
    group.visible = this.annotationsVisible;
    for (const it of items) {
      const selected = it.id === selectedId;
      const color = selected ? 0xffd166 : 0x35d07f;
      const pts = it.points.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const geo = new THREE.BufferGeometry().setFromPoints(it.kind === "area" ? [...pts, pts[0]] : pts);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, depthTest: false }));
      line.renderOrder = 990;
      group.add(line);
      const c = new THREE.Vector3();
      if (it.kind === "angle" && pts.length >= 2) {
        c.copy(pts[1]); // position at angle vertex
      } else {
        pts.forEach((p) => c.add(p));
        c.multiplyScalar(1 / pts.length);
      }
      group.add(this.makeLabelSprite(it.label, c, selected));
    }
  }

  private makeLabelSprite(text: string, at: THREE.Vector3, selected: boolean): THREE.Sprite {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 112;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "rgba(8,14,26,0.88)";
      ctx.fillRect(0, 0, 512, 112);
      ctx.strokeStyle = selected ? "#ffd166" : "#35d07f";
      ctx.lineWidth = 6;
      ctx.strokeRect(3, 3, 506, 106);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 44px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 256, 58, 490);
    }
    const mat = new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, sizeAttenuation: false });
    const sprite = new THREE.Sprite(mat);
    sprite.position.copy(at);
    sprite.scale.set(0.3, 0.3 * (112 / 512), 1);
    sprite.renderOrder = 1000;
    return sprite;
  }

  public focusMeasurement(id: string) {
    const it = this.savedMeasurements.find((m) => m.id === id);
    if (!it || it.points.length === 0) return;
    const n = it.points.length;
    this.flyToPosition(
      it.points.reduce((s, p) => s + p[0], 0) / n,
      it.points.reduce((s, p) => s + p[1], 0) / n,
      it.points.reduce((s, p) => s + p[2], 0) / n,
    );
  }

  /** Snap a picked point to the nearest corner, else edge, of a visible entity's axis-aligned bounding box. */
  private snapToEntityBoxes(hit: THREE.Vector3): THREE.Vector3 {
    const thresh = Math.max(0.05, this.camera.position.distanceTo(hit) * 0.03);
    let best: THREE.Vector3 | null = null;
    let bestD = thresh;
    let bestIsCorner = false;
    const box = new THREE.Box3();
    const tmp = new THREE.Vector3();
    const line = new THREE.Line3();
    for (const mesh of this.entityMeshes.values()) {
      if (!mesh.visible) continue;
      box.setFromObject(mesh);
      if (box.isEmpty()) continue;
      const { min, max } = box;
      const corners: THREE.Vector3[] = [];
      for (let i = 0; i < 8; i++) {
        corners.push(new THREE.Vector3(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z));
      }
      for (const c of corners) {
        const d = c.distanceTo(hit);
        if (d < bestD) { best = c; bestD = d; bestIsCorner = true; }
      }
      if (bestIsCorner) continue;
      for (let i = 0; i < 8; i++) {
        for (let b = 0; b < 3; b++) {
          const j = i ^ (1 << b);
          if (j < i) continue;
          line.set(corners[i], corners[j]);
          line.closestPointToPoint(hit, true, tmp);
          const d = tmp.distanceTo(hit);
          if (d < bestD) { best = tmp.clone(); bestD = d; }
        }
      }
    }
    return best ?? hit;
  }

  private redrawMeasurement() {
    const group = this.measurementGroup;
    if (!group) return;
    this.disposeGroupChildren(group);
    const pts = this.measurePts;
    pts.forEach((p, i) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: i === 0 ? 0x00e5ff : 0x35d07f, depthTest: false }),
      );
      m.position.copy(p);
      m.renderOrder = 999;
      group.add(m);
    });
    if (pts.length >= 2) {
      const loop = this.measureTool === "area" && this.measureDone;
      const geo = new THREE.BufferGeometry().setFromPoints(loop ? [...pts, pts[0]] : pts);
      const line = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: 0x00e5ff, dashSize: 0.1, gapSize: 0.05, depthTest: false }));
      line.computeLineDistances();
      line.renderOrder = 998;
      group.add(line);
    }
    const triples = pts.map((p) => [p.x, p.y, p.z] as Vec3);
    const need = this.measureTool === "area" || this.measureTool === "angle" ? 3 : 2;
    const complete = this.measureTool === "area" ? this.measureDone : pts.length >= (this.measureTool === "angle" ? 3 : 2);
    this.onMeasurePointsCallback?.(triples.length ? { tool: this.measureTool, points: triples, done: complete && triples.length >= need } : null);
    if (this.measureTool !== "area" && this.measureTool !== "angle" && pts.length >= 1) {
      // Legacy two-point result (bottom bar). distance 0 until the second point is placed.
      const a = pts[0], b = pts[1] ?? pts[0];
      this.onMeasureCallback?.({
        p1: [a.x, a.y, a.z],
        p2: [b.x, b.y, b.z],
        distance: pts.length >= 2 ? a.distanceTo(b) : 0,
        delta: [b.x - a.x, b.y - a.y, b.z - a.z],
      });
    }
  }

  public loadWorldData(
    world: WorldIR,
    points: Float32Array | null,
    cameras: CamerasPayload | null,
    mesh: { positions: Float32Array; indices: Uint32Array } | null = null,
    opts: { structurePoints?: StructurePoint[] | null; suppressLegacy?: boolean } = {}
  ) {
    this.structurePoints = opts.structurePoints ?? [];
    this.suppressLegacy = Boolean(opts.suppressLegacy);
    this.cameraAnimation = null;
    if (this.selectedEntityId !== null) {
      this.selectedEntityId = null;
      this.onSelectCallback?.(null);
    }
    this.isolatedSpaceId = null;
    this.world = world;
    if (this.transformGizmo && world) {
      this.transformGizmo.setRoomBoundaries(extractRoomBoundariesFromWorld(world));
    }
    this.pointsData = points;
    this.camerasData = cameras;
    this.meshData = mesh;
    this.root.rotation.x = upAxisOf(world) === "z" ? -Math.PI / 2 : 0;
    this.root.updateMatrixWorld(true);

    this.rebuildScene();
  }

  private disposePointLayer() {
    this.renderedPointsCount = 0;
    if (this.layers.points) {
      this.root.remove(this.layers.points);
      this.layers.points.geometry.dispose();
      (this.layers.points.material as THREE.Material).dispose();
      this.layers.points = null;
    }
  }

  /**
   * The point layer, by `choosePointDisplay`: the structure points (room corners, opening corners, dimension-defining
   * points) by default; the whole cloud only when `fullCloud` is on; otherwise, with no structure points, a cloud
   * decimated to DECIMATED_CAP points. `frame` re-aims the camera (a toggle must not move the view).
   */
  private buildPointLayer(frame: boolean) {
    this.disposePointLayer();
    const total = this.pointsData ? this.pointsData.length / 3 : 0;
    this.pointMode = choosePointDisplay({ structureCount: this.structurePoints.length, fullCloud: this.layerVisibility.fullCloud, totalPoints: total });
    if (this.pointsData && total > 0) this.robustExtent = robustPointsBounds(this.pointsData).extent;
    if (this.pointMode === "none") return;

    let positions: Float32Array;
    let colors: Float32Array;
    let material: THREE.PointsMaterial;
    if (this.pointMode === "structure") {
      positions = new Float32Array(this.structurePoints.length * 3);
      colors = new Float32Array(positions.length);
      const c = new THREE.Color();
      this.structurePoints.forEach((p, i) => {
        positions.set(p.position, i * 3);
        c.setHex(STRUCTURE_COLORS[p.kind]);
        colors.set([c.r, c.g, c.b], i * 3);
      });
      // screen-space markers: legible at any zoom, drawn on top of the shell
      material = new THREE.PointsMaterial({ size: 9, sizeAttenuation: false, vertexColors: true, depthTest: false, transparent: true });
    } else {
      // Practical interactive WebGL rendering limit for dense point clouds
      const MAX_RENDER_POINTS = 1_500_000;
      const src = this.pointsData as Float32Array;
      if (this.pointMode === "decimated") {
        positions = decimatePoints(src);
      } else if (total > MAX_RENDER_POINTS) {
        const stride = Math.ceil(total / MAX_RENDER_POINTS);
        positions = new Float32Array(Math.floor(total / stride) * 3);
        let dst = 0;
        for (let i = 0; i < total; i += stride) {
          positions[dst++] = src[i * 3];
          positions[dst++] = src[i * 3 + 1];
          positions[dst++] = src[i * 3 + 2];
        }
      } else {
        positions = src;
      }
      const b = pointsBounds(positions);
      const span = Math.max(1e-6, b.max[1] - b.min[1]);
      colors = new Float32Array(positions.length);
      for (let i = 0; i < positions.length; i += 3) {
        const t = (positions[i + 1] - b.min[1]) / span; // y in [0, 1]
        colors[i] = 0.25 + 0.65 * t; // r
        colors[i + 1] = 0.85; // g
        colors[i + 2] = 1.0 - 0.45 * t; // b
      }
      material = new THREE.PointsMaterial({ size: this.pointMode === "decimated" ? 0.07 : 0.045, vertexColors: true, sizeAttenuation: true });
    }
    this.renderedPointsCount = positions.length / 3;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.layers.points = new THREE.Points(geo, material);
    this.layers.points.renderOrder = this.pointMode === "structure" ? 10 : 0;
    this.layers.points.visible = this.layerVisibility.points;
    this.root.add(this.layers.points);

    if (frame) {
      // Frame on the structure when there is one, else on the robust bounds of the canonical cloud
      const rb = robustPointsBounds(this.pointMode === "structure" ? positions : (this.pointsData as Float32Array));
      this.camera.position.set(rb.center[0] + rb.extent * 1.3, rb.center[1] + rb.extent * 0.8, rb.center[2] + rb.extent * 1.3);
      this.controls.target.set(rb.center[0], rb.center[1], rb.center[2]);
    }
  }

  private clearSceneLayers() {
    this.disposePointLayer();
    this.disposeSplats();
    if (this.layers.entities) {
      this.root.remove(this.layers.entities);
      for (const mesh of this.entityMeshes.values()) {
        mesh.traverse((child) => {
          if (child instanceof THREE.Mesh || child instanceof THREE.Line || child instanceof THREE.LineSegments) {
            child.geometry.dispose();
            if (Array.isArray(child.material)) {
              child.material.forEach((m) => m.dispose());
            } else {
              child.material.dispose();
            }
          }
        });
      }
      this.entityMeshes.clear();
      this.layers.entities = null;
    }
    if (this.layers.shell) {
      this.root.remove(this.layers.shell);
      this.layers.shell.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
      });
      this.layers.shell = null;
      this.shellSolid = [];
      this.shellWire = [];
    }
    this.clearOutdoorLayer();
    if (this.layers.cameras) {
      this.root.remove(this.layers.cameras);
      this.layers.cameras.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.Line || child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      this.layers.cameras = null;
    }
    if (this.layers.mesh) {
      this.root.remove(this.layers.mesh);
      this.layers.mesh.geometry.dispose();
      (this.layers.mesh.material as THREE.Material).dispose();
      this.layers.mesh = null;
    }
    if (this.layers.selectionOutline) {
      this.root.remove(this.layers.selectionOutline);
      this.layers.selectionOutline.geometry.dispose();
      (this.layers.selectionOutline.material as THREE.Material).dispose();
      this.layers.selectionOutline = null;
    }
    if (this.layers.topology) {
      this.root.remove(this.layers.topology);
      this.layers.topology.traverse((child) => {
        if (child instanceof THREE.Line || child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      this.layers.topology = null;
    }
  }

  public rebuildScene() {
    this.clearSceneLayers();
    this.disposeDimensions();

    // 1. Build the point layer (structure points by default, the full cloud on request)
    this.buildPointLayer(true);

    // 1.5 Real room shell from measured interior semantics (replaces the faint per-entity boxes where it applies).
    // Not drawn when the photos could not be placed in one space: it would present unplaced geometry as a room.
    this.atlasCache.clear();
    this.shellParts = this.suppressLegacy ? [] : buildRoomShellParts(this.world);
    this.buildShell();
    this.buildOutdoorLayer();

    // 2. Build Entities (the per-entity legacy boxes are skipped for the same reason)
    if (this.world && this.world.entities && !this.suppressLegacy) {
      this.layers.entities = new THREE.Group();
      const geometries = this.world.geometries || {};

      for (const e of Object.values(this.world.entities)) {
        const geomId = (e.geometry_ids || [])[0];
        const g = geometries[geomId];
        if (!g || !g.bounds_min || !g.bounds_max) continue;

        const mn = [g.bounds_min.x, g.bounds_min.y, g.bounds_min.z];
        const mx = [g.bounds_max.x, g.bounds_max.y, g.bounds_max.z];
        const size = [
          Math.max(0.02, mx[0] - mn[0]),
          Math.max(0.02, mx[1] - mn[1]),
          Math.max(0.02, mx[2] - mn[2]),
        ];
        if (size.some((s) => !isFinite(s))) continue;

        const extent = Math.max(size[0], size[1], size[2]);
        // The oversized filter targets depth-noise outliers relative to a
        // point cloud; with no cloud (procedural worlds) there is no scale
        // to judge against, so nothing is "oversized" (a 20 m city block is real).
        const hasCloud = Boolean(this.pointsData && this.pointsData.length > 0);
        const isOversized = isOversizedEntity(extent, hasCloud, this.robustExtent);
        const outlierScale = isOversized ? 0.25 : 1.0;

        const baseColor = TYPE_COLORS[e.type.toLowerCase()] || TYPE_COLORS.default;
        const conf = typeof e.confidence === "number" && Number.isFinite(e.confidence) ? e.confidence : null;

        // In uncertainty mode, color shifts from green (1.0) to amber/red (0.0) if confidence is recorded
        let renderColor = baseColor;
        if (this.layerVisibility.uncertainty && conf !== null) {
          const hue = conf * 0.33; // 0.33 is green, 0 is red
          const hslColor = new THREE.Color().setHSL(hue, 0.9, 0.5);
          renderColor = hslColor.getHex();
        }

        const isSpaceVolume = e.type === "room" || e.type === "corridor" || e.type === "space";
        const isOpenOpening = e.type === "door" || e.type === "window";
        const baseOpacity = isSpaceVolume
          ? 0.05
          : Math.max(0.12, (0.15 + 0.35 * renderWeightFor(conf)) * outlierScale);

        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(size[0], size[1], size[2]),
          new THREE.MeshLambertMaterial({
            color: renderColor,
            transparent: true,
            opacity: baseOpacity,
            depthWrite: false,
          })
        );

        mesh.position.set(
          (mn[0] + mx[0]) / 2,
          (mn[1] + mx[1]) / 2,
          (mn[2] + mx[2]) / 2
        );
        mesh.userData = {
          coveredByShell: this.isCoveredByShell(e),
          entityId: e.id,
          entityType: e.type.toLowerCase(),
          confidence: conf,
          isOversized,
          baseColor,
        };

        // Add bounding wireframe for structure clarity (openings have higher accent opacity)
        const wireGeo = new THREE.EdgesGeometry(mesh.geometry);
        const wireMat = new THREE.LineBasicMaterial({
          color: renderColor,
          transparent: true,
          opacity: isOpenOpening ? 0.85 : isSpaceVolume ? 0.6 : 0.35 * outlierScale,
        });
        const wireframe = new THREE.LineSegments(wireGeo, wireMat);
        mesh.add(wireframe);

        this.layers.entities.add(mesh);
        this.entityMeshes.set(e.id, mesh);
      }

      this.root.add(this.layers.entities);
      this.updateMeshVisibilities();

      // Procedural worlds have no cloud to seed the camera; frame once per world so a large scene is on-screen.
      const hasCloud = Boolean(this.pointsData && this.pointsData.length > 0);
      if (shouldAutoFrame(hasCloud, this.entityMeshes.size + this.outdoor.parts.length, this.framedWorldId, this.world.id)) {
        this.framedWorldId = this.world.id;
        this.frameAll(false);
      }
    }

    // 3. Build Camera Frustums
    if (this.camerasData && this.camerasData.cameras && this.camerasData.cameras.length > 0) {
      this.layers.cameras = new THREE.Group();
      const positions: number[] = [];
      const markerGeo = new THREE.SphereGeometry(0.04, 12, 12);
      const markerMat = new THREE.MeshBasicMaterial({ color: 0x35d07f });
      const quat = new THREE.Quaternion();
      const imgSize = this.camerasData.image_size || [1280, 960];
      const cam = new THREE.PerspectiveCamera(55, imgSize[0] / imgSize[1], 0.01, 10);
      const corner = new THREE.Vector3();
      const dist = 0.55;

      for (const c of this.camerasData.cameras) {
        const p = c.position_m;
        quat.set(
          c.rotation_wxyz[1],
          c.rotation_wxyz[2],
          c.rotation_wxyz[3],
          c.rotation_wxyz[0]
        );
        cam.position.set(p[0], p[1], p[2]);
        cam.quaternion.copy(quat);
        cam.updateMatrixWorld(true);

        const frustum: [number, number, number][] = [];
        for (const [nx, ny] of [
          [-1, -1],
          [1, -1],
          [1, 1],
          [-1, 1],
        ]) {
          corner.set(nx, ny, 0.5).unproject(cam);
          const d = corner.sub(cam.position).normalize().multiplyScalar(dist);
          frustum.push([p[0] + d.x, p[1] + d.y, p[2] + d.z]);
        }

        for (let i = 0; i < 4; i++) {
          positions.push(p[0], p[1], p[2], frustum[i][0], frustum[i][1], frustum[i][2]);
          const j = (i + 1) % 4;
          positions.push(
            frustum[i][0],
            frustum[i][1],
            frustum[i][2],
            frustum[j][0],
            frustum[j][1],
            frustum[j][2]
          );
        }

        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.set(p[0], p[1], p[2]);
        marker.userData = { evidenceId: c.evidence_id || c.id };
        this.layers.cameras.add(marker);
      }

      const frustumGeo = new THREE.BufferGeometry();
      frustumGeo.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(positions), 3)
      );
      const frustumLines = new THREE.LineSegments(
        frustumGeo,
        new THREE.LineBasicMaterial({
          color: 0x35d07f,
          transparent: true,
          opacity: 0.55,
        })
      );
      this.layers.cameras.add(frustumLines);

      this.layers.cameras.visible = this.layerVisibility.cameras;
      this.root.add(this.layers.cameras);
    }

    // 4. Build Reconstructed Mesh (if available)
    if (this.meshData && this.meshData.positions.length > 0) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(this.meshData.positions, 3));
      geo.setIndex(new THREE.BufferAttribute(this.meshData.indices, 1));
      geo.computeVertexNormals();

      this.layers.mesh = new THREE.Mesh(
        geo,
        new THREE.MeshLambertMaterial({
          color: 0x8fb8d8,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.6,
        })
      );
      this.layers.mesh.visible = this.layerVisibility.mesh;
      this.root.add(this.layers.mesh);
    }

    // Reselect previous entity if still exists
    if (this.selectedEntityId) {
      this.selectEntity(this.selectedEntityId);
    } else if (this.layerVisibility.topology) {
      this.renderSpatialTopology(null);
    }
  }

  private isCoveredByShell(e: { type: string; semantic_labels?: string[] }): boolean {
    if (!this.shellParts.length) return false;
    return ["room", "wall", "floor", "ceiling", "window"].includes(e.type) ||
      (e.semantic_labels ?? []).includes("furniture_candidate");
  }

  private buildShell() {
    this.shellSolid = [];
    this.shellWire = [];
    if (!this.shellParts.length) return;
    const group = new THREE.Group();
    // thin warm-light architectural lines; furniture in the accent so it reads apart from the structure
    const WIRE: Record<ShellKind, number> = {
      floor: hex(this.palette.textSecondary), wall: hex(this.palette.text), ceiling: hex(this.palette.textSecondary),
      false_ceiling: hex(this.palette.textSecondary), ceiling_step: hex(this.palette.textSecondary),
      window_glass: hex(this.palette.textSecondary), window_frame: hex(this.palette.text), furniture: hex(this.palette.accent),
    };
    this.shellMeta = [];
    const floorCenters = this.shellParts.filter((p) => p.kind === "floor").map((p) => this.toScene(centroid(p.positions) as RGB));
    for (const part of this.shellParts) {
      const center = this.toScene(centroid(part.positions) as RGB);
      this.shellMeta.push({ center, roomCenter: nearestCenter(center, floorCenters) as RGB });
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(part.positions, 3));
      geo.setIndex(part.indices);
      if (part.texture) geo.setAttribute("uv", new THREE.Float32BufferAttribute(part.texture.uvs, 2));
      geo.computeVertexNormals();
      const solid = new THREE.Mesh(geo, this.shellMaterial(part));
      solid.userData = { entityId: part.entityId, shellKind: part.kind };
      const wire = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 25), new THREE.LineBasicMaterial({
        color: WIRE[part.kind], transparent: part.kind === "window_glass", opacity: part.kind === "window_glass" ? 0.8 : 1,
      }));
      wire.userData = { entityId: part.entityId, shellKind: part.kind };
      group.add(solid, wire);
      this.shellSolid.push(solid);
      this.shellWire.push(wire);
    }
    group.visible = this.layerVisibility.shell;
    this.layers.shell = group;
    this.root.add(group);
    this.applyRenderMode();
    this.buildDimensions();
  }

  /**
   * One material per part per mode. Geometry is never touched: modes differ only in this material (colour, stripe
   * overlay) and in which of the solid / edge objects are visible.
   */
  /** The loaded atlas for a part, or null while loading / after a failure (then the part uses its flat/striped fallback). */
  private atlasFor(part: ShellPart): THREE.Texture | null {
    const uri = part.texture?.uri;
    if (!uri) return null;
    const hit = this.atlasCache.get(uri);
    if (hit && hit !== "loading" && hit !== "failed") return hit;
    if (hit === undefined) {
      const url = atlasUrl(uri, this.world?.id);
      if (!url) { this.atlasCache.set(uri, "failed"); return null; }
      this.atlasCache.set(uri, "loading");
      new THREE.TextureLoader().load(
        url,
        (tex) => {
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.anisotropy = 4;
          this.atlasCache.set(uri, tex);
          this.applyRenderMode();
        },
        undefined,
        () => { this.atlasCache.set(uri, "failed"); this.applyRenderMode(); },
      );
    }
    return null;
  }

  private shellMaterial(part: ShellPart): THREE.MeshLambertMaterial {
    // Textured only once its atlas is actually loaded; until then / on failure the honest flat (striped) fallback shows.
    const atlas = part.texture && this.renderMode !== "confidence" ? this.atlasFor(part) : null;
    const m = partMaterial(atlas ? part : { ...part, texture: undefined }, this.renderMode, this.palette.conf);
    if (atlas) {
      // alpha=0 atlas texels are unobserved: they stay transparent (alphaTest), never inpainted or tinted
      return new THREE.MeshLambertMaterial({
        map: atlas, color: 0xffffff, side: THREE.DoubleSide, alphaTest: 0.02, transparent: m.opacity < 1, opacity: m.opacity,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
    }
    const color = new THREE.Color(m.color[0] / 255, m.color[1] / 255, m.color[2] / 255).convertSRGBToLinear();
    const mat = new THREE.MeshLambertMaterial({
      color, side: THREE.DoubleSide, transparent: m.opacity < 1, opacity: m.opacity,
      depthWrite: m.opacity >= 1, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    if (m.hatch) {
      // diagonal dark stripes in world space (0.25 m period): "not measured" / "inferred" overlay
      mat.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader
          .replace("#include <common>", "#include <common>\nvarying vec3 vHatchPos;")
          .replace("#include <begin_vertex>", "#include <begin_vertex>\nvHatchPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
        shader.fragmentShader = shader.fragmentShader
          .replace("#include <common>", "#include <common>\nvarying vec3 vHatchPos;")
          .replace("#include <dithering_fragment>",
            "#include <dithering_fragment>\nif (fract((vHatchPos.x + vHatchPos.y + vHatchPos.z) / 0.25) < 0.4) gl_FragColor.rgb *= 0.45;");
      };
      mat.customProgramCacheKey = () => "shell-hatch";
    }
    return mat;
  }

  // ---- outdoor layer (map-imported / generated buildings, roads, rail, bridges, street furniture, trees) ----------

  private clearOutdoorLayer() {
    if (this.outdoorOutline) {
      this.root.remove(this.outdoorOutline);
      this.outdoorOutline.geometry.dispose();
      (this.outdoorOutline.material as THREE.Material).dispose();
      this.outdoorOutline = null;
    }
    if (this.outdoorGroup) {
      this.root.remove(this.outdoorGroup);
      this.outdoorGroup.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
      });
      this.outdoorGroup = null;
    }
    this.outdoorBatches = [];
    this.outdoorBoxes.clear();
    this.outdoorBox = null;
    this.outdoor = emptyOutdoorResult();
    if (this.gridCustom) {
      this.setGridExtent(20, 20, 0, 0);
      this.gridCustom = false;
    }
    this.distanceCap = 200;
    this.controls.maxDistance = 200 * this.fovScale();
  }

  private setGridExtent(size: number, divisions: number, cx: number, cz: number) {
    const visible = this.gridHelper ? this.gridHelper.visible : true;
    if (this.gridHelper) {
      this.scene.remove(this.gridHelper);
      this.gridHelper.geometry.dispose();
      (this.gridHelper.material as THREE.Material).dispose();
    }
    this.gridHelper = new THREE.GridHelper(size, divisions, hex(this.palette.gridMajor), hex(this.palette.gridMinor));
    this.gridHelper.position.set(cx, -0.01, cz);
    this.gridHelper.visible = visible;
    this.scene.add(this.gridHelper);
  }

  private wireColorFor(part: OutdoorPart): RGB {
    const p = this.palette;
    return part.group === "furniture" || part.group === "trees" || part.group === "rail" || part.group === "hedges" ? p.accent
      : part.group === "roads" || part.group === "gardens" || part.group === "flowerbeds" || part.group === "woodland" || part.group === "water" ? p.textSecondary : p.text;
  }

  private buildOutdoorLayer() {
    const pal = this.palette;
    this.outdoor = buildOutdoor(this.world, derivePalette({
      bg: pal.bg, gridMajor: pal.gridMajor, text: pal.text, textSecondary: pal.textSecondary, accent: pal.accent, high: pal.conf.high,
    }));
    const parts = this.outdoor.parts;
    if (!parts.length) return;
    const up = this.outdoor.up;
    const srgb = (c: RGB): RGB => { const k = new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255).convertSRGBToLinear(); return [k.r, k.g, k.b]; };
    const group = new THREE.Group();
    const tiers = ["normal", "imported", "generated"] as const;
    const world = new THREE.Box3();
    for (const tier of tiers) {
      const idxs = parts.map((_, i) => i).filter((i) => {
        const q = parts[i];
        const t = q.generated ? "generated" : q.imported ? "imported" : "normal";
        return t === tier;
      });
      if (!idxs.length) continue;
      const pos: number[] = [], index: number[] = [], wpos: number[] = [], wcol: number[] = [], col: number[] = [], ccol: number[] = [];
      const b: OutdoorBatch = {
        tier, solid: null as never, wire: null as never, partIdx: [], idxStart: [], idxCount: [], wStart: [], wCount: [], vStart: [], vCount: [],
        fullIndex: new Uint32Array(0), wirePos: new Float32Array(0), wireCol: new Float32Array(0), realistic: new Float32Array(0), confidence: new Float32Array(0),
        activeTriEnd: [], activePart: [],
      };
      for (const gi of idxs) {
        const part = parts[gi];
        const m = partMesh(part, up);
        if (!m.indices.length) continue;
        const vOff = pos.length / 3;
        const bb = meshBounds(m.positions);
        if (bb) {
          const box = new THREE.Box3(new THREE.Vector3(...bb.min), new THREE.Vector3(...bb.max));
          this.outdoorBoxes.set(part.entityId, box);
          world.union(box);
        }
        b.partIdx.push(gi);
        b.vStart.push(vOff); b.vCount.push(m.positions.length / 3);
        b.idxStart.push(index.length); b.idxCount.push(m.indices.length);
        for (const v of m.positions) pos.push(v);
        for (const i of m.indices) index.push(i + vOff);
        const real = srgb(part.color);
        const real2 = part.color2 ? srgb(part.color2) : real; // roof slopes / tree trunks
        const conf = srgb(part.confidence === null ? pal.conf.unknown : confidenceRamp(part.confidence, pal.conf));
        for (let v = 0; v < m.positions.length / 3; v++) { col.push(...(m.tint[v] ? real2 : real)); ccol.push(...conf); }
        // edges only (same warm-light line style as the room wireframe)
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(m.positions, 3));
        g.setIndex(m.indices);
        const edges = new THREE.EdgesGeometry(g, 25);
        const e = edges.getAttribute("position").array as Float32Array;
        const wc = srgb(this.wireColorFor(part));
        b.wStart.push(wpos.length); b.wCount.push(e.length);
        for (let k = 0; k < e.length; k++) wpos.push(e[k]);
        for (let k = 0; k < e.length / 3; k++) wcol.push(...wc);
        g.dispose(); edges.dispose();
      }
      if (!b.partIdx.length) continue;
      b.fullIndex = Uint32Array.from(index);
      b.wirePos = Float32Array.from(wpos);
      b.wireCol = Float32Array.from(wcol);
      b.realistic = Float32Array.from(col);
      b.confidence = Float32Array.from(ccol);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(pos), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(Float32Array.from(col), 3));
      geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(index), 1));
      geo.computeVertexNormals();
      const opacity = tier === "normal" ? 1 : tier === "imported" ? 0.85 : 0.5;
      b.solid = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide, transparent: opacity < 1, opacity, depthWrite: tier !== "generated",
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      }));
      b.solid.frustumCulled = false;
      const wgeo = new THREE.BufferGeometry();
      wgeo.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(wpos), 3));
      wgeo.setAttribute("color", new THREE.BufferAttribute(Float32Array.from(wcol), 3));
      b.wire = new THREE.LineSegments(wgeo, tier === "generated"
        ? new THREE.LineDashedMaterial({ vertexColors: true, dashSize: 1.2, gapSize: 0.8, transparent: true })
        : new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
      b.wire.frustumCulled = false;
      group.add(b.solid, b.wire);
      this.outdoorBatches.push(b);
    }
    group.visible = this.layerVisibility.shell;
    this.outdoorGroup = group;
    this.root.add(group);
    this.root.updateMatrixWorld(true);
    this.outdoorBox = world.isEmpty() ? null : this.robustOutdoorBox(world.clone().applyMatrix4(this.root.matrixWorld));
    this.applyOutdoorVisibility();
    this.applyOutdoorMode();
    if (this.outdoorBox) {
      const sz = this.outdoorBox.getSize(new THREE.Vector3());
      const c = this.outdoorBox.getCenter(new THREE.Vector3());
      const ext = Math.max(sz.x, sz.z, 1);
      const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find((s) => ext / s <= 14) ?? 1000;
      const size = Math.ceil((ext * 1.3) / step) * step;
      this.setGridExtent(size, Math.round(size / step), Math.round(c.x / step) * step, Math.round(c.z / step) * step);
      this.gridCustom = true;
      this.distanceCap = Math.max(200, Math.max(ext, sz.y) * 3);
      this.controls.maxDistance = this.distanceCap * this.fovScale();
    }
  }

  /** Framing box: 2nd-98th percentile of part centres, so one road that runs 800 m past the cluster does not shrink the view. */
  private robustOutdoorBox(full: THREE.Box3): THREE.Box3 {
    if (this.outdoorBoxes.size < 12) return full;
    const c = new THREE.Vector3();
    const xs: number[] = [], zs: number[] = [];
    for (const b of this.outdoorBoxes.values()) {
      b.clone().applyMatrix4(this.root.matrixWorld).getCenter(c);
      xs.push(c.x); zs.push(c.z);
    }
    const q = (a: number[], t: number) => { a.sort((u, v) => u - v); return a[Math.min(a.length - 1, Math.max(0, Math.round(t * (a.length - 1))))]; };
    const box = new THREE.Box3(new THREE.Vector3(q(xs, 0.02), full.min.y, q(zs, 0.02)), new THREE.Vector3(q(xs, 0.98), full.max.y, q(zs, 0.98)));
    const pad = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) * 0.1 + 5;
    box.min.x -= pad; box.max.x += pad; box.min.z -= pad; box.max.z += pad;
    return box.intersect(full);
  }

  /** Keep the depth range proportional to the viewing distance so a 600 m neighbourhood does not z-fight at the far end. */
  private fitDepthRange() {
    const d = this.camera.position.distanceTo(this.controls.target);
    const near = Math.max(0.05, d * 0.004);
    const far = Math.max(5000 * this.fovScale(), d * 40);
    if (Math.abs(near - this.camera.near) > this.camera.near * 0.02 || far !== this.camera.far) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  /** Mode colours + selection tint + outline. Realistic = kind colours; Confidence = green/red/grey; Wireframe = edges only. */
  private applyOutdoorMode() {
    if (!this.outdoorBatches.length) return;
    const accent = (() => { const k = new THREE.Color(this.palette.accent[0] / 255, this.palette.accent[1] / 255, this.palette.accent[2] / 255).convertSRGBToLinear(); return [k.r, k.g, k.b]; })();
    for (const b of this.outdoorBatches) {
      const attr = b.solid.geometry.getAttribute("color") as THREE.BufferAttribute;
      attr.array.set(this.renderMode === "confidence" ? b.confidence : b.realistic);
      b.partIdx.forEach((gi, li) => {
        if (this.outdoor.parts[gi].entityId !== this.selectedEntityId) return;
        for (let v = b.vStart[li]; v < b.vStart[li] + b.vCount[li]; v++) { attr.array[3 * v] = accent[0]; attr.array[3 * v + 1] = accent[1]; attr.array[3 * v + 2] = accent[2]; }
      });
      attr.needsUpdate = true;
      b.solid.visible = this.renderMode !== "wireframe";
      b.wire.visible = this.renderMode !== "solid";
      (b.wire.material as THREE.Material).opacity = this.renderMode === "confidence" ? 0.4 : 1;
    }
    if (this.outdoorOutline) {
      this.root.remove(this.outdoorOutline);
      this.outdoorOutline.geometry.dispose();
      (this.outdoorOutline.material as THREE.Material).dispose();
      this.outdoorOutline = null;
    }
    const sel = this.selectedEntityId;
    if (!sel) return;
    for (const b of this.outdoorBatches) {
      const li = b.partIdx.findIndex((gi) => this.outdoor.parts[gi].entityId === sel);
      if (li < 0) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(b.wirePos.slice(b.wStart[li], b.wStart[li] + b.wCount[li]), 3));
      this.outdoorOutline = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: hex(this.palette.accent), depthTest: false, transparent: true }));
      this.outdoorOutline.renderOrder = 5;
      this.outdoorOutline.frustumCulled = false;
      this.root.add(this.outdoorOutline);
      break;
    }
  }

  /** Re-pack the index / edge buffers without the hidden entities (buffers are preallocated; only the draw range changes). */
  private applyOutdoorVisibility() {
    for (const b of this.outdoorBatches) {
      const idx = b.solid.geometry.getIndex() as THREE.BufferAttribute;
      const wp = b.wire.geometry.getAttribute("position") as THREE.BufferAttribute;
      const wc = b.wire.geometry.getAttribute("color") as THREE.BufferAttribute;
      let io = 0, wo = 0;
      b.activeTriEnd = []; b.activePart = [];
      b.partIdx.forEach((gi, li) => {
        if (this.hiddenEntityIds.has(this.outdoor.parts[gi].entityId)) return;
        (idx.array as Uint32Array).set(b.fullIndex.subarray(b.idxStart[li], b.idxStart[li] + b.idxCount[li]), io);
        io += b.idxCount[li];
        b.activeTriEnd.push(io / 3);
        b.activePart.push(gi);
        (wp.array as Float32Array).set(b.wirePos.subarray(b.wStart[li], b.wStart[li] + b.wCount[li]), wo);
        (wc.array as Float32Array).set(b.wireCol.subarray(b.wStart[li], b.wStart[li] + b.wCount[li]), wo);
        wo += b.wCount[li];
      });
      idx.needsUpdate = true; wp.needsUpdate = true; wc.needsUpdate = true;
      b.solid.geometry.setDrawRange(0, io);
      b.wire.geometry.setDrawRange(0, wo / 3);
      if (b.tier === "generated") b.wire.computeLineDistances();
    }
  }

  private pickOutdoor(): { id: string; distance: number } | null {
    if (!this.outdoorGroup?.visible) return null;
    let best: { id: string; distance: number } | null = null;
    for (const b of this.outdoorBatches) {
      const hit = this.raycaster.intersectObject(b.solid, false)[0];
      if (!hit || hit.faceIndex == null) continue;
      let lo = 0, hi = b.activeTriEnd.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (b.activeTriEnd[mid] > hit.faceIndex) hi = mid; else lo = mid + 1; }
      const gi = b.activePart[lo];
      if (gi === undefined || (best && best.distance <= hit.distance)) continue;
      best = { id: this.outdoor.parts[gi].entityId, distance: hit.distance };
    }
    return best;
  }

  private applyRenderMode() {
    // solid = coloured; wireframe = edges only; both = coloured + edges; confidence = confidence-coloured + faint edges
    this.shellSolid.forEach((o, i) => {
      const mesh = o as THREE.Mesh;
      (mesh.material as THREE.Material).dispose();
      mesh.material = this.shellMaterial(this.shellParts[i]);
    });
    this.refreshShellVisibility();
    this.refreshDimensions();
    this.applyOutdoorMode();
  }

  /** Mode visibility x explicit hides x cutaway (a selected entity is always revealed, even a cut-away wall). */
  public refreshShellVisibility() {
    const camPos: RGB = [this.camera.position.x, this.camera.position.y, this.camera.position.z];
    const total = new Map<string, number>();
    const hidden = new Map<string, number>();
    this.shellParts.forEach((part, i) => {
      const meta = this.shellMeta[i];
      const forced = part.entityId === this.selectedEntityId;
      const userHidden = this.hiddenEntityIds.has(part.entityId);
      const cut = this.cutaway && !forced && !!meta && cutawayHides(part.kind, meta.center, meta.roomCenter, camPos);
      const show = !userHidden && !cut;
      // the mesh arrays are rebuilt with the parts; a render-mode change that lands between the two has nothing to toggle yet
      const solid = this.shellSolid[i], wire = this.shellWire[i];
      if (solid) solid.visible = show && this.renderMode !== "wireframe";
      if (wire) wire.visible = show && this.renderMode !== "solid";
      total.set(part.entityId, (total.get(part.entityId) ?? 0) + 1);
      if (!show) hidden.set(part.entityId, (hidden.get(part.entityId) ?? 0) + 1);
    });
    this.shellHiddenEntities = new Set([...total].filter(([id, n]) => hidden.get(id) === n).map(([id]) => id));
  }

  public setCutaway(on: boolean) {
    this.cutaway = on;
    this.refreshShellVisibility();
  }

  public getCutaway(): boolean {
    return this.cutaway;
  }

  public setHiddenEntities(ids: Set<string> | null) {
    this.hiddenEntityIds = new Set(ids ?? []);
    this.applyOutdoorVisibility();
    this.refreshShellVisibility();
    this.updateMeshVisibilities();
  }

  public setPointerTool(tool: PointerTool) {
    const prevTool = this.pointerTool;
    this.pointerTool = tool;
    this.selectionEnabled = tool === "select";
    if (tool !== "select" && this.transformGizmo) {
      this.transformGizmo.detach();
    } else if (tool === "select" && this.selectedEntityId && this.transformGizmo && this.transformGizmoEnabled) {
      const mesh = this.entityMeshes.get(this.selectedEntityId);
      if (mesh) this.transformGizmo.attach(this.selectedEntityId, mesh);
    }

    if (tool === "walk") {
      this.controls.enabled = false;
      this.initWalkMode();
    } else {
      if (prevTool === "walk") {
        this.controls.enabled = true;
        const dir = new THREE.Vector3();
        this.camera.getWorldDirection(dir);
        this.controls.target.copy(this.camera.position).addScaledVector(dir, 3.0);
        this.controls.update();
      }
      this.controls.mouseButtons.LEFT = tool === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    }
  }

  public getPointerTool(): PointerTool {
    return this.pointerTool;
  }

  private getFloorY(): number {
    for (const p of this.shellParts) {
      if (p.kind === "floor" && p.positions.length >= 3) {
        const pt = this.toScene([p.positions[0], p.positions[1], p.positions[2]]);
        return pt[1];
      }
    }
    if (this.outdoorBox) {
      return this.outdoorBox.min.y;
    }
    if (this.entityMeshes.size > 0) {
      const box = new THREE.Box3();
      for (const mesh of this.entityMeshes.values()) {
        box.expandByObject(mesh);
      }
      return isFinite(box.min.y) ? box.min.y : 0;
    }
    return 0;
  }

  public initWalkMode() {
    const dir = new THREE.Vector3();
    this.camera.getWorldDirection(dir);
    this.walkYaw = Math.atan2(-dir.x, -dir.z);
    this.walkPitch = clampPitch(Math.asin(Math.max(-1, Math.min(1, dir.y))));

    const floorY = this.getFloorY();
    this.camera.position.y = floorY + DEFAULT_EYE_HEIGHT;
    this.updateWalkCameraLook();
  }

  public setWalkPose(x: number, y: number, z: number, yawRad?: number, pitchRad?: number) {
    this.camera.position.set(x, y, z);
    if (yawRad !== undefined) this.walkYaw = yawRad;
    if (pitchRad !== undefined) this.walkPitch = clampPitch(pitchRad);
    this.updateWalkCameraLook();
  }

  public getWalkPose(): { position: [number, number, number]; yawRad: number; pitchRad: number } {
    return {
      position: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      yawRad: this.walkYaw,
      pitchRad: this.walkPitch,
    };
  }

  private updateWalkCameraLook() {
    const dirX = -Math.sin(this.walkYaw) * Math.cos(this.walkPitch);
    const dirY = Math.sin(this.walkPitch);
    const dirZ = -Math.cos(this.walkYaw) * Math.cos(this.walkPitch);
    const target = new THREE.Vector3(
      this.camera.position.x + dirX,
      this.camera.position.y + dirY,
      this.camera.position.z + dirZ
    );
    this.camera.lookAt(target);
  }

  public getObstacles(): Obstacle[] {
    const obstacles: Obstacle[] = [];
    if (this.world) {
      const rooms = extractRoomBoundariesFromWorld(this.world);
      for (const room of rooms) {
        const poly = room.polygon2D;
        if (poly && poly.length >= 2) {
          for (let i = 0; i < poly.length; i++) {
            const p1 = poly[i];
            const p2 = poly[(i + 1) % poly.length];
            obstacles.push({
              p1: [p1[0], p1[1]],
              p2: [p2[0], p2[1]],
            });
          }
        }
      }
    }
    for (const p of this.shellParts) {
      if (p.kind === "wall" && p.positions.length >= 6) {
        for (let i = 0; i < p.positions.length; i += 6) {
          const a = this.toScene([p.positions[i], p.positions[i + 1], p.positions[i + 2]]);
          const b = this.toScene([p.positions[i + 3], p.positions[i + 4], p.positions[i + 5]]);
          obstacles.push({
            p1: [a[0], a[2]],
            p2: [b[0], b[2]],
          });
        }
      }
    }
    return obstacles;
  }

  public applyWalkStep(forward: number, strafe: number, deltaSec: number = 0.016): { x: number; z: number; collided: boolean } {
    if (Math.hypot(forward, strafe) < 0.0001) {
      return { x: this.camera.position.x, z: this.camera.position.z, collided: false };
    }
    const [dx, dz] = computeWalkDelta(forward, strafe, this.walkYaw, DEFAULT_WALK_SPEED, deltaSec);
    const currX = this.camera.position.x;
    const currZ = this.camera.position.z;
    const desiredX = currX + dx;
    const desiredZ = currZ + dz;
    const obstacles = this.getObstacles();
    const resolved = resolveWallCollision(currX, currZ, desiredX, desiredZ, obstacles, DEFAULT_BODY_RADIUS);
    this.camera.position.x = resolved.x;
    this.camera.position.z = resolved.z;
    const floorY = this.getFloorY();
    this.camera.position.y = floorY + DEFAULT_EYE_HEIGHT;
    this.updateWalkCameraLook();
    return resolved;
  }

  /** data-frame point -> scene (Y-up) point */
  private toScene(p: RGB): RGB {
    const v = this.root.localToWorld(new THREE.Vector3(p[0], p[1], p[2]));
    return [v.x, v.y, v.z];
  }

  public setShowDimensions(on: boolean) {
    this.showDimensions = on;
    this.refreshDimensions();
  }

  private disposeDimensions() {
    if (!this.dimGroup) return;
    this.scene.remove(this.dimGroup);
    this.dimGroup.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material as (THREE.Material & { map?: THREE.Texture | null }) | undefined;
      mat?.map?.dispose();
      mat?.dispose();
    });
    this.dimGroup = null;
  }

  private refreshDimensions() {
    if (this.dimGroup) this.dimGroup.visible = this.showDimensions && this.renderMode === "wireframe";
  }

  /** Bounding length / width / height of the measured shell. Built once per shell; shown in wireframe when requested. */
  private buildDimensions() {
    this.disposeDimensions();
    const box = new THREE.Box3();
    for (const p of this.shellParts) {
      if (p.kind !== "floor" && p.kind !== "wall" && p.kind !== "ceiling") continue;
      for (let i = 0; i < p.positions.length; i += 3) box.expandByPoint(this.root.localToWorld(new THREE.Vector3(p.positions[i], p.positions[i + 1], p.positions[i + 2])));
    }
    if (box.isEmpty()) return;
    const g = new THREE.Group();
    const lineMat = new THREE.LineBasicMaterial({ color: hex(this.palette.accent) });
    const o = 0.3;
    const { min, max } = box;
    const dims: Array<{ a: THREE.Vector3; b: THREE.Vector3; text: string }> = [
      { a: new THREE.Vector3(min.x, min.y, max.z + o), b: new THREE.Vector3(max.x, min.y, max.z + o), text: `${(max.x - min.x).toFixed(2)} m` },
      { a: new THREE.Vector3(max.x + o, min.y, min.z), b: new THREE.Vector3(max.x + o, min.y, max.z), text: `${(max.z - min.z).toFixed(2)} m` },
      { a: new THREE.Vector3(min.x - o, min.y, min.z - o), b: new THREE.Vector3(min.x - o, max.y, min.z - o), text: `${(max.y - min.y).toFixed(2)} m` },
    ];
    for (const d of dims) {
      g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([d.a, d.b]), lineMat));
      const sprite = this.textSprite(d.text);
      sprite.position.copy(d.a).add(d.b).multiplyScalar(0.5);
      g.add(sprite);
    }
    g.visible = false;
    this.dimGroup = g;
    this.scene.add(g);
    this.refreshDimensions();
  }

  private textSprite(text: string): THREE.Sprite {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 64;
    const ctx = c.getContext("2d");
    if (ctx) {
      ctx.fillStyle = `rgb(${this.palette.bg.join(",")})`;
      ctx.fillRect(0, 0, 256, 64);
      ctx.fillStyle = `rgb(${this.palette.text.join(",")})`;
      ctx.font = "600 34px ui-monospace, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 128, 34);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
    sprite.scale.set(0.8, 0.2, 1);
    sprite.renderOrder = 5;
    return sprite;
  }

  // ---- camera: projection, zoom, screenshot ---------------------------------------------------------------------

  /** ponytail: orthographic = narrow-FOV perspective at the same framing (see view-model ORTHO_FOV). */
  public setOrthographic(on: boolean) {
    const to = on ? ORTHO_FOV : PERSPECTIVE_FOV;
    if (this.camera.fov === to) return;
    const target = this.controls.target;
    const dir = this.camera.position.clone().sub(target);
    const d = dir.length() || 5;
    dir.normalize().multiplyScalar(dollyForFov(d, this.camera.fov, to));
    this.cameraAnimation = null;
    const k = dollyForFov(1, PERSPECTIVE_FOV, to);
    this.camera.fov = to;
    this.camera.far = 5000 * k;
    this.controls.minDistance = 0.2 * k;
    this.controls.maxDistance = this.distanceCap * k;
    this.camera.position.copy(target).add(dir);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  public isOrthographic(): boolean {
    return this.camera.fov < PERSPECTIVE_FOV;
  }

  /** Distance multiplier that keeps perspective-calibrated framing valid at the current FOV. */
  private fovScale(): number {
    return dollyForFov(1, PERSPECTIVE_FOV, this.camera.fov);
  }

  public zoomBy(factor: number) {
    if (this.isContextLost) return;
    this.cameraAnimation = null;
    const target = this.controls.target;
    const off = this.camera.position.clone().sub(target);
    const d = Math.min(this.controls.maxDistance, Math.max(this.controls.minDistance, off.length() * factor));
    this.camera.position.copy(target).add(off.setLength(d));
    this.controls.update();
  }

  public captureImage(): string | null {
    if (this.isContextLost) return null;
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  public getMetersPerPixel(): number {
    const h = this.container.clientHeight || 600;
    return (2 * this.camera.position.distanceTo(this.controls.target) * Math.tan((this.camera.fov * Math.PI) / 360)) / h;
  }

  public setRenderMode(mode: RenderMode) {
    this.renderMode = mode;
    this.applyRenderMode();
  }

  public getRenderMode(): RenderMode {
    return this.renderMode;
  }

  public hasShell(): boolean {
    return this.shellParts.length > 0;
  }

  /** True when map-imported / generated outdoor parts (buildings, roads, furniture ...) are drawn. */
  public hasOutdoor(): boolean {
    return this.outdoor.parts.length > 0;
  }

  public getOutdoorSummary(): { parts: OutdoorPart[]; skipped: number; imported: number; generated: number } {
    const parts = this.outdoor.parts;
    return { parts, skipped: this.outdoor.diagnostics.skipped.length, imported: parts.filter((q) => q.imported).length, generated: parts.filter((q) => q.generated).length };
  }

  public setLayerVisibility(layers: Partial<ViewportLayers>) {
    const fullCloudChanged = layers.fullCloud !== undefined && layers.fullCloud !== this.layerVisibility.fullCloud;
    this.layerVisibility = { ...this.layerVisibility, ...layers };
    if (fullCloudChanged) this.buildPointLayer(false); // swap the point layer only; the view stays where it is

    if (this.layers.points) {
      this.layers.points.visible = this.layerVisibility.points;
    }
    if (this.layers.entities) {
      this.layers.entities.visible = this.layerVisibility.entities;
    }
    this.updateMeshVisibilities();

    if (this.layers.shell) {
      this.layers.shell.visible = this.layerVisibility.shell;
    }
    if (this.outdoorGroup) this.outdoorGroup.visible = this.layerVisibility.shell;
    if (this.layers.cameras) {
      this.layers.cameras.visible = this.layerVisibility.cameras;
    }
    if (this.layers.mesh) {
      this.layers.mesh.visible = this.layerVisibility.mesh;
    }
    if (layers.splats !== undefined) {
      this.setSplatVisibility(layers.splats);
    }

    if (layers.topology !== undefined) {
      if (this.layers.topology) {
        this.layers.topology.visible = this.layerVisibility.topology;
      } else if (this.layerVisibility.topology) {
        this.renderSpatialTopology(this.selectedEntityId);
      }
    }

    // Update uncertainty shaders if uncertainty toggle changed
    if (layers.uncertainty !== undefined) {
      for (const mesh of this.entityMeshes.values()) {
        const mat = mesh.material as THREE.MeshLambertMaterial;
        const conf = mesh.userData.confidence;
        if (this.layerVisibility.uncertainty && typeof conf === "number") {
          const hue = conf * 0.33;
          mat.color.setHSL(hue, 0.9, 0.5);
        } else {
          // No recorded confidence means no uncertainty colour: the entity
          // keeps its type colour instead of being painted as if it were 0.5.
          mat.color.setHex(mesh.userData.baseColor);
        }
      }
    }
  }

  /**
   * Colour entities by quality band (good / fair / weak / unknown), or restore their type colours with null.
   * The band colours are the app's status colours; "unknown" is a neutral grey, never green or red.
   */
  public setBandOverlay(bands: Record<string, string> | null) {
    const BAND_COLORS: Record<string, number> = { good: 0x2ecc71, fair: 0xf5a623, weak: 0xe74c3c, unknown: 0x8a8fa3 };
    for (const [id, mesh] of this.entityMeshes) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const band = bands?.[id];
      const hex = band && band in BAND_COLORS ? BAND_COLORS[band] : mesh.userData.baseColor;
      mat.color.setHex(hex);
      // the edge lines are children of the mesh; they follow the fill so the outline never disagrees with it
      for (const child of mesh.children) {
        const lineMat = (child as THREE.LineSegments).material as THREE.LineBasicMaterial | undefined;
        if (lineMat && "color" in lineMat) lineMat.color.setHex(hex);
      }
    }
  }

  public setWallsVisibility(visible: boolean) {
    this.setLayerVisibility({ walls: visible });
  }

  public isolateSpace(entityId: string | null) {
    this.isolatedSpaceId = entityId;
    if (this.isContextLost) return;
    this.updateMeshVisibilities();
    if (entityId) {
      this.flyToEntity(entityId);
      this.renderSpatialTopology(entityId);
    } else {
      this.renderSpatialTopology(this.selectedEntityId);
    }
  }

  public isEntityOnActiveLevel(eid: string): boolean {
    if (this.activeLevelIndex === null) return true;
    if (!this.world?.entities?.[eid]) return false;
    const ent = this.world.entities[eid];
    const entLevel = ent.custom_properties?.level ?? ent.custom_properties?.floor_level;
    if (entLevel !== undefined) {
      return Number(entLevel) === this.activeLevelIndex;
    }
    // Check if canonical interior_space_graph contains this entity in the active level
    const metaLevels = (this.world.metadata?.interior_space_graph as Record<string, unknown> | undefined)?.levels as Array<Record<string, unknown>> | undefined;
    const activeMetaLevel = metaLevels?.[this.activeLevelIndex];
    if (activeMetaLevel) {
      const roomIds = (activeMetaLevel.room_ids as string[] | undefined) || [];
      const corridorIds = (activeMetaLevel.corridor_ids as string[] | undefined) || [];
      return roomIds.includes(eid) || corridorIds.includes(eid) || ent.parent_id === activeMetaLevel.level_id;
    }
    // Unassigned entities are not attributed to a level; never fabricate by dividing height
    return false;
  }

  public setLevelFilter(levelIndex: number | null) {
    this.activeLevelIndex = levelIndex;
    if (this.isContextLost) return;
    if (this.selectedEntityId && !this.isEntityOnActiveLevel(this.selectedEntityId)) {
      this.selectEntity(null);
    }
    this.updateMeshVisibilities();
  }

  private updateMeshVisibilities() {
    const matchingIsolatedIds = new Set<string>();
    if (this.isolatedSpaceId && this.world && this.world.entities) {
      matchingIsolatedIds.add(this.isolatedSpaceId);
      const spaceMesh = this.entityMeshes.get(this.isolatedSpaceId);
      const spaceBox = spaceMesh ? new THREE.Box3().setFromObject(spaceMesh) : null;
      const spaceEntity = this.world.entities[this.isolatedSpaceId];

      for (const [id, e] of Object.entries(this.world.entities)) {
        if (id === this.isolatedSpaceId) continue;
        if (e.parent_id === this.isolatedSpaceId) {
          matchingIsolatedIds.add(id);
          continue;
        }
        const isRel =
          (e.relationships || []).some(
            (r) =>
              r.target_id === this.isolatedSpaceId ||
              (r as { target_entity_id?: string }).target_entity_id === this.isolatedSpaceId
          ) ||
          (spaceEntity?.relationships || []).some(
            (r) =>
              r.target_id === id ||
              (r as { target_entity_id?: string }).target_entity_id === id
          );
        if (isRel) {
          matchingIsolatedIds.add(id);
          continue;
        }
        if (spaceBox) {
          const otherMesh = this.entityMeshes.get(id);
          if (otherMesh && spaceBox.containsPoint(otherMesh.position)) {
            matchingIsolatedIds.add(id);
          }
        }
      }
    }

    for (const [eid, mesh] of this.entityMeshes.entries()) {
      const isOversized = mesh.userData.isOversized;
      const isWall = mesh.userData.entityType === "wall";
      const conf = renderWeightFor(mesh.userData.confidence);
      const outlierScale = isOversized ? 0.25 : 1.0;
      const isSpaceVolume =
        mesh.userData.entityType === "room" ||
        mesh.userData.entityType === "corridor" ||
        mesh.userData.entityType === "space";
      const mat = mesh.material as THREE.MeshLambertMaterial;

      const matchesLevel = this.isEntityOnActiveLevel(eid);

      if (this.hiddenEntityIds.has(eid)) {
        mesh.visible = false;
        continue;
      }

      if (mesh.userData.coveredByShell && this.layers.shell) {
        // the shell draws this entity; keep the box only as an invisible picking proxy
        mesh.visible = true;
        mat.visible = false;
        mesh.children.forEach((c) => { c.visible = false; });
        continue;
      }

      if (!this.layerVisibility.entities) {
        mesh.visible = false;
        continue;
      }
      if (isOversized && !this.layerVisibility.oversized) {
        mesh.visible = false;
        continue;
      }
      if (isWall && !this.layerVisibility.walls) {
        mesh.visible = false;
        continue;
      }

      if (!matchesLevel) {
        mesh.visible = true;
        mat.opacity = 0.03;
        continue;
      }

      if (this.isolatedSpaceId) {
        if (matchingIsolatedIds.has(eid)) {
          mesh.visible = true;
          mat.opacity = isSpaceVolume
            ? 0.08
            : Math.max(0.35, (0.35 + 0.45 * conf) * outlierScale);
        } else {
          mesh.visible = true;
          mat.opacity = 0.03;
        }
      } else {
        mesh.visible = true;
        mat.opacity = isSpaceVolume
          ? 0.05
          : Math.max(0.12, (0.15 + 0.35 * conf) * outlierScale);
      }
    }
  }

  public renderSpatialTopology(selectedEntityId: string | null) {
    if (this.isContextLost) return;
    if (this.layers.topology) {
      this.root.remove(this.layers.topology);
      this.layers.topology.traverse((child) => {
        if (child instanceof THREE.Line || child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      this.layers.topology = null;
    }

    if (!this.layerVisibility.topology || !this.world || this.suppressLegacy) return;

    const topologyGroup = new THREE.Group();
    topologyGroup.name = "SpatialTopologyGroup";

    const addConnector = (from: THREE.Vector3, to: THREE.Vector3, colorHex: number) => {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
      const lineMat = new THREE.LineDashedMaterial({
        color: colorHex,
        dashSize: 0.12,
        gapSize: 0.06,
        depthTest: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      line.renderOrder = 995;
      topologyGroup.add(line);

      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.045, 10, 10),
        new THREE.MeshBasicMaterial({ color: colorHex, depthTest: false })
      );
      marker.position.copy(to);
      marker.renderOrder = 996;
      topologyGroup.add(marker);
    };

    if (!selectedEntityId) {
      // Global topology network: render space graph edges between rooms, corridors, and doors
      const spaceEntities = Object.values(this.world.entities || {}).filter(
        (e) => e.type === "room" || e.type === "corridor" || e.type === "space"
      );
      for (const space of spaceEntities) {
        const spaceMesh = this.entityMeshes.get(space.id);
        if (!spaceMesh || !spaceMesh.visible) continue;
        const sPos = spaceMesh.position.clone();

        // Connected corridors
        const connectedCorridors = Array.isArray(space.custom_properties?.connected_corridor_ids)
          ? (space.custom_properties.connected_corridor_ids as string[])
          : [];
        for (const cid of connectedCorridors) {
          const cMesh = this.entityMeshes.get(cid);
          if (cMesh && cMesh.visible) {
            addConnector(sPos, cMesh.position, 0x00e5ff);
          }
        }

        // Adjacent rooms
        const adjacentRooms = Array.isArray(space.custom_properties?.adjacent_room_ids)
          ? (space.custom_properties.adjacent_room_ids as string[])
          : [];
        for (const arid of adjacentRooms) {
          if (arid > space.id) {
            const arMesh = this.entityMeshes.get(arid);
            if (arMesh && arMesh.visible) {
              addConnector(sPos, arMesh.position, 0x38bdf8);
            }
          }
        }

        // Doors connecting to this space
        for (const [eid, ent] of Object.entries(this.world.entities || {})) {
          if (ent.type === "door" || ent.type === "opening") {
            const doorMesh = this.entityMeshes.get(eid);
            if (!doorMesh || !doorMesh.visible) continue;
            const isRel = (ent.relationships || []).some(
              (r) => r.target_id === space.id || (r as { target_entity_id?: string }).target_entity_id === space.id
            ) || ent.parent_id === space.id;
            if (isRel) {
              addConnector(sPos, doorMesh.position, 0x10b981);
            }
          }
        }
      }
      if (topologyGroup.children.length > 0) {
        this.layers.topology = topologyGroup;
        this.root.add(topologyGroup);
      }
      return;
    }

    const targetEntity = this.world.entities?.[selectedEntityId];
    const targetMesh = this.entityMeshes.get(selectedEntityId);
    if (!targetEntity || !targetMesh) return;

    const sourcePos = targetMesh.position.clone();
    const tType = targetEntity.type.toLowerCase();

    if (tType === "room" || tType === "space") {
      // Room relationships: walls, floor, ceiling, doors, windows, corridors, adjacent rooms, level
      const connectedCorridorIds = new Set<string>(
        Array.isArray(targetEntity.custom_properties?.connected_corridor_ids)
          ? (targetEntity.custom_properties.connected_corridor_ids as string[])
          : []
      );
      const adjacentRoomIds = new Set<string>(
        Array.isArray(targetEntity.custom_properties?.adjacent_room_ids)
          ? (targetEntity.custom_properties.adjacent_room_ids as string[])
          : []
      );
      const boundaryIds = new Set<string>(
        Array.isArray(targetEntity.custom_properties?.boundary_element_ids)
          ? (targetEntity.custom_properties.boundary_element_ids as string[])
          : Array.isArray(targetEntity.custom_properties?.wall_ids)
          ? (targetEntity.custom_properties.wall_ids as string[])
          : []
      );

      for (const [otherId, otherEntity] of Object.entries(this.world.entities || {})) {
        if (otherId === selectedEntityId) continue;
        const otherMesh = this.entityMeshes.get(otherId);
        if (!otherMesh) continue;

        const otherType = otherEntity.type.toLowerCase();
        const isContained = otherEntity.parent_id === selectedEntityId;
        const isRelConnected =
          (otherEntity.relationships || []).some(
            (r) =>
              r.target_id === selectedEntityId ||
              (r as { target_entity_id?: string }).target_entity_id === selectedEntityId
          ) ||
          (targetEntity.relationships || []).some(
            (r) =>
              r.target_id === otherId ||
              (r as { target_entity_id?: string }).target_entity_id === otherId
          );

        const isExplicitBoundary = boundaryIds.has(otherId);
        const isExplicitCorridor = connectedCorridorIds.has(otherId);
        const isExplicitAdjRoom = adjacentRoomIds.has(otherId);

        let isGeoInside = false;
        if (!isContained && !isRelConnected && !isExplicitBoundary) {
          const box = new THREE.Box3().setFromObject(targetMesh);
          isGeoInside = box.containsPoint(otherMesh.position);
        }

        if (isContained || isRelConnected || isExplicitBoundary || isExplicitCorridor || isExplicitAdjRoom || isGeoInside) {
          if (otherType === "wall") {
            addConnector(sourcePos, otherMesh.position, 0xf59e0b);
          } else if (otherType === "floor") {
            addConnector(sourcePos, otherMesh.position, 0x3b82f6);
          } else if (otherType === "ceiling") {
            addConnector(sourcePos, otherMesh.position, 0x94a3b8);
          } else if (otherType === "door" || otherType === "window") {
            addConnector(sourcePos, otherMesh.position, otherType === "door" ? 0x10b981 : 0x06b6d4);
            // Trace door onwards to adjacent space
            for (const [adjId, adjEntity] of Object.entries(this.world.entities || {})) {
              if (
                adjId !== selectedEntityId &&
                (adjEntity.type === "room" || adjEntity.type === "corridor")
              ) {
                const adjMesh = this.entityMeshes.get(adjId);
                if (adjMesh) {
                  const adjRel = (otherEntity.relationships || []).some(
                    (r) =>
                      r.target_id === adjId ||
                      (r as { target_entity_id?: string }).target_entity_id === adjId
                  );
                  if (adjRel) {
                    addConnector(otherMesh.position, adjMesh.position, 0x00e5ff);
                  }
                }
              }
            }
          } else if (otherType === "corridor") {
            addConnector(sourcePos, otherMesh.position, 0x00e5ff);
          } else if (otherType === "room" || otherType === "space") {
            addConnector(sourcePos, otherMesh.position, 0x38bdf8);
          } else if (otherType === "stairs" || otherType === "stair") {
            addConnector(sourcePos, otherMesh.position, 0xec4899);
          } else if (otherType === "storey" || otherType === "level") {
            addConnector(sourcePos, otherMesh.position, 0x8b5cf6);
          } else if (otherType === "column" || otherType === "beam") {
            addConnector(sourcePos, otherMesh.position, 0xeab308);
          }
        }
      }

      if (this.camerasData && this.camerasData.cameras) {
        const roomObsIds = new Set<string>();
        (targetEntity.observations || []).forEach((o) => roomObsIds.add(o.id));
        for (const cam of this.camerasData.cameras) {
          const camId = cam.evidence_id || cam.id;
          if (camId && roomObsIds.has(camId)) {
            const camPos = new THREE.Vector3(
              cam.position_m[0],
              cam.position_m[1],
              cam.position_m[2]
            );
            addConnector(sourcePos, camPos, 0x35d07f);
          }
        }
      }
    } else if (tType === "corridor") {
      // Corridor reveals connected rooms, doorway openings, stairs, and level
      const connectedRoomIds = new Set<string>(
        Array.isArray(targetEntity.custom_properties?.connected_room_ids)
          ? (targetEntity.custom_properties.connected_room_ids as string[])
          : []
      );

      // Direct relationships of the corridor
      for (const rel of targetEntity.relationships || []) {
        const tid = (rel as { target_id?: string; target_entity_id?: string }).target_id || (rel as { target_id?: string; target_entity_id?: string }).target_entity_id;
        if (tid) {
          const targetEnt = this.world.entities?.[tid];
          if (targetEnt && (targetEnt.type === "room" || targetEnt.type === "space")) {
            connectedRoomIds.add(tid);
          }
        }
      }

      // 1. Draw connectors to all connected rooms
      for (const roomId of connectedRoomIds) {
        const roomMesh = this.entityMeshes.get(roomId);
        if (roomMesh) {
          addConnector(sourcePos, roomMesh.position, 0x00e5ff);
        }
      }

      // 2. Connect to doors, walls, stairs, and containing level
      for (const [otherId, otherEntity] of Object.entries(this.world.entities || {})) {
        if (otherId === selectedEntityId) continue;
        const otherMesh = this.entityMeshes.get(otherId);
        if (!otherMesh) continue;

        const otherType = otherEntity.type.toLowerCase();
        const isRelConnected =
          (otherEntity.relationships || []).some(
            (r) =>
              r.target_id === selectedEntityId ||
              (r as { target_entity_id?: string }).target_entity_id === selectedEntityId
          ) ||
          (targetEntity.relationships || []).some(
            (r) =>
              r.target_id === otherId ||
              (r as { target_entity_id?: string }).target_entity_id === otherId
          );

        if (isRelConnected) {
          if (otherType === "door") {
            addConnector(sourcePos, otherMesh.position, 0x10b981);
          } else if (otherType === "stairs" || otherType === "stair") {
            addConnector(sourcePos, otherMesh.position, 0xec4899);
          } else if (otherType === "storey" || otherType === "level") {
            addConnector(sourcePos, otherMesh.position, 0x8b5cf6);
          } else if (otherType === "wall") {
            addConnector(sourcePos, otherMesh.position, 0xf59e0b);
          }
        }
      }
    } else if (tType === "wall") {
      for (const [otherId, otherEntity] of Object.entries(this.world.entities || {})) {
        if (otherId === selectedEntityId) continue;
        const otherMesh = this.entityMeshes.get(otherId);
        if (!otherMesh) continue;
        const otherType = otherEntity.type.toLowerCase();
        if (otherType === "door" || otherType === "window") {
          const isHosted =
            otherEntity.parent_id === selectedEntityId ||
            (otherEntity.relationships || []).some(
              (r) =>
                r.target_id === selectedEntityId ||
                (r as { target_entity_id?: string }).target_entity_id === selectedEntityId
            );
          if (isHosted) {
            addConnector(sourcePos, otherMesh.position, 0x10b981);
          }
        } else if (otherType === "room" || otherType === "corridor") {
          const isRoomBound =
            targetEntity.parent_id === otherId ||
            (targetEntity.relationships || []).some(
              (r) =>
                r.target_id === otherId ||
                (r as { target_entity_id?: string }).target_entity_id === otherId
            );
          if (isRoomBound) {
            addConnector(sourcePos, otherMesh.position, 0x3b82f6);
          }
        }
      }
    } else if (tType === "door" || tType === "window") {
      for (const [otherId, otherEntity] of Object.entries(this.world.entities || {})) {
        if (otherId === selectedEntityId) continue;
        const otherMesh = this.entityMeshes.get(otherId);
        if (!otherMesh) continue;
        const otherType = otherEntity.type.toLowerCase();
        if (otherType === "wall") {
          const isHost =
            targetEntity.parent_id === otherId ||
            (targetEntity.relationships || []).some(
              (r) =>
                r.target_id === otherId ||
                (r as { target_entity_id?: string }).target_entity_id === otherId
            );
          if (isHost) {
            addConnector(sourcePos, otherMesh.position, 0xf59e0b);
          }
        } else if (otherType === "room" || otherType === "corridor") {
          const isSpace = (targetEntity.relationships || []).some(
            (r) =>
              r.target_id === otherId ||
              (r as { target_entity_id?: string }).target_entity_id === otherId
          );
          if (isSpace) {
            addConnector(sourcePos, otherMesh.position, 0x00e5ff);
          }
        }
      }
    } else if (tType === "stairs" || tType === "stair") {
      // Stair reveals connected levels and landings
      const box = new THREE.Box3().setFromObject(targetMesh);
      const bottomCenter = new THREE.Vector3(sourcePos.x, box.min.y, sourcePos.z);
      const topCenter = new THREE.Vector3(sourcePos.x, box.max.y, sourcePos.z);
      addConnector(bottomCenter, topCenter, 0xec4899);

      // Connected level IDs from custom_properties or relationships
      const connectedLevelIds = new Set<string>(
        Array.isArray(targetEntity.custom_properties?.connected_level_ids)
          ? (targetEntity.custom_properties.connected_level_ids as string[])
          : Array.isArray(targetEntity.custom_properties?.storey_ids)
          ? (targetEntity.custom_properties.storey_ids as string[])
          : []
      );

      for (const rel of targetEntity.relationships || []) {
        const tid = (rel as { target_id?: string; target_entity_id?: string }).target_id || (rel as { target_id?: string; target_entity_id?: string }).target_entity_id;
        if (tid) {
          const tent = this.world.entities?.[tid];
          if (tent && (tent.type === "storey" || tent.type === "level" || tid.startsWith("storey") || tid.startsWith("level"))) {
            connectedLevelIds.add(tid);
          }
        }
      }

      // Connect to connected level meshes/planes
      for (const levelId of connectedLevelIds) {
        const lvlMesh = this.entityMeshes.get(levelId);
        if (lvlMesh) {
          const lvlPos = lvlMesh.position;
          // Connect to top or bottom depending on elevation
          const fromPt = lvlPos.y >= sourcePos.y ? topCenter : bottomCenter;
          addConnector(fromPt, lvlPos, 0xa855f7);
        } else {
          // If level entity has elevation property but no mesh, connect to elevation plane point
          const lvlEnt = this.world.entities?.[levelId];
          const elev = Number(lvlEnt?.custom_properties?.elevation_m ?? lvlEnt?.custom_properties?.floor_height_m ?? (levelId.includes("1") || levelId.includes("upper") ? box.max.y : box.min.y));
          const targetPt = new THREE.Vector3(sourcePos.x + 1.5, elev, sourcePos.z);
          const fromPt = elev >= sourcePos.y ? topCenter : bottomCenter;
          addConnector(fromPt, targetPt, 0xa855f7);
        }
      }

      // Connect to any connected corridors / landing areas
      for (const rel of targetEntity.relationships || []) {
        const tid = (rel as { target_id?: string; target_entity_id?: string }).target_id || (rel as { target_id?: string; target_entity_id?: string }).target_entity_id;
        if (tid && tid.startsWith("corridor")) {
          const cMesh = this.entityMeshes.get(tid);
          if (cMesh) {
            addConnector(sourcePos, cMesh.position, 0x06b6d4);
          }
        }
      }
    }

    this.layers.topology = topologyGroup;
    this.layers.topology.visible = this.layerVisibility.topology;
    this.root.add(this.layers.topology);
  }

  public setCorrectionPreview(
    entityId: string,
    previewType?: string,
    previewConfidence?: number | null,
    previewTransform?: { position?: { x: number; y: number; z: number } }
  ) {
    const mesh = this.entityMeshes.get(entityId);
    if (!mesh) return;

    if (!this.previewCorrection || this.previewCorrection.entityId !== entityId) {
      this.previewCorrection = {
        entityId,
        type: previewType,
        confidence: typeof previewConfidence === "number" ? previewConfidence : undefined,
        originalPosition: mesh.position.clone(),
      };
    } else {
      if (previewType !== undefined) this.previewCorrection.type = previewType;
      if (previewConfidence !== undefined) {
        this.previewCorrection.confidence = typeof previewConfidence === "number" ? previewConfidence : undefined;
      }
    }

    if (previewType) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const newColor = TYPE_COLORS[previewType.toLowerCase()] || TYPE_COLORS.default;
      mat.color.setHex(newColor);
      mat.emissive.setHex(newColor);
      mat.emissiveIntensity = 0.5;
    }

    if (previewTransform?.position) {
      mesh.position.set(previewTransform.position.x, previewTransform.position.y, previewTransform.position.z);
      if (this.layers.selectionOutline && this.selectedEntityId === entityId) {
        this.layers.selectionOutline.position.copy(mesh.position);
      }
    }
  }

  public clearCorrectionPreview() {
    if (this.previewCorrection) {
      const mesh = this.entityMeshes.get(this.previewCorrection.entityId);
      if (mesh) {
        const mat = mesh.material as THREE.MeshLambertMaterial;
        const isSel = this.previewCorrection.entityId === this.selectedEntityId;
        mat.color.setHex(mesh.userData.baseColor);
        mat.emissive.setHex(isSel ? 0x00e5ff : 0x000000);
        mat.emissiveIntensity = isSel ? 0.6 : 0;
        if (this.previewCorrection.originalPosition) {
          mesh.position.copy(this.previewCorrection.originalPosition);
          if (this.layers.selectionOutline && this.selectedEntityId === this.previewCorrection.entityId) {
            this.layers.selectionOutline.position.copy(mesh.position);
          }
        }
      }
      this.previewCorrection = null;
    }
  }

  public applyEntityFilter(matchingIds: Set<string> | null) {
    for (const [eid, mesh] of this.entityMeshes.entries()) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const conf = renderWeightFor(mesh.userData.confidence);
      const isOversized = mesh.userData.isOversized;
      const outlierScale = isOversized ? 0.25 : 1.0;

      if (matchingIds === null) {
        mesh.visible =
          this.layerVisibility.entities &&
          (!isOversized || this.layerVisibility.oversized);
        mat.opacity = Math.max(0.12, (0.15 + 0.35 * conf) * outlierScale);
      } else {
        const matches = matchingIds.has(eid);
        if (matches) {
          mesh.visible = true;
          mat.opacity = Math.max(0.45, (0.35 + 0.5 * conf) * outlierScale);
        } else {
          mesh.visible = true;
          mat.opacity = 0.04;
        }
      }
    }
  }

  public selectEntity(id: string | null, updateCamera = false) {
    this.selectedEntityId = id;
    if (this.isContextLost) {
      this.onSelectCallback?.(id);
      return;
    }

    // Reset outline
    if (this.layers.selectionOutline) {
      this.root.remove(this.layers.selectionOutline);
      this.layers.selectionOutline.geometry.dispose();
      (this.layers.selectionOutline.material as THREE.Material).dispose();
      this.layers.selectionOutline = null;
    }

    for (const [eid, mesh] of this.entityMeshes.entries()) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const isSel = eid === id;
      mat.emissive = new THREE.Color(isSel ? hex(this.palette.accent) : 0x000000);
      mat.emissiveIntensity = isSel ? 0.6 : 0;

      if (isSel) {
        // Create prominent bright selection outline
        const wireGeo = new THREE.EdgesGeometry(mesh.geometry);
        const wireMat = new THREE.LineBasicMaterial({
          color: hex(this.palette.accent),
          linewidth: 2,
          transparent: true,
          opacity: 0.95,
        });
        this.layers.selectionOutline = new THREE.LineSegments(wireGeo, wireMat);
        this.layers.selectionOutline.position.copy(mesh.position);
        this.root.add(this.layers.selectionOutline);

        if (updateCamera) {
          this.flyToEntity(id);
        }
      }
    }

    this.applyOutdoorMode(); // recolours the selected outdoor part and redraws its outline
    if (updateCamera && id && !this.entityMeshes.has(id)) this.flyToEntity(id);
    this.renderSpatialTopology(id);
    if (this.transformGizmo && this.transformGizmoEnabled) {
      if (id && this.entityMeshes.has(id) && this.selectionEnabled) {
        const mesh = this.entityMeshes.get(id)!;
        this.transformGizmo.attach(id, mesh);
      } else {
        this.transformGizmo.detach();
      }
    }
    this.refreshShellVisibility(); // a selected (cut-away) wall is revealed
    this.onSelectCallback?.(id);
  }

  private animateCameraTo(endPos: THREE.Vector3, endTarget: THREE.Vector3, duration = 380, scale = true) {
    if (
      !Number.isFinite(endPos.x) ||
      !Number.isFinite(endPos.y) ||
      !Number.isFinite(endPos.z) ||
      !Number.isFinite(endTarget.x) ||
      !Number.isFinite(endTarget.y) ||
      !Number.isFinite(endTarget.z)
    ) {
      return;
    }
    const k = scale ? this.fovScale() : 1;
    if (k !== 1) endPos = endTarget.clone().add(endPos.clone().sub(endTarget).multiplyScalar(k));
    this.cameraAnimation = {
      startPos: this.camera.position.clone(),
      endPos: endPos.clone(),
      startTarget: this.controls.target.clone(),
      endTarget: endTarget.clone(),
      startTime: performance.now(),
      duration,
    };
  }

  public flyToEntity(id: string, animate = true) {
    if (this.isContextLost) return;
    const mesh = this.entityMeshes.get(id);
    const outdoorBox = this.outdoorBoxes.get(id);
    if (!mesh && !outdoorBox) return;

    const box = mesh ? new THREE.Box3().setFromObject(mesh) : outdoorBox!.clone().applyMatrix4(this.root.matrixWorld);
    const size = box.getSize(new THREE.Vector3()).length();
    const dir = new THREE.Vector3(0.8, 0.55, 0.8).normalize();
    const d = Math.max(1.2, size * 1.45);

    const meshPos = mesh ? mesh.getWorldPosition(new THREE.Vector3()) : box.getCenter(new THREE.Vector3());
    const endPos = meshPos.clone().addScaledVector(dir, d);
    const endTarget = meshPos.clone();

    if (animate) {
      this.animateCameraTo(endPos, endTarget, 350);
    } else {
      if (!Number.isFinite(endPos.x) || !Number.isFinite(endTarget.x)) return;
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTarget);
      this.controls.update();
    }
  }

  public flyToCamera(evidenceId: string, animate = true) {
    if (this.isContextLost) return;
    if (!this.camerasData || !this.camerasData.cameras) return;
    const cam = this.camerasData.cameras.find(
      (c) => (c.evidence_id || c.id) === evidenceId
    );
    if (!cam) return;
    const p = cam.position_m;
    const quat = new THREE.Quaternion(
      cam.rotation_wxyz[1],
      cam.rotation_wxyz[2],
      cam.rotation_wxyz[3],
      cam.rotation_wxyz[0]
    );
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quat).transformDirection(this.root.matrixWorld);
    const endPos = this.root.localToWorld(new THREE.Vector3(p[0], p[1], p[2]));
    const endTarget = endPos.clone().addScaledVector(forward, 2.5);

    if (animate) {
      this.animateCameraTo(endPos, endTarget, 380);
    } else {
      if (!Number.isFinite(endPos.x) || !Number.isFinite(endTarget.x)) return;
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTarget);
      this.controls.update();
    }
  }

  public flyToPosition(x: number, y: number, z: number, animate = true) {
    if (this.isContextLost) return;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    const endTarget = this.root.localToWorld(new THREE.Vector3(x, y, z));
    const endPos = endTarget.clone().add(new THREE.Vector3(2.5, 2, 2.5));
    if (animate) {
      this.animateCameraTo(endPos, endTarget, 350);
    } else {
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTarget);
      this.controls.update();
    }
  }

  public getCameraPose(): { position: [number, number, number]; target: [number, number, number] } {
    return {
      position: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      target: [this.controls.target.x, this.controls.target.y, this.controls.target.z],
    };
  }

  public flyToBookmark(position: [number, number, number], target: [number, number, number], animate = true) {
    if (this.isContextLost) return;
    if (!Number.isFinite(position[0]) || !Number.isFinite(target[0])) return;
    const endPos = new THREE.Vector3(position[0], position[1], position[2]);
    const endTarget = new THREE.Vector3(target[0], target[1], target[2]);
    if (animate) {
      this.animateCameraTo(endPos, endTarget, 400, false);
    } else {
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTarget);
      this.controls.update();
    }
  }

  public frameAll(animate = true) {
    if (this.isContextLost) return;
    let endPos: THREE.Vector3;
    let endTarget: THREE.Vector3;

    if (this.pointsData && this.pointsData.length > 0) {
      const rb = robustPointsBounds(this.pointsData);
      endTarget = this.root.localToWorld(new THREE.Vector3(rb.center[0], rb.center[1], rb.center[2]));
      endPos = endTarget.clone().addScalar(rb.extent * 1.0);
    } else if (this.entityMeshes.size > 0 || this.outdoorBox) {
      const box = new THREE.Box3();
      for (const mesh of this.entityMeshes.values()) {
        box.expandByObject(mesh);
      }
      if (this.outdoorBox) box.union(this.outdoorBox);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      const k = this.outdoorBox ? 0.62 : 0.8; // neighbourhoods are wide and flat: pull in closer for an isometric overview
      endPos = new THREE.Vector3(
        center.x + size * k,
        center.y + size * k,
        center.z + size * k
      );
      endTarget = center;
    } else {
      endPos = new THREE.Vector3(4, 3, 6);
      endTarget = new THREE.Vector3(0, 0, 0);
    }

    if (animate) {
      this.animateCameraTo(endPos, endTarget, 400);
    } else {
      if (!Number.isFinite(endPos.x) || !Number.isFinite(endTarget.x)) return;
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTarget);
      this.controls.update();
    }
  }

  public setViewPreset(preset: ViewPreset, animate = true) {
    if (this.isContextLost) return;
    const target = this.controls.target.clone();
    const d = (this.camera.position.distanceTo(target) || 5) / this.fovScale();
    let endPos: THREE.Vector3;

    switch (preset) {
      case "top":
        endPos = new THREE.Vector3(target.x, target.y + d, target.z + 0.0001);
        break;
      case "front":
        endPos = new THREE.Vector3(target.x, target.y, target.z + d);
        break;
      case "side":
        endPos = new THREE.Vector3(target.x + d, target.y, target.z);
        break;
      case "isometric":
      default:
        endPos = new THREE.Vector3(
          target.x + d * 0.577,
          target.y + d * 0.577,
          target.z + d * 0.577
        );
        break;
    }

    if (animate) {
      this.animateCameraTo(endPos, target, 350);
    } else {
      if (!Number.isFinite(endPos.x) || !Number.isFinite(target.x)) return;
      this.camera.position.copy(endPos);
      this.controls.target.copy(target);
      this.controls.update();
    }
  }

  private setupEvents() {
    const dom = this.renderer.domElement;
    const { signal } = this.abortController;
    let pointerDownPos = { x: 0, y: 0 };

    // WebGL Context Loss & Restoration
    dom.addEventListener("webglcontextlost", (event: Event) => {
      // Standard WebGL requirement: event.preventDefault() instructs browser that application handles restoration
      event.preventDefault();
      if (this.isContextLost) return;
      this.isContextLost = true;
      if (this.animationFrameId !== null) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }
      this.onContextLossChangeCallback?.(true);
    }, { signal });

    dom.addEventListener("webglcontextrestored", () => {
      if (!this.isContextLost) return;
      this.isContextLost = false;
      this.onContextLossChangeCallback?.(false);
      // Rebuild scene from canonical WorldIR / points / cameras / mesh data
      if (this.world) {
        this.rebuildScene();
        // Restore active storey isolation if configured
        if (this.activeLevelIndex !== null) {
          this.setLevelFilter(this.activeLevelIndex);
        }
        // Restore selection if selected entity still exists
        if (this.selectedEntityId && (this.entityMeshes.has(this.selectedEntityId) || this.outdoorBoxes.has(this.selectedEntityId))) {
          this.selectEntity(this.selectedEntityId);
        } else if (this.selectedEntityId) {
          this.selectEntity(null);
        }
      }
      // Restart animation loop
      this.startLoop();
    }, { signal });

    this.controls.addEventListener("end", () => this.refreshShellVisibility());

    this.controls.addEventListener("start", () => {
      if (this.isContextLost) return;
      // User manual interaction takes precedence: cancel programmatic animation
      this.cameraAnimation = null;
    });

    // OrbitControls does not fire 'start' on mouse wheel zoom, so cancel animation on wheel
    dom.addEventListener("wheel", () => {
      if (this.isContextLost) return;
      this.cameraAnimation = null;
    }, { signal, passive: true });

    dom.addEventListener("pointerdown", (ev) => {
      if (this.isContextLost) return;
      // Any pointer interaction cancels running animation
      this.cameraAnimation = null;
      if (this.pointerTool === "walk" && ev.button === 0) {
        this.isPointerDraggingWalk = true;
        this.lastWalkPointer = { x: ev.clientX, y: ev.clientY };
      }
      if (ev.button !== 0) return;
      pointerDownPos = { x: ev.clientX, y: ev.clientY };
    }, { signal });

    dom.addEventListener("pointermove", (ev) => {
      if (this.isContextLost) return;
      if (this.pointerTool === "walk" && this.isPointerDraggingWalk) {
        const dx = ev.clientX - this.lastWalkPointer.x;
        const dy = ev.clientY - this.lastWalkPointer.y;
        this.lastWalkPointer = { x: ev.clientX, y: ev.clientY };
        const sensitivity = 0.003;
        this.walkYaw -= dx * sensitivity;
        this.walkPitch = clampPitch(this.walkPitch - dy * sensitivity);
        this.updateWalkCameraLook();
      }
    }, { signal });

    window.addEventListener("pointerup", () => {
      this.isPointerDraggingWalk = false;
    }, { signal });
    window.addEventListener("pointercancel", () => {
      this.isPointerDraggingWalk = false;
    }, { signal });

    window.addEventListener("keydown", (ev: KeyboardEvent) => {
      if (this.pointerTool !== "walk") return;
      const tag = (ev.target as HTMLElement)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      if (ev.code === "KeyW" || ev.code === "ArrowUp") {
        this.walkKeysDown.forward = true;
      } else if (ev.code === "KeyS" || ev.code === "ArrowDown") {
        this.walkKeysDown.backward = true;
      } else if (ev.code === "KeyA") {
        this.walkKeysDown.left = true;
      } else if (ev.code === "KeyD") {
        this.walkKeysDown.right = true;
      } else if (ev.code === "ArrowLeft") {
        this.walkYaw += 0.05;
        this.updateWalkCameraLook();
      } else if (ev.code === "ArrowRight") {
        this.walkYaw -= 0.05;
        this.updateWalkCameraLook();
      }
    }, { signal });

    window.addEventListener("keyup", (ev: KeyboardEvent) => {
      if (ev.code === "KeyW" || ev.code === "ArrowUp") {
        this.walkKeysDown.forward = false;
      } else if (ev.code === "KeyS" || ev.code === "ArrowDown") {
        this.walkKeysDown.backward = false;
      } else if (ev.code === "KeyA") {
        this.walkKeysDown.left = false;
      } else if (ev.code === "KeyD") {
        this.walkKeysDown.right = false;
      }
    }, { signal });

    dom.addEventListener("touchstart", (ev: TouchEvent) => {
      if (this.pointerTool !== "walk") return;
      const rect = dom.getBoundingClientRect();
      for (let i = 0; i < ev.changedTouches.length; i++) {
        const touch = ev.changedTouches[i];
        const region = classifyTouchRegion(touch.clientX - rect.left, rect.width);
        if (region === "move" && this.walkTouchState.moveTouchId === null) {
          this.walkTouchState.moveTouchId = touch.identifier;
          this.walkTouchState.startMoveX = touch.clientX;
          this.walkTouchState.startMoveY = touch.clientY;
        } else if (region === "look" && this.walkTouchState.lookTouchId === null) {
          this.walkTouchState.lookTouchId = touch.identifier;
          this.walkTouchState.lastLookX = touch.clientX;
          this.walkTouchState.lastLookY = touch.clientY;
        }
      }
    }, { signal, passive: true });

    dom.addEventListener("touchmove", (ev: TouchEvent) => {
      if (this.pointerTool !== "walk") return;
      for (let i = 0; i < ev.changedTouches.length; i++) {
        const touch = ev.changedTouches[i];
        if (touch.identifier === this.walkTouchState.moveTouchId) {
          const dx = touch.clientX - this.walkTouchState.startMoveX;
          const dy = touch.clientY - this.walkTouchState.startMoveY;
          this.walkTouchMove = computeTouchJoystick(dx, dy);
        } else if (touch.identifier === this.walkTouchState.lookTouchId) {
          const dx = touch.clientX - this.walkTouchState.lastLookX;
          const dy = touch.clientY - this.walkTouchState.lastLookY;
          this.walkTouchState.lastLookX = touch.clientX;
          this.walkTouchState.lastLookY = touch.clientY;
          const sensitivity = 0.005;
          this.walkYaw -= dx * sensitivity;
          this.walkPitch = clampPitch(this.walkPitch - dy * sensitivity);
          this.updateWalkCameraLook();
        }
      }
    }, { signal, passive: true });

    const endTouch = (ev: TouchEvent) => {
      for (let i = 0; i < ev.changedTouches.length; i++) {
        const touch = ev.changedTouches[i];
        if (touch.identifier === this.walkTouchState.moveTouchId) {
          this.walkTouchState.moveTouchId = null;
          this.walkTouchMove = null;
        } else if (touch.identifier === this.walkTouchState.lookTouchId) {
          this.walkTouchState.lookTouchId = null;
        }
      }
    };
    dom.addEventListener("touchend", endTouch, { signal, passive: true });
    dom.addEventListener("touchcancel", endTouch, { signal, passive: true });

    // Handle clicks only on pointerup if pointer did not drag
    dom.addEventListener("pointerup", (ev) => {
      if (this.isContextLost) return;
      if (ev.button !== 0) return;
      const dragDist = Math.hypot(ev.clientX - pointerDownPos.x, ev.clientY - pointerDownPos.y);
      if (dragDist > 4) {
        // Orbit or pan action, ignore selection
        return;
      }

      if (this.transformGizmo && this.transformGizmo.isInteracting()) {
        return;
      }

      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      // Interactive Measurement Mode
      if (this.isMeasurementMode) {
        let pt: THREE.Vector3 | null = null;
        const areaPlaneY = this.measureTool === "area" && this.measurePts.length > 0 && !this.measureDone
          ? this.root.localToWorld(this.measurePts[0].clone()).y
          : null;
        if (areaPlaneY !== null) {
          // Area polygons live on the horizontal plane through the first corner, so they are planar by construction.
          const onPlane = new THREE.Vector3();
          if (this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -areaPlaneY), onPlane)) {
            pt = onPlane;
          }
        } else {
          const meshes = [...Array.from(this.entityMeshes.values()).filter((m) => m.visible), ...this.outdoorBatches.map((b) => b.solid)];
          const hits = this.raycaster.intersectObjects(meshes, false);
          if (hits.length > 0) {
            pt = hits[0].point.clone();
          } else {
            const groundPoint = new THREE.Vector3();
            const hitGround = this.raycaster.ray.intersectPlane(
              new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
              groundPoint
            );
            if (hitGround) pt = groundPoint;
          }
        }

        if (pt) {
          if (this.measureSnap) {
            const snapped = this.snapToEntityBoxes(pt);
            // keep the area on its plane: snap only in-plane
            pt = areaPlaneY !== null ? new THREE.Vector3(snapped.x, areaPlaneY, snapped.z) : snapped;
          }
          this.handleMeasurementPoint(this.root.worldToLocal(pt));
        }
        return;
      }

      // Entity Selection Mode (the orbit/pan tools do not select):
      if (!this.selectionEnabled) return;
      // Filter meshes to visible and respecting active storey isolation (dimmed meshes cannot be clicked)
      const meshes = Array.from(this.entityMeshes.entries())
        .filter(([eid, m]) => {
          if (!m.visible) return false;
          if (this.shellHiddenEntities.has(eid) || this.hiddenEntityIds.has(eid)) return false;
          if (this.activeLevelIndex !== null && !this.isEntityOnActiveLevel(eid)) return false;
          return true;
        })
        .map(([, m]) => m);

      const hits = this.raycaster.intersectObjects(meshes, false);
      const outdoorHit = this.pickOutdoor();

      if (outdoorHit && (hits.length === 0 || outdoorHit.distance < hits[0].distance)) {
        this.selectEntity(outdoorHit.id);
      } else if (hits.length > 0) {
        const entityId = hits[0].object.userData.entityId as string;
        this.selectEntity(entityId);
      } else {
        this.selectEntity(null);
      }
    }, { signal });

    // Resize observer
    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(this.container);
  }

  private handleMeasurementPoint(pt: THREE.Vector3) {
    if (!this.measurementGroup) return;
    if (this.measureDone || (this.measureTool !== "area" && this.measurePts.length >= (this.measureTool === "angle" ? 3 : 2))) {
      this.clearMeasurement();
    }
    this.measurePts.push(pt.clone());
    this.redrawMeasurement();
  }

  private onResize() {
    if (this.isContextLost) return;
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  private startLoop() {
    if (this.isContextLost) return;
    const animate = () => {
      if (this.isContextLost) return;
      if (this.cameraAnimation) {
        const now = performance.now();
        const elapsed = (now - this.cameraAnimation.startTime) / this.cameraAnimation.duration;
        if (elapsed >= 1) {
          this.camera.position.copy(this.cameraAnimation.endPos);
          this.controls.target.copy(this.cameraAnimation.endTarget);
          this.cameraAnimation = null;
          this.refreshShellVisibility();
        } else {
          // Smooth cubic ease-in-out curve
          const t = elapsed < 0.5 ? 4 * elapsed * elapsed * elapsed : 1 - Math.pow(-2 * elapsed + 2, 3) / 2;
          this.camera.position.lerpVectors(this.cameraAnimation.startPos, this.cameraAnimation.endPos, t);
          this.controls.target.lerpVectors(this.cameraAnimation.startTarget, this.cameraAnimation.endTarget, t);
        }
      }

      if (this.pointerTool === "walk") {
        const now = performance.now();
        const deltaSec = this.lastWalkFrameTime ? Math.min(0.05, (now - this.lastWalkFrameTime) / 1000) : 0.016;
        this.lastWalkFrameTime = now;

        let fwd = (this.walkKeysDown.forward ? 1 : 0) - (this.walkKeysDown.backward ? 1 : 0);
        let strafe = (this.walkKeysDown.right ? 1 : 0) - (this.walkKeysDown.left ? 1 : 0);
        if (this.walkTouchMove) {
          fwd += this.walkTouchMove.forward;
          strafe += this.walkTouchMove.strafe;
        }

        if (Math.hypot(fwd, strafe) > 0.0001) {
          this.applyWalkStep(fwd, strafe, deltaSec);
        }
      } else {
        this.lastWalkFrameTime = 0;
        this.controls.update();
      }
      if (this.outdoor.parts.length > 0) this.fitDepthRange();

      // Emit camera azimuth & elevation for Orientation Compass
      if (this.onCameraChangeCallback) {
        const dir = new THREE.Vector3();
        this.camera.getWorldDirection(dir);
        const azimuthRad = Math.atan2(dir.x, dir.z);
        const azimuthDeg = Math.round(((azimuthRad * 180) / Math.PI + 360) % 360);
        const elevationRad = Math.asin(Math.max(-1, Math.min(1, dir.y)));
        const elevationDeg = Math.round((elevationRad * 180) / Math.PI);
        const mpp = this.getMetersPerPixel();
        const mppKey = Math.round(mpp * 1000);
        if (azimuthDeg !== this.lastAzimuthDeg || elevationDeg !== this.lastElevationDeg || mppKey !== this.lastMppKey) {
          this.lastAzimuthDeg = azimuthDeg;
          this.lastElevationDeg = elevationDeg;
          this.lastMppKey = mppKey;
          const e = this.camera.matrixWorld.elements;
          this.onCameraChangeCallback({
            azimuthDeg, elevationDeg, metersPerPx: mpp,
            right: [e[0], e[1], e[2]], up: [e[4], e[5], e[6]],
          });
        }
      }

      this.renderer.render(this.scene, this.camera);
      this.animationFrameId = requestAnimationFrame(animate);
    };
    this.animationFrameId = requestAnimationFrame(animate);
  }

  public getStatistics(): ViewportStatistics {
    const pts = this.pointsData ? this.pointsData.length / 3 : 0;
    const cams = this.camerasData ? (this.camerasData.cameras || []).length : 0;
    const ents = this.entityMeshes.size + this.outdoor.parts.length;
    return {
      points: pts,
      canonicalPoints: pts,
      renderedPoints: this.renderedPointsCount > 0 ? this.renderedPointsCount : pts,
      isLODSampled: this.renderedPointsCount > 0 && this.renderedPointsCount < pts,
      pointMode: this.pointMode,
      cameras: cams,
      entities: ents,
      isContextLost: this.isContextLost,
    };
  }

  public getContextLost(): boolean {
    return this.isContextLost;
  }

  public getGpuResourceInfo() {
    return {
      geometries: this.renderer.info.memory.geometries,
      textures: this.renderer.info.memory.textures,
      programs: this.renderer.info.programs?.length ?? 0,
      calls: this.renderer.info.render.calls,
      triangles: this.renderer.info.render.triangles,
      points: this.renderer.info.render.points,
      lines: this.renderer.info.render.lines,
    };
  }

  public dispose() {
    this.abortController.abort();
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    this.clearSceneLayers();
    this.disposeDimensions();
    if (this.gridHelper) {
      this.scene.remove(this.gridHelper);
      this.gridHelper.geometry.dispose();
      if (Array.isArray(this.gridHelper.material)) {
        this.gridHelper.material.forEach((m) => m.dispose());
      } else {
        this.gridHelper.material.dispose();
      }
    }
    if (this.axesHelper) {
      this.scene.remove(this.axesHelper);
      this.axesHelper.geometry.dispose();
      if (Array.isArray(this.axesHelper.material)) {
        this.axesHelper.material.forEach((m) => m.dispose());
      } else {
        this.axesHelper.material.dispose();
      }
    }
    this.clearMeasurement();
    if (this.measurementGroup) {
      this.root.remove(this.measurementGroup);
    }
    this.disposeGroupChildren(this.annotationGroup);
    if (this.annotationGroup) this.root.remove(this.annotationGroup);
    if (this.transformGizmo) {
      this.transformGizmo.dispose();
      this.transformGizmo = null;
    }
    this.disposeSplats();
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
  }

  // ── Gaussian Splatting & Hybrid Viewport (Track C) ──────────────────────

  /**
   * Load and render Gaussian Splat primitives from binary buffer (.splat or .ply).
   */
  public loadSplatsFromBuffer(buffer: ArrayBuffer, format: "splat" | "ply" = "splat"): ParsedSplatCloud {
    this.disposeSplats();
    const cloud = format === "ply" ? parsePlySplatBuffer(buffer) : parseSplatBuffer(buffer);
    this.splatCloud = cloud;

    if (cloud.count === 0) return cloud;

    // Unit plane billboard geometry
    const geometry = new THREE.PlaneGeometry(1, 1);

    // Custom Gaussian radial falloff shader material
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uOpacity: { value: this.splatOpacity },
        uScaleMultiplier: { value: this.splatScaleMultiplier },
      },
      vertexShader: `
        attribute vec4 aRotation;
        attribute vec3 aScale;
        attribute float aOpacity;

        varying vec2 vUv;
        varying vec3 vColor;
        varying float vOpacity;

        uniform float uOpacity;
        uniform float uScaleMultiplier;

        vec3 rotateVector(vec4 q, vec3 v) {
          return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
        }

        void main() {
          vUv = uv;
          vColor = instanceColor;
          vOpacity = aOpacity * uOpacity;

          vec3 localPos = position * aScale * uScaleMultiplier;
          vec3 rotatedLocal = rotateVector(aRotation, localPos);

          #ifdef USE_INSTANCING
            vec4 worldPos = instanceMatrix * vec4(rotatedLocal, 1.0);
          #else
            vec4 worldPos = modelMatrix * vec4(rotatedLocal, 1.0);
          #endif

          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        varying vec3 vColor;
        varying float vOpacity;

        void main() {
          vec2 coord = (vUv - vec2(0.5)) * 2.0;
          float r2 = dot(coord, coord);
          if (r2 > 1.0) discard;

          float alpha = vOpacity * exp(-2.5 * r2);
          if (alpha < 0.02) discard;

          gl_FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide,
    });

    const instMesh = new THREE.InstancedMesh(geometry, material, cloud.count);

    instMesh.geometry.setAttribute("aRotation", new THREE.InstancedBufferAttribute(cloud.rotations, 4));
    instMesh.geometry.setAttribute("aScale", new THREE.InstancedBufferAttribute(cloud.scales, 3));
    instMesh.geometry.setAttribute("aOpacity", new THREE.InstancedBufferAttribute(cloud.opacities, 1));

    const dummyMatrix = new THREE.Matrix4();
    const dummyColor = new THREE.Color();

    for (let i = 0; i < cloud.count; i++) {
      dummyMatrix.setPosition(cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2]);
      instMesh.setMatrixAt(i, dummyMatrix);

      dummyColor.setRGB(cloud.colors[i * 3] / 255.0, cloud.colors[i * 3 + 1] / 255.0, cloud.colors[i * 3 + 2] / 255.0);
      instMesh.setColorAt(i, dummyColor);
    }

    instMesh.instanceMatrix.needsUpdate = true;
    if (instMesh.instanceColor) instMesh.instanceColor.needsUpdate = true;

    instMesh.frustumCulled = false;
    instMesh.visible = (this.layerVisibility.splats ?? true) && this.hybridMode !== "cad_only";

    this.layers.splats = instMesh;
    this.root.add(instMesh);
    this.applyHybridRenderMode();

    return cloud;
  }

  public setSplatVisibility(visible: boolean) {
    this.layerVisibility.splats = visible;
    if (this.layers.splats) {
      this.layers.splats.visible = visible && this.hybridMode !== "cad_only";
    }
  }

  public setSplatOpacity(opacity: number) {
    this.splatOpacity = Math.max(0.0, Math.min(1.0, opacity));
    if (this.layers.splats) {
      const mat = this.layers.splats.material as THREE.ShaderMaterial;
      if (mat.uniforms && mat.uniforms.uOpacity) {
        mat.uniforms.uOpacity.value = this.splatOpacity;
      }
    }
  }

  public setSplatScale(multiplier: number) {
    this.splatScaleMultiplier = Math.max(0.1, Math.min(10.0, multiplier));
    if (this.layers.splats) {
      const mat = this.layers.splats.material as THREE.ShaderMaterial;
      if (mat.uniforms && mat.uniforms.uScaleMultiplier) {
        mat.uniforms.uScaleMultiplier.value = this.splatScaleMultiplier;
      }
    }
  }

  public setHybridRenderMode(mode: HybridRenderMode) {
    this.hybridMode = mode;
    this.applyHybridRenderMode();
  }

  private applyHybridRenderMode() {
    const isCadOnly = this.hybridMode === "cad_only";
    const isSplatsOnly = this.hybridMode === "splats_only";

    if (this.layers.splats) {
      this.layers.splats.visible = !isCadOnly && (this.layerVisibility.splats ?? true);
    }
    if (this.layers.shell) {
      this.layers.shell.visible = !isSplatsOnly && this.layerVisibility.shell;
    }
    if (this.layers.points) {
      this.layers.points.visible = isCadOnly ? this.layerVisibility.points : false;
    }
    if (this.layers.mesh) {
      this.layers.mesh.visible = !isSplatsOnly && this.layerVisibility.mesh;
    }
  }

  public getSplatState() {
    return {
      loaded: this.layers.splats !== null,
      count: this.splatCloud?.count ?? 0,
      bounds: this.splatCloud?.bounds ?? null,
      visible: (this.layerVisibility.splats ?? true) && this.hybridMode !== "cad_only",
      opacity: this.splatOpacity,
      scaleMultiplier: this.splatScaleMultiplier,
      hybridMode: this.hybridMode,
    };
  }

  public disposeSplats() {
    if (this.layers.splats) {
      this.root.remove(this.layers.splats);
      this.layers.splats.geometry.dispose();
      (this.layers.splats.material as THREE.Material).dispose();
      this.layers.splats = null;
      this.splatCloud = null;
    }
  }
}
