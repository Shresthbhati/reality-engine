/**
 * Reality Studio 3D Interactive Transform Gizmo Controller.
 * Integrates Three.js TransformControls with real-time room boundary snapping,
 * collision prevention, and a 100% reversible undo/redo stack.
 */

import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import type { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { getStudioHistory, type HistoryStack } from "../viewport/history-stack.ts";
import type {
  TransformGizmoMode,
  TransformState,
  SnapConfig,
  RoomBoundary,
  EntityBounds,
} from "./transform-types.ts";
import { DEFAULT_SNAP_CONFIG } from "./transform-types.ts";
import { applyWallSnappingAndCollision, snapAngle } from "./room-boundary-snap.ts";

export interface TransformGizmoOptions {
  camera: THREE.Camera;
  domElement: HTMLElement;
  scene: THREE.Scene;
  orbitControls?: OrbitControls;
  history?: HistoryStack;
  onTransformChange?: (entityId: string, state: TransformState) => void;
  snapConfig?: Partial<SnapConfig>;
}

export class TransformGizmoController {
  private transformControls: TransformControls | null = null;
  private gizmoRoot: THREE.Object3D | null = null;
  private currentMode: TransformGizmoMode = "translate";
  private attachedObject: THREE.Object3D | null = null;
  private attachedEntityId: string | null = null;
  private snapConfig: SnapConfig;
  private roomBoundaries: RoomBoundary[] = [];
  private activeRoom: RoomBoundary | null = null;
  private entityBounds: EntityBounds = { width: 1.0, height: 1.0, depth: 1.0 };
  private initialTransform: TransformState | null = null;
  private isDragging = false;
  private history: HistoryStack;
  private scene: THREE.Scene;
  private camera: THREE.Camera;
  private domElement: HTMLElement;
  private orbitControls?: OrbitControls;
  private onTransformChange?: (entityId: string, state: TransformState) => void;

  constructor(options: TransformGizmoOptions) {
    this.scene = options.scene;
    this.camera = options.camera;
    this.domElement = options.domElement;
    this.orbitControls = options.orbitControls;
    this.history = options.history ?? getStudioHistory();
    this.onTransformChange = options.onTransformChange;
    this.snapConfig = { ...DEFAULT_SNAP_CONFIG, ...options.snapConfig };

    this.initControls();
  }

  private initControls() {
    if (typeof window === "undefined") return;

    try {
      this.transformControls = new TransformControls(this.camera, this.domElement);
      this.transformControls.setMode(this.currentMode);
      this.transformControls.size = 0.85;

      const helper = typeof (this.transformControls as unknown as { getHelper?: () => THREE.Object3D }).getHelper === "function"
        ? (this.transformControls as unknown as { getHelper: () => THREE.Object3D }).getHelper()
        : (this.transformControls as unknown as THREE.Object3D);
      this.gizmoRoot = helper;
      this.scene.add(helper);

      // Event: Dragging changed (start / end drag)
      this.transformControls.addEventListener("dragging-changed", (event: { value?: unknown }) => {
        const dragging = Boolean(event.value);
        this.isDragging = dragging;
        if (this.orbitControls) {
          this.orbitControls.enabled = !dragging;
        }

        if (dragging) {
          // Dragging started: capture baseline transform
          if (this.attachedObject) {
            this.initialTransform = this.captureState(this.attachedObject);
          }
        } else {
          // Dragging finished: evaluate delta and push to history
          if (this.attachedObject && this.initialTransform && this.attachedEntityId) {
            const finalTransform = this.captureState(this.attachedObject);
            if (this.hasTransformChanged(this.initialTransform, finalTransform)) {
              this.recordHistory(this.attachedEntityId, this.attachedObject, this.initialTransform, finalTransform);
            }
          }
          this.initialTransform = null;
        }
      });

      // Event: Real-time object transformation during drag
      this.transformControls.addEventListener("objectChange", () => {
        if (!this.attachedObject || !this.isDragging) return;
        this.applyConstraints(this.attachedObject);
      });
    } catch (e) {
      console.warn("Could not initialize Three.js TransformControls in current environment:", e);
    }
  }

  public setMode(mode: TransformGizmoMode) {
    this.currentMode = mode;
    if (this.transformControls) {
      this.transformControls.setMode(mode);
    }
  }

  public getMode(): TransformGizmoMode {
    return this.currentMode;
  }

  public setSnapConfig(config: Partial<SnapConfig>) {
    this.snapConfig = { ...this.snapConfig, ...config };
  }

  public getSnapConfig(): SnapConfig {
    return { ...this.snapConfig };
  }

  public setRoomBoundaries(boundaries: RoomBoundary[]) {
    this.roomBoundaries = boundaries;
    this.updateActiveRoom();
  }

  private updateActiveRoom() {
    if (!this.attachedObject || !this.roomBoundaries.length) {
      this.activeRoom = this.roomBoundaries[0] ?? null;
      return;
    }
    // Find room containing the entity position
    const pos = this.attachedObject.position;
    for (const room of this.roomBoundaries) {
      // Check vertical bounds first
      const elev = room.upAxis === "y" ? pos.y : pos.z;
      if (elev >= room.floorHeightM - 0.5 && elev <= room.ceilingHeightM + 0.5) {
        this.activeRoom = room;
        return;
      }
    }
    this.activeRoom = this.roomBoundaries[0] ?? null;
  }

  public attach(entityId: string, object: THREE.Object3D, bounds?: EntityBounds) {
    this.attachedEntityId = entityId;
    this.attachedObject = object;

    if (bounds) {
      this.entityBounds = bounds;
    } else {
      // Compute bounding box extents
      const box = new THREE.Box3().setFromObject(object);
      const size = new THREE.Vector3();
      box.getSize(size);
      this.entityBounds = {
        width: Math.max(0.1, size.x),
        height: Math.max(0.1, size.y),
        depth: Math.max(0.1, size.z),
      };
    }

    this.updateActiveRoom();

    if (this.transformControls) {
      this.transformControls.attach(object);
      if (this.gizmoRoot) {
        this.gizmoRoot.visible = true;
      }
    }
  }

  public detach() {
    if (this.transformControls) {
      this.transformControls.detach();
      if (this.gizmoRoot) {
        this.gizmoRoot.visible = false;
      }
    }
    this.attachedObject = null;
    this.attachedEntityId = null;
    this.initialTransform = null;
  }

  public isAttached(): boolean {
    return this.attachedObject !== null;
  }

  public isInteracting(): boolean {
    return this.isDragging || (this.transformControls ? (this.transformControls as unknown as { axis: string | null }).axis !== null : false);
  }

  public getAttachedEntityId(): string | null {
    return this.attachedEntityId;
  }

  /**
   * Applies real-time boundary snapping and collision clamping to object.
   */
  public applyConstraints(object: THREE.Object3D) {
    if (!this.snapConfig.enabled) return;

    if (this.currentMode === "translate" && this.activeRoom) {
      const candidatePos: [number, number, number] = [
        object.position.x,
        object.position.y,
        object.position.z,
      ];
      const candidateRot: [number, number, number] = [
        object.rotation.x,
        object.rotation.y,
        object.rotation.z,
      ];

      const snapResult = applyWallSnappingAndCollision(
        candidatePos,
        candidateRot,
        this.entityBounds,
        this.activeRoom,
        this.snapConfig
      );

      // Mutate object position directly for 60 FPS performance without React re-render
      object.position.set(...snapResult.snappedPosition);

      if (snapResult.snapped && snapResult.snappedRotation) {
        object.rotation.set(...snapResult.snappedRotation);
      }
      object.updateMatrixWorld(true);
    } else if (this.currentMode === "rotate" && this.snapConfig.angleSnapIncrementDeg > 0) {
      // Snap rotations to increments
      const isYup = this.activeRoom ? this.activeRoom.upAxis === "y" : true;
      if (isYup) {
        object.rotation.y = snapAngle(object.rotation.y, this.snapConfig.angleSnapIncrementDeg);
      } else {
        object.rotation.z = snapAngle(object.rotation.z, this.snapConfig.angleSnapIncrementDeg);
      }
      object.updateMatrixWorld(true);
    }
  }

  private captureState(object: THREE.Object3D): TransformState {
    return {
      position: [object.position.x, object.position.y, object.position.z],
      rotation: [object.rotation.x, object.rotation.y, object.rotation.z],
      scale: [object.scale.x, object.scale.y, object.scale.z],
    };
  }

  private hasTransformChanged(a: TransformState, b: TransformState): boolean {
    const posDist = Math.hypot(
      a.position[0] - b.position[0],
      a.position[1] - b.position[1],
      a.position[2] - b.position[2]
    );
    const rotDist = Math.hypot(
      a.rotation[0] - b.rotation[0],
      a.rotation[1] - b.rotation[1],
      a.rotation[2] - b.rotation[2]
    );
    const scaleDist = Math.hypot(
      a.scale[0] - b.scale[0],
      a.scale[1] - b.scale[1],
      a.scale[2] - b.scale[2]
    );
    return posDist > 0.001 || rotDist > 0.005 || scaleDist > 0.005;
  }

  private recordHistory(
    entityId: string,
    object: THREE.Object3D,
    before: TransformState,
    after: TransformState
  ) {
    this.history.push<TransformState>({
      type: "transform",
      description: `Transform ${entityId} (${this.currentMode})`,
      entityId,
      before,
      after,
      apply: (state: TransformState) => {
        object.position.set(...state.position);
        object.rotation.set(...state.rotation);
        object.scale.set(...state.scale);
        object.updateMatrixWorld(true);
        this.onTransformChange?.(entityId, state);
      },
    });

    this.onTransformChange?.(entityId, after);
  }

  public undo(): boolean {
    return this.history.undo() !== null;
  }

  public redo(): boolean {
    return this.history.redo() !== null;
  }

  public dispose() {
    this.detach();
    if (this.transformControls) {
      if (this.gizmoRoot && this.scene) {
        this.scene.remove(this.gizmoRoot);
      }
      this.transformControls.dispose();
      this.transformControls = null;
    }
  }
}
