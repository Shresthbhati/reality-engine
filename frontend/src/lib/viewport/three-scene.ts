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

export interface ViewportLayers {
  points: boolean;
  entities: boolean;
  cameras: boolean;
  mesh: boolean;
  oversized: boolean;
  uncertainty: boolean;
  walls: boolean;
  topology: boolean;
}

export type ViewPreset = "isometric" | "top" | "front" | "side";

export interface MeasurementResult {
  p1: [number, number, number];
  p2: [number, number, number];
  distance: number;
  delta: [number, number, number];
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
  };

  private entityMeshes = new Map<string, THREE.Mesh>();
  private selectedEntityId: string | null = null;
  private isolatedSpaceId: string | null = null;
  private activeLevelIndex: number | null = null;
  private previewCorrection: { entityId: string; type: string; confidence?: number } | null = null;
  private robustExtent = 1.0;
  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  // Data cache
  private world: WorldIR | null = null;
  private pointsData: Float32Array | null = null;
  private camerasData: CamerasPayload | null = null;
  private meshData: { positions: Float32Array; indices: Uint32Array } | null = null;

  private layerVisibility: ViewportLayers = {
    points: true,
    entities: true,
    cameras: true,
    mesh: true,
    oversized: false,
    uncertainty: false,
    walls: true,
    topology: true,
  };

  private onSelectCallback?: (id: string | null) => void;
  private onHoverCallback?: (id: string | null) => void;
  private onMeasureCallback?: (res: MeasurementResult | null) => void;

  // Spatial context & measurement tools
  private gridHelper: THREE.GridHelper | null = null;
  private axesHelper: THREE.AxesHelper | null = null;
  private isMeasurementMode = false;
  private measureP1: THREE.Vector3 | null = null;
  private measureP2: THREE.Vector3 | null = null;
  private measurementGroup: THREE.Group | null = null;

  constructor(container: HTMLElement) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x08090b);

    // Grid & Coordinate Frame
    this.gridHelper = new THREE.GridHelper(20, 20, 0x1f222b, 0x151821);
    this.gridHelper.position.y = -0.01;
    this.scene.add(this.gridHelper);

    // Subtle axes indicator
    this.axesHelper = new THREE.AxesHelper(1.5);
    (this.axesHelper.material as THREE.Material).depthTest = false;
    this.axesHelper.renderOrder = 1;
    this.scene.add(this.axesHelper);

    // Measurement group
    this.measurementGroup = new THREE.Group();
    this.scene.add(this.measurementGroup);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(6, 14, 5);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x6080a0, 0.4);
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
    this.controls.dampingFactor = 0.08;
    this.controls.maxDistance = 200;
    this.controls.minDistance = 0.2;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.setupEvents();
    this.startLoop();
  }

  public setCallbacks(callbacks: {
    onSelect?: (id: string | null) => void;
    onHover?: (id: string | null) => void;
    onMeasure?: (res: MeasurementResult | null) => void;
  }) {
    this.onSelectCallback = callbacks.onSelect;
    this.onHoverCallback = callbacks.onHover;
    this.onMeasureCallback = callbacks.onMeasure;
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

  public clearMeasurement() {
    this.measureP1 = null;
    this.measureP2 = null;
    if (this.measurementGroup) {
      while (this.measurementGroup.children.length > 0) {
        const obj = this.measurementGroup.children[0];
        this.measurementGroup.remove(obj);
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry.dispose();
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      }
    }
    this.onMeasureCallback?.(null);
  }

  public loadWorldData(
    world: WorldIR,
    points: Float32Array | null,
    cameras: CamerasPayload | null,
    mesh: { positions: Float32Array; indices: Uint32Array } | null = null
  ) {
    this.world = world;
    this.pointsData = points;
    this.camerasData = cameras;
    this.meshData = mesh;

    this.rebuildScene();
  }

  private clearSceneLayers() {
    if (this.layers.points) {
      this.scene.remove(this.layers.points);
      this.layers.points.geometry.dispose();
      (this.layers.points.material as THREE.Material).dispose();
      this.layers.points = null;
    }
    if (this.layers.entities) {
      this.scene.remove(this.layers.entities);
      for (const mesh of this.entityMeshes.values()) {
        mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((m) => m.dispose());
        } else {
          mesh.material.dispose();
        }
      }
      this.entityMeshes.clear();
      this.layers.entities = null;
    }
    if (this.layers.cameras) {
      this.scene.remove(this.layers.cameras);
      this.layers.cameras = null;
    }
    if (this.layers.mesh) {
      this.scene.remove(this.layers.mesh);
      this.layers.mesh.geometry.dispose();
      (this.layers.mesh.material as THREE.Material).dispose();
      this.layers.mesh = null;
    }
    if (this.layers.selectionOutline) {
      this.scene.remove(this.layers.selectionOutline);
      this.layers.selectionOutline.geometry.dispose();
      (this.layers.selectionOutline.material as THREE.Material).dispose();
      this.layers.selectionOutline = null;
    }
    if (this.layers.topology) {
      this.scene.remove(this.layers.topology);
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

    // 1. Build Point Cloud
    if (this.pointsData && this.pointsData.length > 0) {
      const b = pointsBounds(this.pointsData);
      const span = Math.max(1e-6, b.max[1] - b.min[1]);
      const colors = new Float32Array(this.pointsData.length);

      for (let i = 0; i < this.pointsData.length; i += 3) {
        const t = (this.pointsData[i + 1] - b.min[1]) / span; // y in [0, 1]
        colors[i] = 0.25 + 0.65 * t; // r
        colors[i + 1] = 0.85; // g
        colors[i + 2] = 1.0 - 0.45 * t; // b
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(this.pointsData, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      this.layers.points = new THREE.Points(
        geo,
        new THREE.PointsMaterial({
          size: 0.045,
          vertexColors: true,
          sizeAttenuation: true,
        })
      );
      this.layers.points.visible = this.layerVisibility.points;
      this.scene.add(this.layers.points);

      // Frame camera based on robust bounds
      const rb = robustPointsBounds(this.pointsData);
      this.robustExtent = rb.extent;
      this.camera.position.set(
        rb.center[0] + rb.extent * 1.3,
        rb.center[1] + rb.extent * 0.8,
        rb.center[2] + rb.extent * 1.3
      );
      this.controls.target.set(rb.center[0], rb.center[1], rb.center[2]);
    }

    // 2. Build Entities
    if (this.world && this.world.entities) {
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
        const isOversized = this.robustExtent > 0 && extent > 4 * this.robustExtent;
        const outlierScale = isOversized ? 0.25 : 1.0;

        const baseColor = TYPE_COLORS[e.type.toLowerCase()] || TYPE_COLORS.default;
        const conf = typeof e.confidence === "number" ? e.confidence : 0.5;

        // In uncertainty mode, color shifts from green (1.0) to amber/red (0.0)
        let renderColor = baseColor;
        if (this.layerVisibility.uncertainty) {
          const hue = conf * 0.33; // 0.33 is green, 0 is red
          const hslColor = new THREE.Color().setHSL(hue, 0.9, 0.5);
          renderColor = hslColor.getHex();
        }

        const isSpaceVolume = e.type === "room" || e.type === "corridor" || e.type === "space";
        const isOpenOpening = e.type === "door" || e.type === "window";
        const baseOpacity = isSpaceVolume
          ? 0.05
          : Math.max(0.12, (0.15 + 0.35 * conf) * outlierScale);

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

      this.scene.add(this.layers.entities);
      this.updateMeshVisibilities();
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
      this.scene.add(this.layers.cameras);
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
      this.scene.add(this.layers.mesh);
    }

    // Reselect previous entity if still exists
    if (this.selectedEntityId) {
      this.selectEntity(this.selectedEntityId);
    }
  }

  public setLayerVisibility(layers: Partial<ViewportLayers>) {
    this.layerVisibility = { ...this.layerVisibility, ...layers };

    if (this.layers.points) {
      this.layers.points.visible = this.layerVisibility.points;
    }
    if (this.layers.entities) {
      this.layers.entities.visible = this.layerVisibility.entities;
    }
    this.updateMeshVisibilities();

    if (this.layers.cameras) {
      this.layers.cameras.visible = this.layerVisibility.cameras;
    }
    if (this.layers.mesh) {
      this.layers.mesh.visible = this.layerVisibility.mesh;
    }

    if (layers.topology !== undefined) {
      if (this.layers.topology) {
        this.layers.topology.visible = this.layerVisibility.topology;
      } else if (this.layerVisibility.topology && this.selectedEntityId) {
        this.renderSpatialTopology(this.selectedEntityId);
      }
    }

    // Update uncertainty shaders if uncertainty toggle changed
    if (layers.uncertainty !== undefined) {
      for (const mesh of this.entityMeshes.values()) {
        const conf = mesh.userData.confidence ?? 0.5;
        const mat = mesh.material as THREE.MeshLambertMaterial;
        if (this.layerVisibility.uncertainty) {
          const hue = conf * 0.33;
          mat.color.setHSL(hue, 0.9, 0.5);
        } else {
          mat.color.setHex(mesh.userData.baseColor);
        }
      }
    }
  }

  public setWallsVisibility(visible: boolean) {
    this.setLayerVisibility({ walls: visible });
  }

  public isolateSpace(entityId: string | null) {
    this.isolatedSpaceId = entityId;
    this.updateMeshVisibilities();
    if (entityId) {
      this.flyToEntity(entityId);
      this.renderSpatialTopology(entityId);
    } else {
      this.renderSpatialTopology(this.selectedEntityId);
    }
  }

  public setLevelFilter(levelIndex: number | null) {
    this.activeLevelIndex = levelIndex;
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
      const conf = mesh.userData.confidence ?? 0.5;
      const outlierScale = isOversized ? 0.25 : 1.0;
      const isSpaceVolume =
        mesh.userData.entityType === "room" ||
        mesh.userData.entityType === "corridor" ||
        mesh.userData.entityType === "space";
      const mat = mesh.material as THREE.MeshLambertMaterial;

      let matchesLevel = true;
      if (this.activeLevelIndex !== null && this.world?.entities?.[eid]) {
        const ent = this.world.entities[eid];
        const entLevel =
          ent.custom_properties?.level ?? ent.custom_properties?.floor_level;
        if (entLevel !== undefined) {
          matchesLevel = Number(entLevel) === this.activeLevelIndex;
        } else {
          const estLevel = Math.max(0, Math.floor(mesh.position.y / 2.8));
          matchesLevel = estLevel === this.activeLevelIndex;
        }
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
    if (this.layers.topology) {
      this.scene.remove(this.layers.topology);
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

    if (!this.layerVisibility.topology || !this.world || !selectedEntityId) return;

    const targetEntity = this.world.entities?.[selectedEntityId];
    const targetMesh = this.entityMeshes.get(selectedEntityId);
    if (!targetEntity || !targetMesh) return;

    const topologyGroup = new THREE.Group();
    topologyGroup.name = "SpatialTopologyGroup";

    const sourcePos = targetMesh.position.clone();
    const tType = targetEntity.type.toLowerCase();

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

    if (tType === "room" || tType === "space" || tType === "corridor") {
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

        let isGeoInside = false;
        if (!isContained && !isRelConnected) {
          const box = new THREE.Box3().setFromObject(targetMesh);
          isGeoInside = box.containsPoint(otherMesh.position);
        }

        if (isContained || isRelConnected || isGeoInside) {
          if (otherType === "wall") {
            addConnector(sourcePos, otherMesh.position, 0xf59e0b);
          } else if (otherType === "door" || otherType === "window") {
            addConnector(sourcePos, otherMesh.position, 0x10b981);
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
          } else if (otherType === "stairs" || otherType === "stair") {
            addConnector(sourcePos, otherMesh.position, 0xec4899);
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
      const box = new THREE.Box3().setFromObject(targetMesh);
      const bottomCenter = new THREE.Vector3(sourcePos.x, box.min.y, sourcePos.z);
      const topCenter = new THREE.Vector3(sourcePos.x, box.max.y, sourcePos.z);
      addConnector(bottomCenter, topCenter, 0xec4899);
    }

    this.layers.topology = topologyGroup;
    this.layers.topology.visible = this.layerVisibility.topology;
    this.scene.add(this.layers.topology);
  }

  public setCorrectionPreview(
    entityId: string,
    previewType: string,
    previewConfidence?: number
  ) {
    this.previewCorrection = { entityId, type: previewType, confidence: previewConfidence };
    const mesh = this.entityMeshes.get(entityId);
    if (!mesh) return;

    const mat = mesh.material as THREE.MeshLambertMaterial;
    const newColor = TYPE_COLORS[previewType.toLowerCase()] || TYPE_COLORS.default;
    mat.color.setHex(newColor);
    mat.emissive.setHex(newColor);
    mat.emissiveIntensity = 0.5;
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
      }
      this.previewCorrection = null;
    }
  }

  public applyEntityFilter(matchingIds: Set<string> | null) {
    for (const [eid, mesh] of this.entityMeshes.entries()) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const conf = mesh.userData.confidence ?? 0.5;
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

    // Reset outline
    if (this.layers.selectionOutline) {
      this.scene.remove(this.layers.selectionOutline);
      this.layers.selectionOutline.geometry.dispose();
      (this.layers.selectionOutline.material as THREE.Material).dispose();
      this.layers.selectionOutline = null;
    }

    for (const [eid, mesh] of this.entityMeshes.entries()) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      const isSel = eid === id;
      mat.emissive = new THREE.Color(isSel ? 0x00e5ff : 0x000000);
      mat.emissiveIntensity = isSel ? 0.6 : 0;

      if (isSel) {
        // Create prominent bright selection outline
        const wireGeo = new THREE.EdgesGeometry(mesh.geometry);
        const wireMat = new THREE.LineBasicMaterial({
          color: 0x00e5ff,
          linewidth: 2,
          transparent: true,
          opacity: 0.95,
        });
        this.layers.selectionOutline = new THREE.LineSegments(wireGeo, wireMat);
        this.layers.selectionOutline.position.copy(mesh.position);
        this.scene.add(this.layers.selectionOutline);

        if (updateCamera) {
          this.flyToEntity(id);
        }
      }
    }

    this.renderSpatialTopology(id);
    this.onSelectCallback?.(id);
  }

  public flyToEntity(id: string) {
    const mesh = this.entityMeshes.get(id);
    if (!mesh) return;

    const box = new THREE.Box3().setFromObject(mesh);
    const size = box.getSize(new THREE.Vector3()).length();
    const dir = new THREE.Vector3(0.8, 0.55, 0.8).normalize();
    const d = Math.max(1.2, size * 1.45);

    this.camera.position.copy(mesh.position).addScaledVector(dir, d);
    this.controls.target.copy(mesh.position);
    this.controls.update();
  }

  public flyToCamera(evidenceId: string) {
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
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quat);
    this.camera.position.set(p[0], p[1], p[2]);
    this.controls.target.set(
      p[0] + forward.x * 2.5,
      p[1] + forward.y * 2.5,
      p[2] + forward.z * 2.5
    );
    this.controls.update();
  }

  public flyToPosition(x: number, y: number, z: number) {
    this.camera.position.set(x + 2.5, y + 2, z + 2.5);
    this.controls.target.set(x, y, z);
    this.controls.update();
  }


  public frameAll() {
    if (this.pointsData && this.pointsData.length > 0) {
      const rb = robustPointsBounds(this.pointsData);
      this.camera.position.set(
        rb.center[0] + rb.extent * 1.2,
        rb.center[1] + rb.extent * 0.7,
        rb.center[2] + rb.extent * 1.2
      );
      this.controls.target.set(rb.center[0], rb.center[1], rb.center[2]);
    } else if (this.entityMeshes.size > 0) {
      const box = new THREE.Box3();
      for (const mesh of this.entityMeshes.values()) {
        box.expandByObject(mesh);
      }
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      this.camera.position.set(
        center.x + size * 0.9,
        center.y + size * 0.5,
        center.z + size * 0.9
      );
      this.controls.target.copy(center);
    } else {
      this.camera.position.set(4, 3, 6);
      this.controls.target.set(0, 0, 0);
    }
    this.controls.update();
  }

  public setViewPreset(preset: ViewPreset) {
    const target = this.controls.target.clone();
    const d = this.camera.position.distanceTo(target) || 5;

    switch (preset) {
      case "top":
        this.camera.position.set(target.x, target.y + d, target.z + 0.0001);
        break;
      case "front":
        this.camera.position.set(target.x, target.y, target.z + d);
        break;
      case "side":
        this.camera.position.set(target.x + d, target.y, target.z);
        break;
      case "isometric":
      default:
        this.camera.position.set(
          target.x + d * 0.7,
          target.y + d * 0.5,
          target.z + d * 0.7
        );
        break;
    }
    this.controls.update();
  }

  private setupEvents() {
    const dom = this.renderer.domElement;

    // Handle clicks for 3D selection and measurement
    dom.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      // Interactive Measurement Mode
      if (this.isMeasurementMode) {
        let pt: THREE.Vector3 | null = null;
        const meshes = Array.from(this.entityMeshes.values()).filter((m) => m.visible);
        const hits = this.raycaster.intersectObjects(meshes, false);
        if (hits.length > 0) {
          pt = hits[0].point;
        } else {
          const groundPoint = new THREE.Vector3();
          const hitGround = this.raycaster.ray.intersectPlane(
            new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
            groundPoint
          );
          if (hitGround) pt = groundPoint;
        }

        if (pt) {
          this.handleMeasurementPoint(pt);
        }
        return;
      }

      // Entity Selection Mode
      const meshes = Array.from(this.entityMeshes.values()).filter(
        (m) => m.visible
      );
      const hits = this.raycaster.intersectObjects(meshes, false);

      if (hits.length > 0) {
        const entityId = hits[0].object.userData.entityId as string;
        this.selectEntity(entityId);
      } else {
        this.selectEntity(null);
      }
    });

    // Resize observer
    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(this.container);
  }

  private handleMeasurementPoint(pt: THREE.Vector3) {
    if (!this.measurementGroup) return;

    if (!this.measureP1) {
      this.measureP1 = pt.clone();
      const marker1 = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x00e5ff, depthTest: false })
      );
      marker1.position.copy(pt);
      marker1.renderOrder = 999;
      this.measurementGroup.add(marker1);
      this.onMeasureCallback?.({
        p1: [pt.x, pt.y, pt.z],
        p2: [pt.x, pt.y, pt.z],
        distance: 0,
        delta: [0, 0, 0],
      });
    } else if (!this.measureP2) {
      this.measureP2 = pt.clone();
      const marker2 = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x35d07f, depthTest: false })
      );
      marker2.position.copy(pt);
      marker2.renderOrder = 999;
      this.measurementGroup.add(marker2);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([this.measureP1, this.measureP2]);
      const lineMat = new THREE.LineDashedMaterial({
        color: 0x00e5ff,
        dashSize: 0.1,
        gapSize: 0.05,
        depthTest: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      line.renderOrder = 998;
      this.measurementGroup.add(line);

      const d = this.measureP1.distanceTo(this.measureP2);
      const delta: [number, number, number] = [
        this.measureP2.x - this.measureP1.x,
        this.measureP2.y - this.measureP1.y,
        this.measureP2.z - this.measureP1.z,
      ];
      this.onMeasureCallback?.({
        p1: [this.measureP1.x, this.measureP1.y, this.measureP1.z],
        p2: [this.measureP2.x, this.measureP2.y, this.measureP2.z],
        distance: d,
        delta,
      });
    } else {
      this.clearMeasurement();
      this.handleMeasurementPoint(pt);
    }
  }

  private onResize() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  private startLoop() {
    const animate = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.animationFrameId = requestAnimationFrame(animate);
    };
    this.animationFrameId = requestAnimationFrame(animate);
  }

  public getStatistics() {
    const pts = this.pointsData ? this.pointsData.length / 3 : 0;
    const cams = this.camerasData ? (this.camerasData.cameras || []).length : 0;
    const ents = this.entityMeshes.size;
    return { points: pts, cameras: cams, entities: ents };
  }

  public dispose() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    this.clearSceneLayers();
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
  }
}
