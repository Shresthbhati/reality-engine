import * as THREE from 'three';
import type { CameraStation, PointCloudResponse, WorldEntity, WorldIRPayload } from '../types/studio';

export type LayerVisibility = {
  geometry: boolean;
  pointCloud: boolean;
  cameras: boolean;
  uncertaintyColoring: boolean;
};

/** Interactive Three.js 3D viewport for WorldIR geometry, point clouds, and cameras. */
export class StudioScene3D {
  private readonly container: HTMLElement;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly entityGroup = new THREE.Group();
  private readonly pointGroup = new THREE.Group();
  private readonly cameraGroup = new THREE.Group();
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly onSelectEntity: (entityId: string | null) => void;
  private animId = 0;

  constructor(container: HTMLElement, onSelectEntity: (entityId: string | null) => void) {
    this.container = container;
    this.onSelectEntity = onSelectEntity;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f141c);

    const w = Math.max(container.clientWidth, 320);
    const h = Math.max(container.clientHeight, 240);
    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 500);
    this.camera.position.set(6, 5, 6);
    this.camera.lookAt(0, 1, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(8, 12, 6);
    this.scene.add(dir);

    this.scene.add(new THREE.GridHelper(20, 20, 0x2c3848, 0x1a222d));
    this.scene.add(this.entityGroup, this.pointGroup, this.cameraGroup);

    this.renderer.domElement.addEventListener('click', this.handleClick);
    window.addEventListener('resize', this.handleResize);
    this.animate();
  }

  private colorForEntity(ent: WorldEntity, uncertaintyMode: boolean): number {
    if (uncertaintyMode) {
      if (ent.confidence == null) return 0x64748b; // unmeasured
      if (ent.confidence >= 0.8) return 0x22c55e;
      if (ent.confidence >= 0.5) return 0xf59e0b;
      return 0xef4444;
    }
    switch (ent.type) {
      case 'wall':
        return 0x94a3b8;
      case 'floor':
        return 0x475569;
      case 'door':
      case 'window':
        return 0x38bdf8;
      case 'sensor':
        return 0xfacc15;
      default:
        return 0x60a5fa;
    }
  }

  setWorld(world: WorldIRPayload, uncertaintyMode = false): void {
    this.entityGroup.clear();
    for (const [eid, ent] of Object.entries(world.entities ?? {})) {
      const color = this.colorForEntity(ent, uncertaintyMode);
      for (const gid of ent.geometry_ids ?? []) {
        const g = world.geometries?.[gid];
        if (!g?.bounds_min || !g?.bounds_max) continue;
        const sx = Math.max(0.04, g.bounds_max.x - g.bounds_min.x);
        const sy = Math.max(0.04, g.bounds_max.y - g.bounds_min.y);
        const sz = Math.max(0.04, g.bounds_max.z - g.bounds_min.z);
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(sx, sz, sy),
          new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.72 }),
        );
        mesh.position.set(
          (g.bounds_min.x + g.bounds_max.x) / 2,
          (g.bounds_min.z + g.bounds_max.z) / 2,
          (g.bounds_min.y + g.bounds_max.y) / 2,
        );
        mesh.userData = { entityId: eid };
        this.entityGroup.add(mesh);
      }
    }
  }

  setPointCloud(cloud: PointCloudResponse): void {
    this.pointGroup.clear();
    if (!cloud.points?.length) return;
    const positions = new Float32Array(cloud.points.length * 3);
    for (let i = 0; i < cloud.points.length; i++) {
      const [x, y, z] = cloud.points[i];
      positions[i * 3] = x;
      positions[i * 3 + 1] = z;
      positions[i * 3 + 2] = y;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0x38bdf8, size: 0.04 });
    this.pointGroup.add(new THREE.Points(geo, mat));
  }

  setCameras(cameras: CameraStation[]): void {
    this.cameraGroup.clear();
    for (const cam of cameras) {
      const marker = new THREE.Mesh(
        new THREE.ConeGeometry(0.12, 0.28, 4),
        new THREE.MeshBasicMaterial({ color: 0xf472b6, wireframe: true }),
      );
      marker.position.set(cam.position[0], cam.position[2], cam.position[1]);
      this.cameraGroup.add(marker);
    }
  }

  setLayerVisibility(vis: LayerVisibility): void {
    this.entityGroup.visible = vis.geometry;
    this.pointGroup.visible = vis.pointCloud;
    this.cameraGroup.visible = vis.cameras;
  }

  private readonly handleClick = (ev: MouseEvent): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.entityGroup.children, false);
    const entityId = hits.length > 0 ? String(hits[0].object.userData.entityId ?? '') : null;
    this.onSelectEntity(entityId || null);
  };

  private readonly handleResize = (): void => {
    const w = Math.max(this.container.clientWidth, 320);
    const h = Math.max(this.container.clientHeight, 240);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  private readonly animate = (): void => {
    this.animId = requestAnimationFrame(this.animate);
    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void {
    cancelAnimationFrame(this.animId);
    this.renderer.domElement.removeEventListener('click', this.handleClick);
    window.removeEventListener('resize', this.handleResize);
    this.renderer.dispose();
  }
}
