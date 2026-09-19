'use client';

import { useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, PerspectiveCamera, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useREStore, ScaleLevel, ReconstructionViewMode } from '@/store/re-store';

// ── Deterministic Spatial Geometry Buffer (Initialized at Module Load) ───────

function pseudoRand(i: number, offset = 0) {
  const v = Math.sin(i * 12.9898 + offset * 78.233) * 43758.5453;
  return v - Math.floor(v);
}

const POINT_COUNT = 60_000;
const BASE_POSITIONS = new Float32Array(POINT_COUNT * 3);

for (let i = 0; i < POINT_COUNT; i++) {
  const zone = pseudoRand(i, 1);
  let x = 0,
    y = 0,
    z = 0;

  if (zone < 0.4) {
    const angle = pseudoRand(i, 2) * Math.PI * 2;
    const r = 10 * Math.sqrt(pseudoRand(i, 3));
    x = Math.cos(angle) * r;
    z = Math.sin(angle) * r;
    y = 5 + Math.sqrt(Math.max(0, 100 - x * x - z * z)) * 1.2;
  } else if (zone < 0.7) {
    x = (pseudoRand(i, 4) - 0.5) * 50;
    z = (pseudoRand(i, 5) - 0.5) * 35;
    y = pseudoRand(i, 6) * 8;
  } else {
    const dist = 25 + pseudoRand(i, 7) * 30;
    const angle = pseudoRand(i, 8) * Math.PI * 2;
    x = Math.cos(angle) * dist;
    z = Math.sin(angle) * dist;
    y = (pseudoRand(i, 9) - 0.5) * 1.5;
  }

  BASE_POSITIONS[i * 3] = x;
  BASE_POSITIONS[i * 3 + 1] = y;
  BASE_POSITIONS[i * 3 + 2] = z;
}

// ── Axes Helper ───────────────────────────────────────────────────────────────

function AxesHelper({ size = 15 }: { size?: number }) {
  const axes = useMemo(() => {
    const createAxis = (dir: THREE.Vector3, color: string) => {
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        dir.clone().multiplyScalar(size),
      ]);
      return new THREE.Line(geo, mat);
    };
    return [
      createAxis(new THREE.Vector3(1, 0, 0), '#e74c3c'), // X Red (East)
      createAxis(new THREE.Vector3(0, 1, 0), '#2ecc71'), // Y Green (Up)
      createAxis(new THREE.Vector3(0, 0, 1), '#3d8ef7'), // Z Blue (North)
    ];
  }, [size]);

  return (
    <group>
      {axes.map((line, i) => (
        <primitive key={i} object={line} />
      ))}
    </group>
  );
}

// ── Camera Frustum Pyramids ───────────────────────────────────────────────────

function CameraFrustums() {
  const frustums = useMemo(() => {
    const items: Array<{ pos: [number, number, number]; rot: [number, number, number] }> = [];
    const radius = 45;
    for (let i = 0; i < 16; i++) {
      const angle = (i / 16) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = 20 + Math.sin(i * 1.5) * 6;
      items.push({ pos: [x, y, z], rot: [0, -angle - Math.PI / 2, 0] });
    }
    return items;
  }, []);

  return (
    <group>
      {frustums.map((c, idx) => (
        <group key={idx} position={c.pos} rotation={c.rot}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <coneGeometry args={[2.5, 5, 4, 1, true]} />
            <meshBasicMaterial color="#3d8ef7" wireframe transparent opacity={0.4} />
          </mesh>
          <mesh>
            <sphereGeometry args={[0.5, 8, 8]} />
            <meshBasicMaterial color="#3d8ef7" />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// ── Multi-Shading Point Cloud ─────────────────────────────────────────────────

function ArchitecturalPointCloud({ shadingMode }: { shadingMode: string }) {
  const colors = useMemo(() => {
    const col = new Float32Array(POINT_COUNT * 3);
    const color = new THREE.Color();

    for (let i = 0; i < POINT_COUNT; i++) {
      const x = BASE_POSITIONS[i * 3];
      const y = BASE_POSITIONS[i * 3 + 1];
      const z = BASE_POSITIONS[i * 3 + 2];

      if (shadingMode === 'CONFIDENCE') {
        const conf = Math.max(0.4, 1.0 - Math.abs(x) / 60);
        if (conf > 0.85) {
          color.setRGB(0.18, 0.8, 0.44);
        } else if (conf > 0.7) {
          color.setRGB(0.95, 0.61, 0.07);
        } else {
          color.setRGB(0.91, 0.3, 0.24);
        }
      } else if (shadingMode === 'COVERAGE') {
        const dens = Math.sin(x * 0.2) * 0.5 + 0.5;
        color.setRGB(0.24, 0.56 + dens * 0.3, 0.97);
      } else if (shadingMode === 'DEPTH') {
        const t = (z + 30) / 60;
        color.setHSL(t * 0.7, 0.8, 0.5);
      } else if (shadingMode === 'NORMALS') {
        color.setRGB(Math.abs(x) / 30, Math.abs(y) / 20, Math.abs(z) / 30);
      } else {
        const lum = 0.75 + (y / 25) * 0.25;
        color.setRGB(lum * 0.95, lum * 0.95, lum * 1.0);
      }

      col[i * 3] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }

    return col;
  }, [shadingMode]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[BASE_POSITIONS, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.16} vertexColors sizeAttenuation transparent opacity={0.88} />
    </points>
  );
}

// ── Reconstructed Monument Architectural Massing ──────────────────────────────

function BenchmarkArchitecturalMassing({
  category,
  isSelected,
  viewMode = 'GEOMETRY',
  scaleLevel = 'STRUCTURE',
}: {
  category?: string;
  isSelected: boolean;
  viewMode?: ReconstructionViewMode;
  scaleLevel?: ScaleLevel;
}) {
  const isSemantic = viewMode === 'SEMANTICS';
  const isConfidence = viewMode === 'CONFIDENCE';
  const isDetail = viewMode === 'DETAIL' || scaleLevel === 'DETAIL' || scaleLevel === 'MICRO_DETAIL';
  const isWire = viewMode === 'GEOMETRY';

  const domeColor = isSemantic ? '#38bdf8' : isConfidence ? '#22c55e' : isSelected ? '#3d8ef7' : '#e2e4ea';
  const wallColor = isSemantic ? '#94a3b8' : isConfidence ? '#22c55e' : isSelected ? '#2563eb' : '#c8cbd5';
  const columnColor = isSemantic ? '#f59e0b' : isConfidence ? '#22c55e' : '#e2e8f0';
  const reliefColor = isSemantic ? '#a855f7' : isConfidence ? '#22c55e' : '#d8b4fe';

  if (category === 'SKYSCRAPER') {
    return (
      <group position={[0, 0, 0]}>
        <mesh position={[0, 8, 0]}>
          <boxGeometry args={[18, 16, 18]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#2b3345'} roughness={0.3} metalness={0.6} />
        </mesh>
        <mesh position={[0, 24, 0]}>
          <boxGeometry args={[12, 16, 12]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#333d52'} roughness={0.3} metalness={0.6} />
        </mesh>
        <mesh position={[0, 40, 0]}>
          <boxGeometry args={[7, 16, 7]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#3b475f'} roughness={0.3} metalness={0.6} />
        </mesh>
        <mesh position={[0, 56, 0]}>
          <cylinderGeometry args={[0.3, 1.2, 16, 8]} />
          <meshStandardMaterial color="#94a3b8" roughness={0.2} metalness={0.8} />
        </mesh>
      </group>
    );
  }

  if (category === 'INDIAN_FORT') {
    return (
      <group position={[0, 0, 0]}>
        <mesh position={[0, 6, 0]}>
          <boxGeometry args={[44, 12, 28]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#8c6b45'} roughness={0.8} />
        </mesh>
        <mesh position={[0, 3, 20]}>
          <boxGeometry args={[60, 6, 4]} />
          <meshStandardMaterial color="#735432" roughness={0.9} />
        </mesh>
        <mesh position={[-28, 5, 20]}>
          <cylinderGeometry args={[4, 4, 10, 16]} />
          <meshStandardMaterial color="#735432" roughness={0.85} />
        </mesh>
        <mesh position={[28, 5, 20]}>
          <cylinderGeometry args={[4, 4, 10, 16]} />
          <meshStandardMaterial color="#735432" roughness={0.85} />
        </mesh>
      </group>
    );
  }

  if (category === 'WORLD_WONDER') {
    return (
      <group position={[0, 0, 0]}>
        <mesh position={[0, 4, 0]}>
          <boxGeometry args={[48, 8, 48]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#9a856a'} roughness={0.7} />
        </mesh>
        <mesh position={[0, 10, 0]}>
          <boxGeometry args={[34, 6, 34]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#ab9376'} roughness={0.7} />
        </mesh>
        <mesh position={[0, 15, 0]}>
          <boxGeometry args={[20, 4, 20]} />
          <meshStandardMaterial color={isSelected ? '#3d8ef7' : '#bfa585'} roughness={0.7} />
        </mesh>
      </group>
    );
  }

  // European Heritage / Victoria Memorial (Default with deep architectural details)
  return (
    <group position={[0, 0, 0]}>
      {/* Central Queen's Dome */}
      <mesh position={[0, 14, 0]}>
        <sphereGeometry args={[9, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.5]} />
        <meshStandardMaterial
          color={domeColor}
          roughness={0.25}
          metalness={0.1}
          wireframe={isWire && viewMode === 'GEOMETRY'}
          transparent
          opacity={0.88}
        />
      </mesh>

      {/* Main Building Massing Block */}
      <mesh position={[0, 4, 0]}>
        <boxGeometry args={[48, 8, 32]} />
        <meshStandardMaterial
          color={wallColor}
          roughness={0.4}
          wireframe={isWire && viewMode === 'GEOMETRY'}
          transparent
          opacity={0.82}
        />
      </mesh>

      {/* Front North Portico Colonnade (6 Fluted Columns) */}
      <group position={[0, 0, 17]}>
        {[-10, -6, -2, 2, 6, 10].map((cx, i) => (
          <group key={i} position={[cx, 0, 0]}>
            {/* Column Shaft */}
            <mesh position={[0, 4.5, 0]}>
              <cylinderGeometry args={[0.45, 0.5, 9, 16]} />
              <meshStandardMaterial color={columnColor} roughness={0.3} />
            </mesh>
            {/* Ionic Capital with Volutes */}
            <mesh position={[0, 9.2, 0]}>
              <boxGeometry args={[1.4, 0.6, 1.1]} />
              <meshStandardMaterial color={isSemantic ? '#a855f7' : columnColor} roughness={0.25} />
            </mesh>
          </group>
        ))}

        {/* Triangular Pediment Above Colonnade */}
        <mesh position={[0, 11, 0]} rotation={[0, 0, 0]}>
          <coneGeometry args={[12, 3, 3]} />
          <meshStandardMaterial color={wallColor} roughness={0.35} />
        </mesh>

        {/* High-Resolution Carved Floral Relief Medallion (Disclosed at Detail / Micro Detail levels) */}
        {isDetail && (
          <group position={[0, 10.5, 1.2]}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[1.2, 1.2, 0.25, 24]} />
              <meshStandardMaterial color={reliefColor} roughness={0.2} metalness={0.15} />
            </mesh>
            <mesh position={[0, 0, 0.15]}>
              <torusGeometry args={[0.9, 0.12, 12, 24]} />
              <meshStandardMaterial color="#f0abfc" roughness={0.15} />
            </mesh>
          </group>
        )}
      </group>

      {/* Wireframe Accent */}
      <mesh position={[0, 4, 0]}>
        <boxGeometry args={[48.2, 8.2, 32.2]} />
        <meshBasicMaterial color="#3d8ef7" wireframe transparent opacity={0.18} />
      </mesh>
    </group>
  );
}

// ── Spatial Measurement Dimension Line with Uncertainty ───────────────────────

function SpatialMeasurementLine() {
  const points = useMemo(() => {
    return [new THREE.Vector3(-24, 8.5, 16.2), new THREE.Vector3(24, 8.5, 16.2)];
  }, []);

  const lineGeo = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  const lineObj = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#a855f7', linewidth: 3 });
    return new THREE.Line(lineGeo, mat);
  }, [lineGeo]);

  return (
    <group>
      <primitive object={lineObj} />

      <mesh position={[-24, 8.5, 16.2]}>
        <sphereGeometry args={[0.35, 8, 8]} />
        <meshBasicMaterial color="#a855f7" />
      </mesh>
      <mesh position={[24, 8.5, 16.2]}>
        <sphereGeometry args={[0.35, 8, 8]} />
        <meshBasicMaterial color="#a855f7" />
      </mesh>

      <Html position={[0, 10.5, 16.2]} center distanceFactor={45}>
        <div className="px-2 py-0.5 rounded bg-[#0f1014]/90 border border-[#a855f7]/50 shadow-xl text-[10px] font-mono whitespace-nowrap select-none pointer-events-none text-[#ededf2]">
          <span className="text-[#a855f7] font-bold">48.32 m</span>
          <span className="text-[#c4c7d4] ml-1">± 0.04 m</span>
        </div>
      </Html>
    </group>
  );
}

// ── Main Scene ────────────────────────────────────────────────────────────────

interface SceneProps {
  showGrid: boolean;
  showCameras: boolean;
  showPointCloud: boolean;
  showMesh: boolean;
  shadingMode: string;
  selectedEntityId: string | null;
  benchmarkCategory?: string;
  onCoord: (v: THREE.Vector3) => void;
  viewMode?: ReconstructionViewMode;
  scaleLevel?: ScaleLevel;
}

function Scene({
  showGrid,
  showCameras,
  showPointCloud,
  showMesh,
  shadingMode,
  selectedEntityId,
  benchmarkCategory,
  onCoord,
  viewMode,
  scaleLevel,
}: SceneProps) {
  useFrame(({ camera }) => {
    onCoord(camera.position);
  });

  return (
    <>
      <PerspectiveCamera makeDefault fov={50} position={[48, 36, 52]} />
      <OrbitControls makeDefault target={[0, 8, 0]} maxPolarAngle={Math.PI / 2 + 0.05} />

      <ambientLight intensity={0.5} />
      <directionalLight position={[60, 90, 40]} intensity={1.4} castShadow />
      <directionalLight position={[-40, 20, -30]} intensity={0.4} />

      <AxesHelper size={15} />

      {showGrid && (
        <Grid
          args={[140, 140]}
          cellSize={2}
          cellColor="#1f222b"
          sectionColor="#2d323f"
          sectionSize={10}
          position={[0, -0.1, 0]}
          fadeDistance={160}
        />
      )}

      {showCameras && <CameraFrustums />}
      {showPointCloud && <ArchitecturalPointCloud shadingMode={shadingMode} />}
      {showMesh && (
        <BenchmarkArchitecturalMassing
          category={benchmarkCategory}
          isSelected={selectedEntityId === 'ent-structure-main' || selectedEntityId === 'ent-dome-central'}
          viewMode={viewMode}
          scaleLevel={scaleLevel}
        />
      )}

      <SpatialMeasurementLine />
    </>
  );
}

export function Viewport3D() {
  const {
    showGrid,
    showCameras,
    showPointCloud,
    showMesh,
    shadingMode,
    viewportMode,
    selection,
    activeMeasurementTool,
    selectedBenchmarkId,
    benchmarks,
    reconstructionViewMode,
    activeScaleLevel,
  } = useREStore();

  const activeBenchmark = useMemo(
    () => benchmarks.find((b) => b.id === selectedBenchmarkId) || benchmarks[0],
    [benchmarks, selectedBenchmarkId]
  );

  const cameraCoordsDomRef = useRef<HTMLSpanElement>(null);

  const selectedEntityId =
    selection.selectedEntityIds.length > 0 ? selection.selectedEntityIds[0] : null;

  const handleCoord = useCallback((v: THREE.Vector3) => {
    if (cameraCoordsDomRef.current) {
      cameraCoordsDomRef.current.textContent = `CAMERA: X ${v.x.toFixed(1)}   Y ${v.y.toFixed(1)}   Z ${v.z.toFixed(1)}`;
    }
  }, []);

  return (
    <div className="relative flex-1 h-full w-full overflow-hidden bg-[#08090b]">
      <Canvas gl={{ antialias: true, alpha: false }} style={{ width: '100%', height: '100%' }}>
        <Scene
          showGrid={showGrid}
          showCameras={showCameras}
          showPointCloud={showPointCloud}
          showMesh={showMesh}
          shadingMode={shadingMode}
          selectedEntityId={selectedEntityId}
          benchmarkCategory={activeBenchmark?.category}
          onCoord={handleCoord}
          viewMode={reconstructionViewMode}
          scaleLevel={activeScaleLevel}
        />
      </Canvas>

      {/* ── Floating Overlays ── */}
      <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none select-none">
        <div className="flex items-center gap-1.5">
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider text-[#3d8ef7] bg-[#0f1014]/90 border border-[#3d8ef7]/40 shadow-lg">
            {viewportMode}
          </span>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono text-[#2ecc71] bg-[#0f1014]/90 border border-[#1f222b]">
            SHADING: {shadingMode}
          </span>
          {activeBenchmark && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono text-[#f0f1f6] bg-[#0f1014]/90 border border-[#1f222b] truncate max-w-xs">
              TARGET: {activeBenchmark.name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <span
            ref={cameraCoordsDomRef}
            className="px-2 py-0.5 rounded text-[10px] font-mono text-[#9296a6] bg-[#0f1014]/85 border border-[#1f222b] num-tabular"
          >
            CAMERA: X 48.0 &nbsp; Y 36.0 &nbsp; Z 52.0
          </span>
          {activeBenchmark?.captureAvailability === 'NOT_YET_CAPTURED' && (
            <span className="px-2 py-0.5 rounded text-[9px] font-mono text-amber-400 bg-[#0f1014]/90 border border-amber-500/40">
              Pending Field Capture
            </span>
          )}
        </div>
      </div>

      <div className="absolute top-3 right-3 pointer-events-none select-none">
        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#0f1014]/90 border border-[#1f222b] text-[#2ecc71] num-tabular">
          60 FPS (Vsync)
        </span>
      </div>

      <div className="absolute bottom-3 left-3 flex items-center gap-2 pointer-events-none select-none">
        {selectedEntityId && (
          <span className="px-2.5 py-1 rounded-md text-[11px] font-mono text-[#3d8ef7] bg-[#0f1014]/95 border border-[#3d8ef7]/40 shadow-xl">
            Selected Node: <strong>{selectedEntityId}</strong>
          </span>
        )}
        {activeMeasurementTool && (
          <span className="px-2.5 py-1 rounded-md text-[11px] font-mono text-[#a855f7] bg-[#0f1014]/95 border border-[#a855f7]/40 shadow-xl">
            Active Tool: <strong>{activeMeasurementTool} (± 0.04m)</strong>
          </span>
        )}
      </div>
    </div>
  );
}
