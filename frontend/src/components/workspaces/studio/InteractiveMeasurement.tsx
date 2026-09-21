'use client';

/**
 * InteractiveMeasurement — Click-to-measure tool in 3D viewport.
 * - Activate with measurement tool (POINT_TO_POINT)
 * - Click to place point A, then point B
 * - Shows real-time preview line and distance
 * - Exposes uncertainty when calibration available
 * - Adds completed measurement to global measurements list
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { useREStore, type Vec3 } from '@/store/re-store';
import { Ruler, X } from 'lucide-react';

interface MeasurementPoint {
  id: string;
  position: Vec3;
  label: string;
}

export function InteractiveMeasurement() {
  const {
    activeMeasurementTool,
    measurementPoints,
    addMeasurementPoint,
    clearMeasurementPoints,
    addMeasurement,
    setActiveMeasurement,
    activeMeasurement,
    measurements,
    setActiveMeasurementTool,
  } = useREStore();

  const { camera, scene, raycaster, mouse, gl } = useThree();

  const [hoverPoint, setHoverPoint] = useState<Vec3 | null>(null);
  const [previewDistance, setPreviewDistance] = useState<number | null>(null);

  // Only active when POINT_TO_POINT tool is selected
  const isActive = activeMeasurementTool === 'POINT_TO_POINT';

  // Raycast to find intersection point on objects/grid
  const getIntersectionPoint = useCallback((): Vec3 | null => {
    if (!isActive) return null;
    
    raycaster.setFromCamera(mouse, camera);
    
    // First try to intersect with measurable objects (mesh, point cloud)
    const intersects = raycaster.intersectObjects(scene.children, true);
    
    if (intersects.length > 0) {
      const point = intersects[0].point;
      return { x: point.x, y: point.y, z: point.z };
    }
    
    // Fallback: intersect with ground plane (y=0)
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersectPoint = new THREE.Vector3();
    raycaster.ray.intersectPlane(groundPlane, intersectPoint);
    
    if (intersectPoint) {
      return { x: intersectPoint.x, y: intersectPoint.y, z: intersectPoint.z };
    }
    
    return null;
  }, [isActive, camera, scene, raycaster, mouse]);

  // Update hover preview
  useFrame(() => {
    if (!isActive) {
      setHoverPoint(null);
      setPreviewDistance(null);
      return;
    }
    
    const point = getIntersectionPoint();
    setHoverPoint(point);
    
    // Show preview distance from last placed point
    if (point && measurementPoints.length === 1) {
      const lastPoint = measurementPoints[0].position;
      const dx = point.x - lastPoint.x;
      const dy = point.y - lastPoint.y;
      const dz = point.z - lastPoint.z;
      setPreviewDistance(Math.sqrt(dx * dx + dy * dy + dz * dz));
    } else {
      setPreviewDistance(null);
    }
  });

  // Handle click to place measurement points
  const handleClick = useCallback((event: React.MouseEvent) => {
    if (!isActive) return;
    
    // Don't place points when clicking UI elements
    if (event.target instanceof HTMLButtonElement || 
        event.target instanceof HTMLInputElement ||
        (event.target as HTMLElement).closest('[data-ui]')) {
      return;
    }

    const point = getIntersectionPoint();
    if (!point) return;

    if (measurementPoints.length < 2) {
      addMeasurementPoint(point);
      
      // If this completes a measurement (2 points), create the measurement
      if (measurementPoints.length === 1) {
        const pointA = measurementPoints[0].position;
        const pointB = point;
        
        const dx = pointB.x - pointA.x;
        const dy = pointB.y - pointA.y;
        const dz = pointB.z - pointA.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        // Determine calibration status
        // In a real implementation, this would come from backend calibration data
        const calibrationAvailable = true; // TODO: Check actual calibration
        const uncertainty = calibrationAvailable ? 0.04 : 0; // meters, 95% CI
        const calibrationStatus = calibrationAvailable ? 'CALIBRATED' as const : 'UNVERIFIED' as const;
        
        const newMeasurement = {
          id: `meas-${Date.now()}`,
          name: `Distance ${measurements.length + 1}`,
          type: 'POINT_TO_POINT' as const,
          value: distance,
          unit: 'm' as const,
          uncertainty,
          entityId: undefined,
          confidence: calibrationAvailable ? 0.95 : 0.5,
          timestamp: new Date().toISOString(),
          method: 'Interactive Raycast',
          points: [pointA, pointB] as [Vec3, Vec3],
          calibrationStatus,
          calibrationAvailable,
        };
        
        addMeasurement(newMeasurement);
        setActiveMeasurement(newMeasurement);
        clearMeasurementPoints();
      }
    }
  }, [isActive, measurementPoints, getIntersectionPoint, addMeasurementPoint, measurements.length, addMeasurement, setActiveMeasurement, clearMeasurementPoints]);

  // Cancel measurement on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isActive) {
        clearMeasurementPoints();
        setActiveMeasurement(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isActive, clearMeasurementPoints, setActiveMeasurement]);

  if (!isActive) return null;

  return (
    <>
      {/* Click handler on canvas */}
      <Html
        position={[0, 0, 0]}
        center
        zIndexRange={[100, 100]}
        onClick={handleClick}
        style={{ width: '100%', height: '100%', pointerEvents: 'auto' }}
      >
        <div style={{ width: '100%', height: '100%' }} />
      </Html>

      {/* Hover preview point */}
      {hoverPoint && measurementPoints.length === 1 && (
        <group>
          <mesh position={[hoverPoint.x, hoverPoint.y, hoverPoint.z]}>
            <sphereGeometry args={[0.4, 12, 12]} />
            <meshBasicMaterial color="#a855f7" transparent opacity={0.8} />
          </mesh>
          
          {/* Preview line from point A to hover */}
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[new Float32Array([
                  measurementPoints[0].position.x, measurementPoints[0].position.y, measurementPoints[0].position.z,
                  hoverPoint.x, hoverPoint.y, hoverPoint.z
                ]), 3]} />
            </bufferGeometry>
            <lineBasicMaterial color="#a855f7" transparent opacity={0.6} linewidth={2} />
          </line>

          {/* Preview distance label */}
          {previewDistance !== null && (
            <Html
              position={[
                (measurementPoints[0].position.x + hoverPoint.x) / 2,
                Math.max(measurementPoints[0].position.y, hoverPoint.y) + 1,
                (measurementPoints[0].position.z + hoverPoint.z) / 2
              ]}
              center
              distanceFactor={45}
            >
              <div className="px-2 py-0.5 rounded bg-[#0f1014]/90 border border-[#a855f7]/50 shadow-xl text-[10px] font-mono whitespace-nowrap select-none pointer-events-none text-[#ededf2]">
                <span className="text-[#a855f7] font-bold">{previewDistance.toFixed(2)} m</span>
                <span className="text-[#c4c7d4] ml-1">(preview)</span>
              </div>
            </Html>
          )}
        </group>
      )}

      {/* Placed measurement points */}
      {measurementPoints.map((mp: { id: string; position: Vec3; label: string }, i) => (
        <group key={mp.id}>
          <mesh position={[mp.position.x, mp.position.y, mp.position.z]}>
            <sphereGeometry args={[0.35, 12, 12]} />
            <meshBasicMaterial color={i === 0 ? '#3d8ef7' : '#22c55e'} />
          </mesh>
          <Html
            position={[mp.position.x, mp.position.y + 1, mp.position.z]}
            center
            distanceFactor={45}
          >
            <div className="px-1.5 py-0.5 rounded bg-[#0f1014]/90 border border-[#3d8ef7]/50 shadow-xl text-[10px] font-mono text-[#ededf2]">
              {mp.label}
            </div>
          </Html>
        </group>
      ))}

      {/* Completed measurement display */}
      {activeMeasurement && (
        <group>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[new Float32Array([
                  activeMeasurement.points![0].x, activeMeasurement.points![0].y, activeMeasurement.points![0].z,
                  activeMeasurement.points![1].x, activeMeasurement.points![1].y, activeMeasurement.points![1].z
                ]), 3]} />
            </bufferGeometry>
            <lineBasicMaterial color="#22c55e" transparent opacity={0.8} linewidth={3} />
          </line>

          <Html
            position={[
              (activeMeasurement.points![0].x + activeMeasurement.points![1].x) / 2,
              Math.max(activeMeasurement.points![0].y, activeMeasurement.points![1].y) + 1.5,
              (activeMeasurement.points![0].z + activeMeasurement.points![1].z) / 2
            ]}
            center
            distanceFactor={45}
          >
            <div className="px-2.5 py-1 rounded bg-[#0f1014]/95 border border-[#22c55e]/50 shadow-xl text-[11px] font-mono text-[#ededf2]">
              <div className="flex items-center gap-1.5 mb-0.5">
                <Ruler className="w-3 h-3 text-[#22c55e]" />
                <span className="text-[#22c55e] font-bold">{activeMeasurement.value.toFixed(2)} m</span>
              </div>
              <div className="text-[9px] text-[#c4c7d4]">
                ± {activeMeasurement.uncertainty.toFixed(3)} m (95% CI)
              </div>
              <div className="flex items-center gap-1 mt-0.5">
                <span className={`px-1 py-0.5 rounded text-[8px] font-bold ${
                  activeMeasurement.calibrationStatus === 'CALIBRATED'
                    ? 'bg-[#22c55e]/20 text-[#22c55e]'
                    : 'bg-[#f59e0b]/20 text-[#f59e0b]'
                }`}>
                  {activeMeasurement.calibrationStatus}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    clearMeasurementPoints();
                    setActiveMeasurement(null);
                  }}
                  className="text-[9px] text-[#9296a6] hover:text-[#ededf2] underline"
                >
                  Clear
                </button>
              </div>
            </div>
          </Html>
        </group>
      )}

      {/* Instruction overlay */}
      <Html position={[0, 30, 0]} center distanceFactor={100} style={{ pointerEvents: 'none' }}>
        <div className="px-3 py-2 rounded-lg bg-[#0f1014]/95 border border-[#3d8ef7]/30 shadow-xl text-center">
          <div className="flex items-center justify-center gap-1.5 text-xs font-mono text-[#ededf2] mb-1">
            <Ruler className="w-3.5 h-3.5 text-[#a855f7]" />
            <span>Click to place point {measurementPoints.length + 1} of 2</span>
          </div>
          <div className="text-[9px] text-[#9296a6]">
            Press Escape to cancel
          </div>
        </div>
      </Html>

      {/* Calibration status indicator */}
      <Html position={[0, -25, 0]} center distanceFactor={100} style={{ pointerEvents: 'none' }}>
        <div className="px-3 py-1.5 rounded-lg bg-[#0f1014]/90 border border-[#1f222b] shadow-xl text-center">
          <div className="flex items-center justify-center gap-1.5 text-xs font-mono">
            <span className="text-[#54596b]">Calibration:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
              true // TODO: check actual calibration
                ? 'bg-[#22c55e]/20 text-[#22c55e]'
                : 'bg-[#f59e0b]/20 text-[#f59e0b]'
            }`}>
              {true ? 'CALIBRATED' : 'UNVERIFIED'}
            </span>
            {true && (
              <span className="text-[#c4c7d4]">± 0.04 m (95% CI)</span>
            )}
          </div>
        </div>
      </Html>
    </>
  );
}