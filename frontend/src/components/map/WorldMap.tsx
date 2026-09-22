"use client";

import { useEffect, useRef } from "react";
import { Map as MapLibreMap, NavigationControl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

// Free, no-key demo style — real tiles, not fabricated. Swap for the
// project's own dark cartography style once one is provisioned.
const STYLE_URL = "https://demotiles.maplibre.org/style.json";

export interface WorldMapProps {
  center?: [number, number];
  zoom?: number;
  /** Real MapLibre camera pitch in degrees. 0 = flat 2D top-down, >0 = tilted
   * 3D perspective — a genuine map capability, not a decorative toggle. */
  pitch?: number;
  bearing?: number;
  className?: string;
  /** Hands back the live map instance so callers can add real layers
   * (Session/Evidence/Place markers, a spatial-selection rectangle) without
   * WorldMap needing to know about every possible layer type. */
  onLoad?: (map: MapLibreMap) => void;
  onClick?: (lngLat: { lng: number; lat: number }) => void;
}

export default function WorldMap({ center = [0, 20], zoom = 1.4, pitch = 0, bearing = 0, className, onLoad, onClick }: WorldMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const onLoadRef = useRef(onLoad);
  const onClickRef = useRef(onClick);
  // Keep the latest callbacks in refs from an effect — refs must not be
  // written during render; the map's long-lived listeners read the ref, so
  // they always invoke the current callbacks.
  useEffect(() => {
    onLoadRef.current = onLoad;
    onClickRef.current = onClick;
  });

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new MapLibreMap({
      container: containerRef.current,
      style: STYLE_URL,
      center,
      zoom,
      pitch,
      bearing,
      attributionControl: { compact: true },
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => onLoadRef.current?.(map));
    map.on("click", (e) => onClickRef.current?.({ lng: e.lngLat.lng, lat: e.lngLat.lat }));
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to pitch/bearing changes after initial mount (e.g. a 2D/3D toggle).
  useEffect(() => {
    mapRef.current?.easeTo({ pitch, bearing, duration: 400 });
  }, [pitch, bearing]);

  return <div ref={containerRef} className={className} style={{ width: "100%", height: "100%" }} />;
}
