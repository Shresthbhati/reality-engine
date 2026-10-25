"use client";

import { useEffect, useRef } from "react";
import { Map as MapLibreMap, NavigationControl, LngLatBounds } from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

export type GeoKind = "origin" | "session" | "evidence";

export interface GeoPoint {
  kind: GeoKind;
  lat: number;
  lng: number;
  label: string;
}

/** kind -> CSS token used for its dots (also what the legend prints). */
export const GEO_KIND_TOKEN: Record<GeoKind, string> = {
  origin: "--primary",
  session: "--purple",
  evidence: "--warning",
};

interface Props {
  points: GeoPoint[];
  visible: Record<GeoKind, boolean>;
  onCursor?: (c: { lat: number; lng: number } | null) => void;
}

const token = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/**
 * Real geographic points on a blank basemap. There is deliberately no tile
 * source: no provider, no key, no network - the dots are the data.
 */
export default function GeoLayerMap({ points, visible, onCursor }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const readyRef = useRef(false);
  const latest = useRef({ points, visible, onCursor });
  useEffect(() => {
    latest.current = { points, visible, onCursor };
  });

  useEffect(() => {
    if (!containerRef.current) return;
    const first = latest.current.points[0];
    const map = new MapLibreMap({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "bg", type: "background", paint: { "background-color": token("--bg-base", "#0b0d12") } }],
      },
      center: first ? [first.lng, first.lat] : [0, 0],
      zoom: 15,
      attributionControl: false,
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("pts", { type: "geojson", data: collection(latest.current.points) });
      (Object.keys(GEO_KIND_TOKEN) as GeoKind[]).forEach((k) => {
        map.addLayer({
          id: `pts-${k}`,
          type: "circle",
          source: "pts",
          filter: ["==", ["get", "kind"], k],
          paint: {
            "circle-radius": k === "origin" ? 8 : 5,
            "circle-color": token(GEO_KIND_TOKEN[k], "#38bdf8"),
            "circle-stroke-width": 1.5,
            "circle-stroke-color": token("--bg-surface", "#11141a"),
          },
          layout: { visibility: latest.current.visible[k] ? "visible" : "none" },
        });
      });
      readyRef.current = true;
      fit(map, latest.current.points);
    });
    map.on("mousemove", (e) => latest.current.onCursor?.({ lat: e.lngLat.lat, lng: e.lngLat.lng }));
    map.on("mouseout", () => latest.current.onCursor?.(null));
    mapRef.current = map;
    return () => {
      readyRef.current = false;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    (map.getSource("pts") as GeoJSONSource | undefined)?.setData(collection(points));
    fit(map, points);
  }, [points]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    (Object.keys(visible) as GeoKind[]).forEach((k) =>
      map.setLayoutProperty(`pts-${k}`, "visibility", visible[k] ? "visible" : "none"),
    );
  }, [visible]);

  return <div ref={containerRef} className="w-full h-full" data-testid="geo-layer-map" />;
}

function collection(pts: GeoPoint[]) {
  return {
    type: "FeatureCollection" as const,
    features: pts.map((p) => ({
      type: "Feature" as const,
      properties: { kind: p.kind, label: p.label },
      geometry: { type: "Point" as const, coordinates: [p.lng, p.lat] },
    })),
  };
}

function fit(map: MapLibreMap, pts: GeoPoint[]) {
  if (pts.length === 0) return;
  if (pts.length === 1) return map.jumpTo({ center: [pts[0].lng, pts[0].lat], zoom: 15 });
  const b = new LngLatBounds([pts[0].lng, pts[0].lat], [pts[0].lng, pts[0].lat]);
  pts.forEach((p) => b.extend([p.lng, p.lat]));
  map.fitBounds(b, { padding: 60, maxZoom: 17, duration: 0 });
}
