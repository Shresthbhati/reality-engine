"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Cpu,
  Layers,
  Search,
  CheckCircle,
  AlertTriangle,
  Play,
  RotateCw,
  ExternalLink,
  Zap,
  Shield,
  ChevronDown,
  ChevronUp,
  X,
} from "lucide-react";
import {
  type TechnologyItem,
  type BenchmarkMetric,
  type SubsystemType,
  SUBSYSTEM_METADATA,
  filterTechnologies,
  isCopyleftLicense,
  isHardwareCompliant,
} from "@/lib/studio/technologies-model";

export interface TechnologiesPanelProps {
  onClose?: () => void;
  className?: string;
}

export function TechnologiesPanel({
  onClose,
  className = "",
}: TechnologiesPanelProps) {
  const [technologies, setTechnologies] = useState<TechnologyItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSubsystem, setSelectedSubsystem] = useState<SubsystemType | "all">("all");
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchmarkMetrics, setBenchmarkMetrics] = useState<Record<string, BenchmarkMetric>>({});
  const [isExpanded, setIsExpanded] = useState(true);

  // Fetch technologies from backend API
  useEffect(() => {
    let active = true;
    async function fetchTechs() {
      try {
        setLoading(true);
        const res = await fetch("/api/technologies");
        if (!res.ok) {
          throw new Error(`Failed to load technologies: ${res.statusText}`);
        }
        const data = await res.json();
        if (active && Array.isArray(data.items)) {
          setTechnologies(data.items);
        }
      } catch (err) {
        console.warn("Using fallback static technologies catalog", err);
      } finally {
        if (active) setLoading(false);
      }
    }
    fetchTechs();
    return () => {
      active = false;
    };
  }, []);

  const filteredItems = useMemo(() => {
    return filterTechnologies(technologies, searchQuery, selectedSubsystem);
  }, [technologies, searchQuery, selectedSubsystem]);

  const handleRunBenchmark = async (techId?: string) => {
    setBenchmarking(true);
    try {
      const url = techId
        ? `/api/technologies/benchmark?tech_id=${techId}`
        : "/api/technologies/benchmark";
      const res = await fetch(url, { method: "POST" });
      if (res.ok) {
        const body = await res.json();
        const results: BenchmarkMetric[] = Array.isArray(body.results)
          ? body.results
          : body.result ? [body.result] : [];
        const nextMap: Record<string, BenchmarkMetric> = { ...benchmarkMetrics };
        for (const m of results) {
          nextMap[m.tech_id] = m;
        }
        setBenchmarkMetrics(nextMap);
      }
    } catch (err) {
      console.error("Benchmark failed", err);
    } finally {
      setBenchmarking(false);
    }
  };

  const subsystemTabs: Array<{ id: SubsystemType | "all"; label: string }> = [
    { id: "all", label: "All Techs (50)" },
    { id: "visual_perception", label: "Perception" },
    { id: "geometric_estimation", label: "Geometry" },
    { id: "reconstruction_mapping", label: "Mapping" },
    { id: "pointcloud_3d_perception", label: "3D Cloud" },
    { id: "geometry_cad_mesh", label: "CAD & Mesh" },
    { id: "geospatial_environment", label: "GIS Terrain" },
    { id: "inference_acceleration", label: "Acceleration" },
  ];

  return (
    <div
      className={`bg-zinc-900/95 backdrop-blur-md border border-zinc-700/80 rounded-xl shadow-2xl overflow-hidden transition-all duration-200 text-zinc-200 text-xs flex flex-col max-h-[82vh] ${className}`}
      data-testid="technologies-panel"
    >
      {/* Panel Header */}
      <div className="flex items-center justify-between px-3.5 py-2.5 bg-zinc-800/80 border-b border-zinc-700/60 select-none">
        <div
          className="flex items-center gap-2 cursor-pointer flex-1"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-zinc-100 flex items-center gap-1.5">
              <span>50 Open-Source Technologies</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-indigo-500/20 text-indigo-300 font-mono">
                {technologies.length > 0 ? `${technologies.length} Evaluated` : "50"}
              </span>
            </div>
            <div className="text-[10px] text-zinc-400 flex items-center gap-1.5">
              <Shield className="w-3 h-3 text-emerald-400 inline" />
              <span>RTX 4050 6GB Bound Guard Active</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => handleRunBenchmark()}
            disabled={benchmarking}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-[11px] disabled:opacity-50 transition"
            title="Benchmark all 50 technologies"
          >
            {benchmarking ? (
              <RotateCw className="w-3 h-3 animate-spin" />
            ) : (
              <Play className="w-3 h-3 fill-current" />
            )}
            <span>Benchmark</span>
          </button>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-zinc-200"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {onClose && (
            <button
              onClick={onClose}
              className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-zinc-200"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Search & Filter Toolbar */}
          <div className="p-2.5 border-b border-zinc-800 bg-zinc-900/60 space-y-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-zinc-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search technologies, libraries, algorithms..."
                className="w-full bg-zinc-800/90 border border-zinc-700/80 rounded-lg pl-8 pr-3 py-1.5 text-zinc-200 placeholder-zinc-500 text-xs focus:outline-none focus:border-indigo-500"
              />
            </div>

            {/* Subsystem category tabs */}
            <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-thin">
              {subsystemTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedSubsystem(tab.id)}
                  className={`px-2 py-1 rounded-md text-[10px] whitespace-nowrap transition ${
                    selectedSubsystem === tab.id
                      ? "bg-zinc-700 text-zinc-100 font-medium border border-zinc-600"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Technology Cards List */}
          <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
            {loading && technologies.length === 0 ? (
              <div className="text-center py-8 text-zinc-500 flex flex-col items-center gap-2">
                <RotateCw className="w-5 h-5 animate-spin text-indigo-400" />
                <span>Loading 50 technologies catalog...</span>
              </div>
            ) : filteredItems.length === 0 ? (
              <div className="text-center py-8 text-zinc-500">
                No technologies found matching query.
              </div>
            ) : (
              filteredItems.map((tech) => {
                const metric = benchmarkMetrics[tech.id];
                const isAGPL = isCopyleftLicense(tech.license);

                return (
                  <div
                    key={tech.id}
                    className="p-2.5 rounded-lg bg-zinc-800/40 border border-zinc-700/50 hover:border-zinc-600 transition flex flex-col gap-1.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-semibold text-zinc-100 text-xs">
                          {tech.name}
                        </span>

                        {tech.top_priority && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] bg-amber-500/20 text-amber-300 border border-amber-500/30 font-medium">
                            ★ High Priority
                          </span>
                        )}

                        <span
                          className={`px-1.5 py-0.2 rounded text-[9px] border font-mono ${
                            isAGPL
                              ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                              : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          }`}
                        >
                          {tech.license}
                        </span>

                        <span className="px-1.5 py-0.2 rounded text-[9px] bg-zinc-700/50 text-zinc-400">
                          {SUBSYSTEM_METADATA[tech.subsystem]?.label ?? tech.subsystem}
                        </span>
                      </div>

                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          onClick={() => handleRunBenchmark(tech.id)}
                          disabled={benchmarking}
                          className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-indigo-300"
                          title="Run single benchmark"
                        >
                          <Play className="w-3 h-3" />
                        </button>
                        <a
                          href={tech.repository_url}
                          target="_blank"
                          rel="noreferrer"
                          className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-zinc-200"
                          title="View Repository"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>

                    <p className="text-zinc-300 text-[11px] leading-relaxed">
                      {tech.role}
                    </p>

                    <div className="flex items-center justify-between text-[10px] text-zinc-400 pt-1 border-t border-zinc-800">
                      <span className="font-mono text-zinc-500 truncate max-w-[240px]">
                        Target: {tech.reality_engine_integration_target}
                      </span>

                      {metric ? (
                        <div className="flex items-center gap-2 font-mono">
                          <span className="text-emerald-400">
                            {metric.latency_ms.toFixed(1)}ms
                          </span>
                          <span className="text-zinc-400">
                            {metric.memory_peak_mb.toFixed(0)}MB RAM
                          </span>
                          <span
                            className={
                              isHardwareCompliant(metric)
                                ? "text-cyan-400"
                                : "text-rose-400"
                            }
                          >
                            {metric.vram_peak_mb.toFixed(0)}MB VRAM
                          </span>
                        </div>
                      ) : (
                        <span className="text-zinc-600 italic">Not benchmarked</span>
                      )}
                    </div>

                    {isAGPL && (
                      <div className="flex items-center gap-1 text-[9px] text-amber-400/90 bg-amber-500/10 px-2 py-0.5 rounded">
                        <AlertTriangle className="w-3 h-3 shrink-0" />
                        <span>Copyleft license advisory: evaluate as isolated sidecar or microservice.</span>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
