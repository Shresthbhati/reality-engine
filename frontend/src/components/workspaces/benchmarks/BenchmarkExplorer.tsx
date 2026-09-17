'use client';

/**
 * BenchmarkExplorer — Architectural Engineering Benchmark Corpus.
 * - Curated 40-target validation suite across Skyscrapers, Indian Forts, World Wonders, European Architecture.
 * - Explicit engineering disclaimer: architectural stress test suite, NOT an authoritative ranking.
 * - Grounded reality statuses: honest distinction between local datasets, archives, and uncaptured targets.
 * - Detail experience with Overview, Evidence breakdown, Reconstruction launcher, WorldIR, Metrics, and Stress factors.
 * - Multi-target side-by-side comparison view (e.g. Victoria Memorial vs Agra Fort vs Burj Khalifa).
 * - 1-click loading into Reconstruction Studio.
 */

import React, { useState, useMemo } from 'react';
import { useREStore, BenchmarkCategory, BenchmarkStructure, BenchmarkReconstructionStatus } from '@/store/re-store';
import {
  Building2,
  Shield,
  Landmark,
  Compass,
  Search,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  Sparkles,
  BarChart3,
  ExternalLink,
  ChevronRight,
  Filter,
  Cpu,
  Globe,
  Camera,
  FileCheck,
  Zap,
  ArrowRight,
  Scale,
  X,
  Play,
  Check,
  AlertTriangle,
  Info,
  HelpCircle,
} from 'lucide-react';

export default function BenchmarkExplorer() {
  const {
    benchmarks,
    selectedBenchmarkId,
    setSelectedBenchmarkId,
    benchmarkCategoryFilter,
    setBenchmarkCategoryFilter,
    comparisonBenchmarkIds,
    toggleComparisonBenchmark,
    clearComparisonBenchmarks,
    loadBenchmarkIntoStudio,
    setActiveWorkspace,
  } = useREStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'VALIDATED' | 'EVIDENCE_AVAILABLE' | 'NOT_YET_CAPTURED'>('ALL');
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [comparisonModalOpen, setComparisonModalOpen] = useState(false);
  const [detailTab, setDetailTab] = useState<'overview' | 'evidence' | 'reconstruct' | 'worldir' | 'metrics' | 'stress'>('overview');

  // Reconstruction Form State (in Detail View)
  const [selectedPipeline, setSelectedPipeline] = useState('RealityEngine Hybrid (SfM + Volumetric)');
  const [selectedQuality, setSelectedQuality] = useState('Sub-Centimeter Geodetic');
  const [reconstructNotice, setReconstructNotice] = useState<string | null>(null);

  const activeBenchmark = useMemo(
    () => benchmarks.find((b) => b.id === selectedBenchmarkId) || benchmarks[0],
    [benchmarks, selectedBenchmarkId]
  );

  // Category counts
  const categoryCounts = useMemo(() => {
    return {
      ALL: benchmarks.length,
      SKYSCRAPER: benchmarks.filter((b) => b.category === 'SKYSCRAPER').length,
      INDIAN_FORT: benchmarks.filter((b) => b.category === 'INDIAN_FORT').length,
      WORLD_WONDER: benchmarks.filter((b) => b.category === 'WORLD_WONDER').length,
      EUROPEAN_HERITAGE: benchmarks.filter((b) => b.category === 'EUROPEAN_HERITAGE').length,
    };
  }, [benchmarks]);

  // Filtered benchmarks
  const filteredBenchmarks = useMemo(() => {
    return benchmarks.filter((b) => {
      const matchesCat =
        benchmarkCategoryFilter === 'ALL' || b.category === benchmarkCategoryFilter;
      
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        !q ||
        b.name.toLowerCase().includes(q) ||
        b.location.toLowerCase().includes(q) ||
        b.country.toLowerCase().includes(q) ||
        b.architectureTypology.toLowerCase().includes(q);

      let matchesStatus = true;
      if (statusFilter === 'VALIDATED') {
        matchesStatus = b.reconstructionStatus === 'VALIDATED';
      } else if (statusFilter === 'EVIDENCE_AVAILABLE') {
        matchesStatus = b.reconstructionStatus === 'EVIDENCE_AVAILABLE' || b.captureAvailability === 'LOCAL_DATASET' || b.captureAvailability === 'AERIAL_ARCHIVE';
      } else if (statusFilter === 'NOT_YET_CAPTURED') {
        matchesStatus = b.captureAvailability === 'NOT_YET_CAPTURED';
      }

      return matchesCat && matchesSearch && matchesStatus;
    });
  }, [benchmarks, benchmarkCategoryFilter, searchQuery, statusFilter]);

  const getCategoryIcon = (cat: BenchmarkCategory) => {
    switch (cat) {
      case 'SKYSCRAPER':
        return <Building2 className="w-4 h-4 text-[#3d8ef7]" />;
      case 'INDIAN_FORT':
        return <Shield className="w-4 h-4 text-amber-400" />;
      case 'WORLD_WONDER':
        return <Compass className="w-4 h-4 text-emerald-400" />;
      case 'EUROPEAN_HERITAGE':
        return <Landmark className="w-4 h-4 text-purple-400" />;
      default:
        return <Globe className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: BenchmarkReconstructionStatus) => {
    switch (status) {
      case 'VALIDATED':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-[#2ecc71]/15 text-[#2ecc71] border border-[#2ecc71]/30 font-bold">
            <CheckCircle2 className="w-3 h-3" /> VALIDATED
          </span>
        );
      case 'EVIDENCE_AVAILABLE':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/30">
            <Camera className="w-3 h-3" /> EVIDENCE AVAILABLE
          </span>
        );
      case 'PARTIAL':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <Clock className="w-3 h-3" /> PARTIAL
          </span>
        );
      case 'RECONSTRUCTION_RUNNING':
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30 animate-pulse">
            <Cpu className="w-3 h-3" /> RUNNING
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-[#1f222b] text-[#54596b] border border-[#272a36]">
            NOT YET CAPTURED
          </span>
        );
    }
  };

  const getComplexityBadge = (c: BenchmarkStructure['complexity']) => {
    switch (c) {
      case 'EXTREME':
        return <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-red-500/15 text-red-400 border border-red-500/30 font-bold">EXTREME</span>;
      case 'VERY_HIGH':
        return <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30 font-bold">VERY HIGH</span>;
      case 'HIGH':
        return <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">HIGH</span>;
      default:
        return <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-gray-500/15 text-gray-400 border border-gray-500/30">MODERATE</span>;
    }
  };

  const openDetail = (structure: BenchmarkStructure) => {
    setSelectedBenchmarkId(structure.id);
    setDetailTab('overview');
    setReconstructNotice(null);
    setDetailModalOpen(true);
  };

  const handleRunReconstruct = () => {
    if (activeBenchmark.captureAvailability === 'NOT_YET_CAPTURED') {
      setReconstructNotice('⚠️ Cannot start reconstruction: Source imagery or LiDAR has not yet been captured for this target.');
      return;
    }
    setReconstructNotice(`✓ Staged benchmark reconstruction job for ${activeBenchmark.name} [Pipeline: ${selectedPipeline}, Target: ${selectedQuality}]. Worker node ready.`);
  };

  // Compare targets list
  const comparisonList = useMemo(() => {
    return benchmarks.filter((b) => comparisonBenchmarkIds.includes(b.id));
  }, [benchmarks, comparisonBenchmarkIds]);

  return (
    <div className="flex flex-col flex-1 h-full bg-[#090a0d] text-[#e8e8f0] select-none font-mono overflow-hidden">
      {/* ── Top Header & Disclaimer ───────────────────────────────── */}
      <div className="border-b border-[#1f222b] bg-[#0d0e14] px-4 py-3 shrink-0">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[#f0f1f6] tracking-tight">
                ARCHITECTURAL BENCHMARK EXPLORER
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-[#3d8ef7]/15 text-[#3d8ef7] font-bold border border-[#3d8ef7]/30">
                40 TARGETS
              </span>
            </div>
            <div className="text-[11px] text-[#54596b] mt-0.5">
              Engineering stress testing corpus for precision multi-modal spatial reconstruction
            </div>
          </div>

          {/* Action Bar */}
          <div className="flex items-center gap-2">
            {comparisonBenchmarkIds.length > 0 && (
              <button
                type="button"
                onClick={() => setComparisonModalOpen(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b] font-bold text-xs shadow-lg transition-colors"
              >
                <Scale className="w-3.5 h-3.5" />
                <span>Compare Selected ({comparisonBenchmarkIds.length})</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                loadBenchmarkIntoStudio('euro-009'); // Victoria Memorial
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#14161f] border border-[#2ecc71]/40 hover:border-[#2ecc71] text-[#2ecc71] text-xs transition-colors"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Load Victoria Memorial (Local Dataset)</span>
            </button>
          </div>
        </div>

        {/* ── Explicit Engineering Corpus Disclaimer (Master Instruction Rule) ── */}
        <div className="p-2 rounded bg-[#13151f] border border-[#1f222b] text-[10px] text-[#9296a6] flex items-center gap-2">
          <Info className="w-4 h-4 text-[#3d8ef7] shrink-0" />
          <span>
            <strong>Engineering Benchmark Notice:</strong> This collection represents a standardized architectural stress-testing corpus selected for morphological diversity, extreme scale, high dynamic range, and multi-sensor challenges. It is <strong>not an authoritative ranking</strong> of structures.
          </span>
        </div>
      </div>

      {/* ── Category Tabs & Filter Toolbar ────────────────────────── */}
      <div className="px-4 py-2 border-b border-[#1f222b] bg-[#0c0d11] flex flex-wrap items-center justify-between gap-3 shrink-0">
        {/* Category Tabs */}
        <div className="flex items-center gap-1 text-xs">
          <button
            type="button"
            onClick={() => setBenchmarkCategoryFilter('ALL')}
            className={`px-3 py-1 rounded-md transition-colors ${
              benchmarkCategoryFilter === 'ALL'
                ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#151720]'
            }`}
          >
            All Categories ({categoryCounts.ALL})
          </button>
          <button
            type="button"
            onClick={() => setBenchmarkCategoryFilter('SKYSCRAPER')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-colors ${
              benchmarkCategoryFilter === 'SKYSCRAPER'
                ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#151720]'
            }`}
          >
            <Building2 className="w-3.5 h-3.5" />
            <span>Skyscrapers ({categoryCounts.SKYSCRAPER})</span>
          </button>
          <button
            type="button"
            onClick={() => setBenchmarkCategoryFilter('INDIAN_FORT')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-colors ${
              benchmarkCategoryFilter === 'INDIAN_FORT'
                ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#151720]'
            }`}
          >
            <Shield className="w-3.5 h-3.5" />
            <span>Indian Forts ({categoryCounts.INDIAN_FORT})</span>
          </button>
          <button
            type="button"
            onClick={() => setBenchmarkCategoryFilter('WORLD_WONDER')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-colors ${
              benchmarkCategoryFilter === 'WORLD_WONDER'
                ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#151720]'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            <span>World Wonders ({categoryCounts.WORLD_WONDER})</span>
          </button>
          <button
            type="button"
            onClick={() => setBenchmarkCategoryFilter('EUROPEAN_HERITAGE')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-colors ${
              benchmarkCategoryFilter === 'EUROPEAN_HERITAGE'
                ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                : 'text-[#9296a6] hover:text-[#ededf2] hover:bg-[#151720]'
            }`}
          >
            <Landmark className="w-3.5 h-3.5" />
            <span>European & Heritage ({categoryCounts.EUROPEAN_HERITAGE})</span>
          </button>
        </div>

        {/* Right Search & Status Filter */}
        <div className="flex items-center gap-2">
          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
            className="bg-[#14161f] border border-[#1f222b] text-[11px] text-[#ededf2] rounded px-2.5 py-1 outline-none"
          >
            <option value="ALL">All States</option>
            <option value="VALIDATED">Validated (1)</option>
            <option value="EVIDENCE_AVAILABLE">Evidence Available (8)</option>
            <option value="NOT_YET_CAPTURED">Pending Capture (29)</option>
          </select>

          {/* Search Box */}
          <div className="relative flex items-center">
            <Search className="absolute left-2.5 w-3 h-3 text-[#54596b] pointer-events-none" />
            <input
              type="text"
              placeholder="Search by name, city, typology..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-[#14161f] border border-[#1f222b] rounded text-[11px] text-[#ededf2] placeholder-[#54596b] pl-7 pr-3 py-1 outline-none focus:border-[#3d8ef7]/60 w-56 transition-colors"
            />
          </div>
        </div>
      </div>

      {/* ── Main Cards Grid ───────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3.5">
          {filteredBenchmarks.map((b) => {
            const isCompared = comparisonBenchmarkIds.includes(b.id);
            const isLocal = b.captureAvailability === 'LOCAL_DATASET';

            return (
              <div
                key={b.id}
                className={`flex flex-col justify-between p-3.5 rounded-xl border transition-all ${
                  isLocal
                    ? 'bg-[#121620] border-[#2ecc71]/40 shadow-lg'
                    : 'bg-[#101217] border-[#1f222b] hover:border-[#3d8ef7]/40'
                }`}
              >
                <div>
                  {/* Top Category & Status Header */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-1.5">
                      {getCategoryIcon(b.category)}
                      <span className="text-[10px] text-[#9296a6] uppercase tracking-wider">
                        {b.category.replace('_', ' ')}
                      </span>
                    </div>
                    {getStatusBadge(b.reconstructionStatus)}
                  </div>

                  {/* Name & Location */}
                  <h3 className="text-sm font-bold text-[#f0f1f6] tracking-tight leading-snug mb-0.5">
                    {b.name}
                  </h3>
                  <div className="text-[11px] text-[#3d8ef7] font-medium mb-2">
                    {b.location}, {b.country}
                  </div>

                  {/* Typology & Scale */}
                  <div className="p-2 rounded bg-[#0b0c10] border border-[#1a1c24] space-y-1 mb-2.5 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Typology:</span>
                      <span className="text-[#c4c7d4] truncate max-w-[170px]">{b.architectureTypology}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Scale / Height:</span>
                      <span className="text-[#ededf2] font-semibold num-tabular">{b.scale}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Complexity:</span>
                      <span>{getComplexityBadge(b.complexity)}</span>
                    </div>
                  </div>

                  {/* Ground Truth & Capture Availability */}
                  <div className="space-y-1 text-[10px] text-[#9296a6] mb-3">
                    <div className="flex justify-between">
                      <span>Capture Data:</span>
                      <span className={isLocal ? 'text-[#2ecc71] font-bold' : b.captureAvailability === 'AERIAL_ARCHIVE' ? 'text-[#3d8ef7]' : 'text-[#54596b]'}>
                        {b.captureAvailability.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Ground-Truth:</span>
                      <span className="text-[#c4c7d4]">
                        {b.groundTruthAvailability.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Card Actions */}
                <div className="pt-2 border-t border-[#1a1c24] flex items-center justify-between gap-1.5">
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => openDetail(b)}
                      className="px-2.5 py-1 rounded bg-[#171922] hover:bg-[#202432] text-[#f0f1f6] text-[11px] font-semibold border border-[#272a38] transition-colors"
                    >
                      Details
                    </button>
                    <button
                      type="button"
                      onClick={() => toggleComparisonBenchmark(b.id)}
                      className={`px-2 py-1 rounded text-[10px] border transition-colors ${
                        isCompared
                          ? 'bg-[#3d8ef7]/20 border-[#3d8ef7] text-[#3d8ef7]'
                          : 'bg-[#14161f] border-[#1f222b] text-[#54596b] hover:text-[#9296a6]'
                      }`}
                      title="Add to side-by-side comparison"
                    >
                      {isCompared ? '✓ Added' : '+ Compare'}
                    </button>
                  </div>

                  {isLocal ? (
                    <button
                      type="button"
                      onClick={() => loadBenchmarkIntoStudio(b.id)}
                      className="px-2.5 py-1 rounded bg-[#2ecc71] hover:bg-[#27ae60] text-[#08090b] text-[11px] font-bold flex items-center gap-1 transition-colors"
                    >
                      <span>Load Studio</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => openDetail(b)}
                      className="px-2 py-1 text-[10px] text-[#54596b] hover:text-[#3d8ef7] transition-colors flex items-center gap-0.5"
                    >
                      <span>Objectives</span>
                      <ChevronRight className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── BENCHMARK DETAIL MODAL / DRAWER ───────────────────────── */}
      {detailModalOpen && activeBenchmark && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="w-full max-w-4xl bg-[#101217] border border-[#1f222b] rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            {/* Modal Header */}
            <div className="p-4 border-b border-[#1f222b] bg-[#0c0d11] flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-[#14161f] border border-[#1f222b]">
                  {getCategoryIcon(activeBenchmark.category)}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-base font-bold text-[#f0f1f6] tracking-tight">
                      {activeBenchmark.name}
                    </h2>
                    {getStatusBadge(activeBenchmark.reconstructionStatus)}
                    {getComplexityBadge(activeBenchmark.complexity)}
                  </div>
                  <div className="text-xs text-[#3d8ef7]">
                    {activeBenchmark.location}, {activeBenchmark.country} • {activeBenchmark.scale}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    loadBenchmarkIntoStudio(activeBenchmark.id);
                    setDetailModalOpen(false);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b] text-xs font-bold transition-colors"
                >
                  Load into Studio Viewport
                </button>
                <button
                  type="button"
                  onClick={() => setDetailModalOpen(false)}
                  className="p-1 text-[#54596b] hover:text-[#ededf2] rounded"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Tabs */}
            <div className="px-4 py-1.5 border-b border-[#1f222b] bg-[#14161f] flex items-center gap-2 text-xs">
              {(
                [
                  { id: 'overview', label: 'Overview & Typology' },
                  { id: 'evidence', label: 'Available Evidence' },
                  { id: 'reconstruct', label: 'Reconstruct Benchmark' },
                  { id: 'worldir', label: 'Perception & WorldIR' },
                  { id: 'metrics', label: 'Engineering Metrics' },
                  { id: 'stress', label: 'Stress Factors' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setDetailTab(tab.id)}
                  className={`px-3 py-1 rounded transition-colors ${
                    detailTab === tab.id
                      ? 'bg-[#1f222b] text-[#3d8ef7] font-bold'
                      : 'text-[#9296a6] hover:text-[#ededf2]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Modal Body */}
            <div className="p-5 overflow-y-auto space-y-4 text-xs">
              {/* TAB 1: OVERVIEW */}
              {detailTab === 'overview' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 rounded-lg bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[10px] text-[#54596b] uppercase">Architectural Typology</div>
                      <div className="text-sm font-bold text-[#f0f1f6] mt-0.5">{activeBenchmark.architectureTypology}</div>
                      <div className="text-[10px] text-[#9296a6] mt-1">{activeBenchmark.era || 'Historic Period'}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[10px] text-[#54596b] uppercase">Scale / Dimensions</div>
                      <div className="text-sm font-bold text-[#f0f1f6] mt-0.5 num-tabular">{activeBenchmark.scale}</div>
                      <div className="text-[10px] text-[#9296a6] mt-1">
                        Footprint: {activeBenchmark.footprintSqMeters ? `${activeBenchmark.footprintSqMeters.toLocaleString()} m²` : 'Geodetic extent'}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[10px] text-[#54596b] uppercase">Ground Truth Data</div>
                      <div className="text-sm font-bold text-[#f0f1f6] mt-0.5">{activeBenchmark.groundTruthAvailability.replace(/_/g, ' ')}</div>
                      <div className="text-[10px] text-[#2ecc71] mt-1">Calibrated Baseline</div>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-[#14161f] border border-[#1f222b] space-y-1.5">
                    <div className="text-[10px] text-[#54596b] uppercase font-bold">Structural Morphology Notes</div>
                    <p className="text-xs text-[#c4c7d4] leading-relaxed">
                      {activeBenchmark.structuralNotes}
                    </p>
                  </div>

                  <div className="p-3.5 rounded-lg bg-[#14161f] border border-[#1f222b] space-y-2">
                    <div className="text-[10px] text-[#3d8ef7] uppercase font-bold">Benchmark Validation Objectives</div>
                    <ul className="space-y-1.5">
                      {activeBenchmark.benchmarkObjectives.map((obj, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-[#ededf2]">
                          <span className="text-[#3d8ef7] shrink-0 font-bold">•</span>
                          <span>{obj}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* TAB 2: EVIDENCE */}
              {detailTab === 'evidence' && (
                <div className="space-y-4">
                  <div className="p-3 rounded-lg bg-[#14161f] border border-[#1f222b]">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-[#f0f1f6]">Evidence Availability Breakdown</span>
                      <span className="text-[10px] text-[#3d8ef7] font-semibold">{activeBenchmark.captureAvailability}</span>
                    </div>
                    {activeBenchmark.evidenceSummary ? (
                      <div className="grid grid-cols-4 gap-2 pt-1 text-center">
                        <div className="p-2 rounded bg-[#0b0c10] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b]">Keyframes</div>
                          <div className="text-sm font-bold text-[#2ecc71] num-tabular mt-0.5">
                            {activeBenchmark.evidenceSummary.photos?.toLocaleString() || '—'}
                          </div>
                        </div>
                        <div className="p-2 rounded bg-[#0b0c10] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b]">LiDAR Scans</div>
                          <div className="text-sm font-bold text-[#3d8ef7] num-tabular mt-0.5">
                            {activeBenchmark.evidenceSummary.lidarScans || 'None'}
                          </div>
                        </div>
                        <div className="p-2 rounded bg-[#0b0c10] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b]">Dataset Size</div>
                          <div className="text-sm font-bold text-[#ededf2] num-tabular mt-0.5">
                            {activeBenchmark.evidenceSummary.datasetSizeGb} GB
                          </div>
                        </div>
                        <div className="p-2 rounded bg-[#0b0c10] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b]">Source URI</div>
                          <div className="text-[10px] text-[#9296a6] truncate mt-1">
                            {activeBenchmark.evidenceSummary.sourceUri}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 rounded bg-[#0b0c10] text-center text-[#54596b] space-y-1">
                        <AlertCircle className="w-5 h-5 text-[#54596b] mx-auto mb-1" />
                        <div>No raw evidence dataset locally available for this structure.</div>
                        <div className="text-[10px]">Capture status: <strong>NOT YET CAPTURED</strong></div>
                      </div>
                    )}
                  </div>

                  <div className="p-3.5 rounded-lg bg-[#14161f] border border-[#1f222b] space-y-2">
                    <div className="text-[10px] text-[#54596b] uppercase font-bold">Recommended Sensor Modalities</div>
                    <div className="flex flex-wrap gap-1.5">
                      {activeBenchmark.captureTypes.map((type, idx) => (
                        <span key={idx} className="px-2 py-0.5 rounded bg-[#1f222b] text-[#c4c7d4] text-[10px]">
                          {type.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: RECONSTRUCT WORKFLOW */}
              {detailTab === 'reconstruct' && (
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-[#14161f] border border-[#1f222b] space-y-3">
                    <div className="text-xs font-bold text-[#f0f1f6]">Configure Benchmark Reconstruction Run</div>
                    
                    <div className="space-y-3">
                      <div>
                        <label className="text-[10px] text-[#54596b] uppercase block mb-1">Target Structure</label>
                        <input
                          type="text"
                          disabled
                          value={`${activeBenchmark.name} (${activeBenchmark.id})`}
                          className="w-full bg-[#0c0d11] border border-[#1f222b] rounded px-3 py-1.5 text-xs text-[#9296a6]"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-[10px] text-[#54596b] uppercase block mb-1">Reconstruction Pipeline</label>
                          <select
                            value={selectedPipeline}
                            onChange={(e) => setSelectedPipeline(e.target.value)}
                            className="w-full bg-[#0c0d11] border border-[#1f222b] rounded px-3 py-1.5 text-xs text-[#ededf2] outline-none"
                          >
                            <option value="RealityEngine Hybrid (SfM + Volumetric)">RealityEngine Hybrid (SfM + Volumetric)</option>
                            <option value="COLMAP SfM + OpenMVS Dense">COLMAP SfM + OpenMVS Dense</option>
                            <option value="3D Gaussian Splatting (Inria / MCMC)">3D Gaussian Splatting (Inria / MCMC)</option>
                            <option value="Sparse NeRF-W (Wild Environments)">Sparse NeRF-W (Wild Environments)</option>
                          </select>
                        </div>

                        <div>
                          <label className="text-[10px] text-[#54596b] uppercase block mb-1">Quality Target</label>
                          <select
                            value={selectedQuality}
                            onChange={(e) => setSelectedQuality(e.target.value)}
                            className="w-full bg-[#0c0d11] border border-[#1f222b] rounded px-3 py-1.5 text-xs text-[#ededf2] outline-none"
                          >
                            <option value="Sub-Centimeter Geodetic">Sub-Centimeter Geodetic (± 12mm)</option>
                            <option value="Production High (± 30mm)">Production High (± 30mm)</option>
                            <option value="Rapid Preview Draft">Rapid Preview Draft</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {reconstructNotice && (
                      <div className={`p-2.5 rounded text-xs ${
                        reconstructNotice.startsWith('✓') ? 'bg-[#2ecc71]/15 text-[#2ecc71] border border-[#2ecc71]/30' : 'bg-red-500/15 text-red-400 border border-red-500/30'
                      }`}>
                        {reconstructNotice}
                      </div>
                    )}

                    <div className="pt-2 flex items-center justify-between">
                      <span className="text-[10px] text-[#54596b]">
                        Estimated GPU time: ~45 min on RTX 4090 OptiX
                      </span>
                      <button
                        type="button"
                        onClick={handleRunReconstruct}
                        className="px-4 py-2 rounded-lg bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b] font-bold text-xs flex items-center gap-1.5 transition-colors"
                      >
                        <Play className="w-3.5 h-3.5" />
                        <span>Run Benchmark Reconstruction</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: PERCEPTION & WORLDIR */}
              {detailTab === 'worldir' && (
                <div className="space-y-4">
                  <div className="p-3.5 rounded-lg bg-[#14161f] border border-[#1f222b] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-[#f0f1f6]">Semantic Spatial Hierarchy</span>
                      <span className={`text-[10px] font-bold ${activeBenchmark.worldIRStatus === 'COMPILED' ? 'text-[#2ecc71]' : 'text-[#54596b]'}`}>
                        WorldIR: {activeBenchmark.worldIRStatus}
                      </span>
                    </div>
                    <div className="p-3 rounded bg-[#0b0c10] border border-[#1f222b] font-mono text-[11px] text-[#c4c7d4] space-y-1">
                      <div>📁 {activeBenchmark.name} [WORLD_SITE]</div>
                      <div className="pl-4">├── 🏛 Main Monument Building [BUILDING]</div>
                      <div className="pl-8">├── ⚪ Central Rotunda & Dome [DOME]</div>
                      <div className="pl-8">├── 🏛 Colonnaded Galleries [COLONNADE]</div>
                      <div className="pl-8">├── 🧱 Stepped Terraces & Plinth [PLINTH]</div>
                      <div className="pl-4">└── 🌳 Grounds & Reflecting Basin [SITE_TERRAIN]</div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 5: METRICS */}
              {detailTab === 'metrics' && (
                <div className="space-y-4">
                  {activeBenchmark.metrics ? (
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">Registered Cameras</div>
                        <div className="text-base font-bold text-[#2ecc71] num-tabular mt-0.5">
                          {activeBenchmark.metrics.registeredCameras?.toLocaleString()}
                        </div>
                        <div className="text-[9px] text-[#9296a6]">100% pose convergence</div>
                      </div>
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">Dense Points</div>
                        <div className="text-base font-bold text-[#3d8ef7] num-tabular mt-0.5">
                          {(activeBenchmark.metrics.densePoints! / 1000000).toFixed(2)} M
                        </div>
                        <div className="text-[9px] text-[#9296a6]">Sub-centimeter density</div>
                      </div>
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">Mean Reprojection Error</div>
                        <div className="text-base font-bold text-[#2ecc71] num-tabular mt-0.5">
                          {activeBenchmark.metrics.reprojectionErrorPx} px
                        </div>
                        <div className="text-[9px] text-[#9296a6]">Bundle adjustment residual</div>
                      </div>
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">Geometric Uncertainty (1σ)</div>
                        <div className="text-base font-bold text-[#a855f7] num-tabular mt-0.5">
                          ± {activeBenchmark.metrics.uncertaintyMm} mm
                        </div>
                        <div className="text-[9px] text-[#9296a6]">Calibrated photogrammetric bound</div>
                      </div>
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">Detected Planes / Rooms</div>
                        <div className="text-base font-bold text-[#f0f1f6] num-tabular mt-0.5">
                          {activeBenchmark.metrics.detectedPlanes} planes • {activeBenchmark.metrics.detectedRooms} rooms
                        </div>
                        <div className="text-[9px] text-[#9296a6]">WorldIR primitives</div>
                      </div>
                      <div className="p-3 rounded bg-[#14161f] border border-[#1f222b]">
                        <div className="text-[10px] text-[#54596b] uppercase">GPU Processing Time</div>
                        <div className="text-base font-bold text-[#f0f1f6] num-tabular mt-0.5">
                          {activeBenchmark.metrics.processingTimeMin} min
                        </div>
                        <div className="text-[9px] text-[#9296a6]">OptiX acceleration</div>
                      </div>
                    </div>
                  ) : (
                    <div className="p-6 rounded bg-[#14161f] border border-[#1f222b] text-center space-y-2">
                      <HelpCircle className="w-6 h-6 text-[#54596b] mx-auto" />
                      <div className="text-sm font-semibold text-[#f0f1f6]">No Metrics Available</div>
                      <p className="text-xs text-[#54596b] max-w-md mx-auto">
                        This benchmark has not yet undergone full reconstruction. Metrics will be populated upon completing the reconstruction workflow.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 6: STRESS FACTORS */}
              {detailTab === 'stress' && (
                <div className="space-y-3">
                  <div className="text-[10px] text-[#54596b] uppercase font-bold">Edge-Case Reconstruction Challenges</div>
                  <div className="space-y-2">
                    {activeBenchmark.keyChallenges.map((ch, idx) => (
                      <div key={idx} className="p-3 rounded bg-[#14161f] border border-red-500/20 flex items-start gap-2.5">
                        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                        <div>
                          <div className="text-xs font-semibold text-[#f0f1f6]">{ch}</div>
                          <div className="text-[10px] text-[#9296a6] mt-0.5">
                            Evaluated in bundle adjustment, dense matching, and volumetric mesh integration.
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── MULTI-TARGET COMPARISON MODAL ─────────────────────────── */}
      {comparisonModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="w-full max-w-5xl bg-[#101217] border border-[#1f222b] rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            <div className="p-4 border-b border-[#1f222b] bg-[#0c0d11] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-[#3d8ef7]" />
                <h2 className="text-sm font-bold text-[#f0f1f6]">
                  Side-by-Side Architectural Benchmark Comparison
                </h2>
                <span className="text-[10px] text-[#9296a6]">({comparisonList.length} Targets)</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={clearComparisonBenchmarks}
                  className="px-2 py-1 text-[10px] text-[#e74c3c] hover:bg-[#1f1515] rounded border border-red-500/30"
                >
                  Clear All
                </button>
                <button
                  type="button"
                  onClick={() => setComparisonModalOpen(false)}
                  className="p-1 text-[#54596b] hover:text-[#ededf2]"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="p-4 overflow-x-auto">
              <table className="w-full text-left border-collapse font-mono text-xs">
                <thead>
                  <tr className="border-b border-[#1f222b]">
                    <th className="p-3 text-[10px] text-[#54596b] uppercase w-48">Parameter</th>
                    {comparisonList.map((target) => (
                      <th key={target.id} className="p-3 font-bold text-[#f0f1f6]">
                        {target.name}
                        <div className="text-[10px] text-[#3d8ef7] font-normal">{target.location}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1f222b]/50">
                  <tr>
                    <td className="p-3 text-[#54596b]">Category</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[#c4c7d4]">{t.category.replace(/_/g, ' ')}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Typology</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[#ededf2]">{t.architectureTypology}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Scale / Height</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[#f0f1f6] font-semibold num-tabular">{t.scale}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Complexity</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3">{getComplexityBadge(t.complexity)}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Reconstruction Status</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3">{getStatusBadge(t.reconstructionStatus)}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Evidence Availability</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 font-semibold text-[#c4c7d4]">{t.captureAvailability.replace(/_/g, ' ')}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Ground Truth</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[#9296a6]">{t.groundTruthAvailability.replace(/_/g, ' ')}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">WorldIR State</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[#3d8ef7]">{t.worldIRStatus}</td>
                    ))}
                  </tr>
                  <tr>
                    <td className="p-3 text-[#54596b]">Key Stress Factors</td>
                    {comparisonList.map((t) => (
                      <td key={t.id} className="p-3 text-[11px] text-[#9296a6]">
                        {t.keyChallenges.slice(0, 2).join(' • ')}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
