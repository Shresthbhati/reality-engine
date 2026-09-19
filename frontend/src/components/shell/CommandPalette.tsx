'use client';

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from 'react';
import { useREStore } from '@/store/re-store';
import type { WorkspaceId } from '@/types/reality-engine';

// ─── Command Definitions ──────────────────────────────────────────────────────

type CommandCategory =
  | 'Mode'
  | 'Scale'
  | 'World'
  | 'Session'
  | 'Build'
  | 'View'
  | 'Export'
  | 'Workspace'
  | 'Simulate';

interface Command {
  id: string;
  label: string;
  category: CommandCategory;
  keywords?: string[];
  shortcut?: string;
  action: () => void;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function CommandPalette() {
  const commandPaletteOpen = useREStore((s) => s.commandPaletteOpen);
  const setCommandPaletteOpen = useREStore((s) => s.setCommandPaletteOpen);
  const setActiveWorkspace = useREStore((s) => s.setActiveWorkspace);
  const globalMode = useREStore((s) => s.globalMode);
  const setGlobalMode = useREStore((s) => s.setGlobalMode);
  const spatialScale = useREStore((s) => s.spatialScale);
  const setSpatialScale = useREStore((s) => s.setSpatialScale);
  const toggleViewportOption = useREStore((s) => s.toggleViewportOption);
  const setActiveMeasurementTool = useREStore((s) => s.setActiveMeasurementTool);
  const toggleOutliner = useREStore((s) => s.toggleOutliner);
  const toggleLeftNav = useREStore((s) => s.toggleLeftNav);
  const toggleInspector = useREStore((s) => s.toggleInspector);
  const toggleBottomDrawer = useREStore((s) => s.toggleBottomDrawer);
  const setShadingMode = useREStore((s) => s.setShadingMode);
  const showAllEntities = useREStore((s) => s.showAllEntities);

  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const itemRefs = useRef<(HTMLLIElement | null)[]>([]);

  const makeWorkspaceSwitcher = useCallback(
    (id: WorkspaceId, label: string): Command => ({
      id: `workspace-${id}`,
      label,
      category: 'Workspace',
      action: () => {
        setActiveWorkspace(id);
        setCommandPaletteOpen(false);
      },
    }),
    [setActiveWorkspace, setCommandPaletteOpen],
  );

  const commands: Command[] = useMemo(
    () => [
      // ── Global Modes
      {
        id: 'mode-explore',
        label: 'Mode: EXPLORE (Free Navigation & View)',
        category: 'Mode',
        keywords: ['explore', 'fly', 'orbit', 'walk'],
        action: () => {
          setGlobalMode('EXPLORE');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'mode-inspect',
        label: 'Mode: INSPECT (Entity & Uncertainty Analysis)',
        category: 'Mode',
        keywords: ['inspect', 'metrics', 'provenance', 'error'],
        action: () => {
          setGlobalMode('INSPECT');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'mode-edit',
        label: 'Mode: EDIT (Spatial Geometry & Annotations)',
        category: 'Mode',
        keywords: ['edit', 'modify', 'transform', 'tag'],
        action: () => {
          setGlobalMode('EDIT');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'mode-capture',
        label: 'Mode: CAPTURE (Multi-Sensor Instrument & Guidance)',
        category: 'Mode',
        keywords: ['capture', 'sensor', 'camera', 'lidar', 'mobile'],
        action: () => {
          setGlobalMode('CAPTURE');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'mode-build',
        label: 'Mode: BUILD (Reconstruction Pipeline & DAG)',
        category: 'Mode',
        keywords: ['build', 'reconstruct', 'pipeline', 'dag'],
        action: () => {
          setGlobalMode('BUILD');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'mode-review',
        label: 'Mode: REVIEW (Evidence Lineage & Validation)',
        category: 'Mode',
        keywords: ['review', 'evidence', 'lineage', 'qa'],
        action: () => {
          setGlobalMode('REVIEW');
          setCommandPaletteOpen(false);
        },
      },

      // ── Global Spatial Scales
      {
        id: 'scale-room',
        label: 'Scale: ROOM (0 – 10 m · Sub-Centimeter Detail)',
        category: 'Scale',
        keywords: ['scale', 'room', 'interior', 'lod4'],
        action: () => {
          setSpatialScale('ROOM');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'scale-building',
        label: 'Scale: BUILDING (10 – 50 m · Facades & Massing)',
        category: 'Scale',
        keywords: ['scale', 'building', 'structure', 'lod3'],
        action: () => {
          setSpatialScale('BUILDING');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'scale-street',
        label: 'Scale: STREET (50 – 200 m · Right-of-Way & Utilities)',
        category: 'Scale',
        keywords: ['scale', 'street', 'road', 'corridor'],
        action: () => {
          setSpatialScale('STREET');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'scale-block',
        label: 'Scale: BLOCK (200 – 1,000 m · Urban Parcels)',
        category: 'Scale',
        keywords: ['scale', 'block', 'parcels', 'lod2'],
        action: () => {
          setSpatialScale('BLOCK');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'scale-district',
        label: 'Scale: DISTRICT (15 – 40 km · Urban Sectors)',
        category: 'Scale',
        keywords: ['scale', 'district', 'borough', 'lod1'],
        action: () => {
          setSpatialScale('DISTRICT');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'scale-city',
        label: 'Scale: CITY (40+ km · Metropolitan Area)',
        category: 'Scale',
        keywords: ['scale', 'city', 'metro', 'lod0'],
        action: () => {
          setSpatialScale('CITY');
          setCommandPaletteOpen(false);
        },
      },

      // Navigation & Panels
      {
        id: 'toggle-left-nav',
        label: 'Toggle World Navigation Panel (⌘B)',
        category: 'Workspace',
        shortcut: '⌘B',
        action: () => {
          toggleLeftNav();
          setCommandPaletteOpen(false);
        },
      },

      // Reconstruction & Build
      {
        id: 'start-reconstruction',
        label: 'Reconstruct Scene (Launch Build DAG)',
        category: 'Build',
        keywords: ['sfm', 'colmap', 'openmvs', 'pipeline'],
        action: () => {
          setActiveWorkspace('build');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'evidence-investigate',
        label: 'Trace Entity Evidence Lineage (Provenance)',
        category: 'World',
        keywords: ['provenance', 'observations', 'why', 'confidence'],
        action: () => {
          setActiveWorkspace('evidence');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'ingest-sources',
        label: 'Ingest Evidence / Add Sensor Sources',
        category: 'Session',
        keywords: ['import', 'drone', 'lidar', 'gnss', 'dropzone'],
        action: () => {
          setActiveWorkspace('loader');
          setCommandPaletteOpen(false);
        },
      },
      // 3D Viewport Controls
      {
        id: 'toggle-cameras',
        label: 'Toggle Camera Frustums',
        category: 'View',
        keywords: ['frustum', 'poses', 'trajectories'],
        action: () => {
          setActiveWorkspace('studio');
          toggleViewportOption('showCameras');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'toggle-grid',
        label: 'Toggle Ground Plane Grid',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          toggleViewportOption('showGrid');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'toggle-pointcloud',
        label: 'Toggle Dense Point Cloud Layer',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          toggleViewportOption('showPointCloud');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'shading-confidence',
        label: 'Shading: Confidence Heatmap',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          setShadingMode('CONFIDENCE');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'shading-rgb',
        label: 'Shading: True Color (RGB)',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          setShadingMode('RGB');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'show-all-entities',
        label: 'Show All Entities (Unhide All)',
        category: 'View',
        action: () => {
          showAllEntities();
          setCommandPaletteOpen(false);
        },
      },
      // Measurement Tools
      {
        id: 'measure-distance',
        label: 'Measure Distance (Point-to-Point)',
        category: 'View',
        keywords: ['ruler', 'dimension', 'length'],
        action: () => {
          setActiveWorkspace('studio');
          setActiveMeasurementTool('POINT_TO_POINT');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'measure-height',
        label: 'Measure Vertical Height',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          setActiveMeasurementTool('HEIGHT');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'measure-wall',
        label: 'Measure Wall Thickness',
        category: 'View',
        action: () => {
          setActiveWorkspace('studio');
          setActiveMeasurementTool('WALL_THICKNESS');
          setCommandPaletteOpen(false);
        },
      },
      // Studio Panels
      {
        id: 'toggle-outliner',
        label: 'Toggle World Outliner Panel (⌘B)',
        category: 'Workspace',
        action: () => {
          toggleOutliner();
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'toggle-inspector',
        label: 'Toggle Entity Inspector Panel (⌘I)',
        category: 'Workspace',
        action: () => {
          toggleInspector();
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'toggle-bottom-drawer',
        label: 'Toggle Bottom Docking Drawer (⌘J)',
        category: 'Workspace',
        action: () => {
          toggleBottomDrawer();
          setCommandPaletteOpen(false);
        },
      },
      // Export & Settings
      {
        id: 'export-usd',
        label: 'Export OpenUSD Scene Package (.usdc / .usdz)',
        category: 'Export',
        keywords: ['usd', 'pixar', 'omniverse'],
        action: () => {
          setActiveWorkspace('settings');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'export-ifc',
        label: 'Export Building Information Model (IFC 4x3)',
        category: 'Export',
        keywords: ['bim', 'ifc', 'revit', 'archicad'],
        action: () => {
          setActiveWorkspace('settings');
          setCommandPaletteOpen(false);
        },
      },
      {
        id: 'export-gltf',
        label: 'Export glTF 2.0 Mesh & Splats (.glb)',
        category: 'Export',
        keywords: ['gltf', 'glb', 'threejs'],
        action: () => {
          setActiveWorkspace('settings');
          setCommandPaletteOpen(false);
        },
      },
      // Workspace Switchers
      makeWorkspaceSwitcher('capture',     'Switch to Capture Instrument (Mobile Field)'),
      makeWorkspaceSwitcher('loader',      'Switch to Loader Workstation (Ingest & Sync)'),
      makeWorkspaceSwitcher('build',       'Switch to Build Pipeline (10-Stage DAG)'),
      makeWorkspaceSwitcher('studio',      'Switch to Reconstruction Studio (3D Viewport)'),
      makeWorkspaceSwitcher('evidence',    'Switch to Evidence Investigation Console'),
      makeWorkspaceSwitcher('benchmarks',  'Switch to Benchmark Explorer (40 Targets)'),
      makeWorkspaceSwitcher('city',        'Switch to City Builder (Downstream WorldIR)'),
      makeWorkspaceSwitcher('simulation',  'Switch to Simulation Control Room'),
      makeWorkspaceSwitcher('diagnostics', 'Switch to Engine Hardware Diagnostics'),
      makeWorkspaceSwitcher('settings',    'Switch to Workstation Settings & Standards'),
    ],
    [
      makeWorkspaceSwitcher,
      setCommandPaletteOpen,
      setActiveWorkspace,
      toggleViewportOption,
      setActiveMeasurementTool,
      toggleOutliner,
      toggleInspector,
      toggleBottomDrawer,
      setShadingMode,
      showAllEntities,
    ],
  );

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) return commands;
    return commands.filter(
      (c) =>
        c.label.toLowerCase().includes(q) ||
        c.category.toLowerCase().includes(q) ||
        c.keywords?.some((k) => k.includes(q)),
    );
  }, [commands, query]);

  const [prevQuery, setPrevQuery] = useState(query);
  if (query !== prevQuery) {
    setPrevQuery(query);
    setActiveIndex(0);
  }

  const [prevOpen, setPrevOpen] = useState(commandPaletteOpen);
  if (commandPaletteOpen !== prevOpen) {
    setPrevOpen(commandPaletteOpen);
    if (commandPaletteOpen) {
      setQuery('');
      setActiveIndex(0);
    }
  }

  // Auto-scroll active item into view
  useEffect(() => {
    itemRefs.current[activeIndex]?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  // Focus input when opened
  useEffect(() => {
    if (commandPaletteOpen) {
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [commandPaletteOpen]);

  // Global Cmd/Ctrl+K to open
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(!commandPaletteOpen);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [commandPaletteOpen, setCommandPaletteOpen]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        filtered[activeIndex]?.action();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setCommandPaletteOpen(false);
      }
    },
    [activeIndex, filtered, setCommandPaletteOpen],
  );

  if (!commandPaletteOpen) return null;

  const CATEGORY_COLORS: Record<CommandCategory, string> = {
    Mode:      '#00e5ff',
    Scale:     '#38bdf8',
    World:     '#3d8ef7',
    Session:   '#9898b0',
    Build:     '#f0a050',
    View:      '#7c6ef7',
    Export:    '#27c370',
    Workspace: '#5c5c78',
    Simulate:  '#e05050',
  };

  return (
    /* Overlay */
    <div
      className="fixed inset-0 z-50 flex items-start justify-center"
      style={{
        background: 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(4px)',
        paddingTop: '15vh',
      }}
      onClick={() => setCommandPaletteOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      {/* Dialog */}
      <div
        className="flex flex-col overflow-hidden"
        style={{
          width: '560px',
          maxHeight: '420px',
          background: 'var(--re-bg-elevated)',
          border: '1px solid var(--re-border-default)',
          borderRadius: '8px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div
          className="flex items-center gap-2.5 px-3.5"
          style={{
            borderBottom: '1px solid var(--re-border-default)',
            height: '44px',
            flexShrink: 0,
          }}
        >
          {/* Search icon */}
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            aria-hidden="true"
            style={{ color: 'var(--re-text-tertiary)', flexShrink: 0 }}
          >
            <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3" />
            <path d="M9.5 9.5L12.5 12.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>

          <input
            ref={inputRef}
            type="text"
            className="flex-1 bg-transparent outline-none"
            placeholder="Search commands…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              fontSize: '13px',
              color: 'var(--re-text-primary)',
              fontFamily: 'Inter, sans-serif',
            }}
            autoComplete="off"
            spellCheck={false}
          />

          <kbd
            className="flex items-center justify-center rounded px-1.5"
            style={{
              fontSize: '10px',
              color: 'var(--re-text-tertiary)',
              border: '1px solid var(--re-border-default)',
              background: 'var(--re-bg-surface)',
              fontFamily: 'inherit',
              minWidth: '22px',
              height: '18px',
            }}
          >
            esc
          </kbd>
        </div>

        {/* Results list */}
        <ul
          ref={listRef}
          role="listbox"
          className="overflow-y-auto"
          style={{
            flex: 1,
            margin: 0,
            padding: '4px 0',
            listStyle: 'none',
          }}
        >
          {filtered.length === 0 && (
            <li
              className="flex items-center justify-center"
              style={{
                height: '60px',
                color: 'var(--re-text-tertiary)',
                fontSize: '12px',
              }}
            >
              No commands found
            </li>
          )}
          {filtered.map((cmd, i) => {
            const isActive = i === activeIndex;
            return (
              <li
                key={cmd.id}
                ref={(el) => { itemRefs.current[i] = el; }}
                role="option"
                aria-selected={isActive}
                onClick={() => { setActiveIndex(i); cmd.action(); }}
                onMouseEnter={() => setActiveIndex(i)}
                className="flex items-center justify-between px-3.5 cursor-pointer"
                style={{
                  height: '34px',
                  background: isActive ? 'var(--re-bg-surface)' : 'transparent',
                  borderLeft: isActive
                    ? `2px solid var(--re-accent)`
                    : '2px solid transparent',
                }}
              >
                {/* Label + category */}
                <span className="flex items-center gap-2.5 min-w-0">
                  <span
                    style={{
                      fontSize: '11px',
                      color: CATEGORY_COLORS[cmd.category],
                      fontWeight: 500,
                      letterSpacing: '0.04em',
                      minWidth: '62px',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {cmd.category}
                  </span>
                  <span
                    className="truncate"
                    style={{
                      fontSize: '13px',
                      color: isActive
                        ? 'var(--re-text-primary)'
                        : 'var(--re-text-secondary)',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {cmd.label}
                  </span>
                </span>

                {/* Shortcut */}
                {cmd.shortcut != null && (
                  <kbd
                    className="shrink-0"
                    style={{
                      fontSize: '10px',
                      color: 'var(--re-text-tertiary)',
                      fontFamily: 'inherit',
                    }}
                  >
                    {cmd.shortcut}
                  </kbd>
                )}
              </li>
            );
          })}
        </ul>

        {/* Footer */}
        <div
          className="flex items-center gap-4 px-3.5"
          style={{
            height: '28px',
            borderTop: '1px solid var(--re-border-default)',
            flexShrink: 0,
          }}
        >
          {[
            ['↑↓', 'navigate'],
            ['↵', 'execute'],
            ['esc', 'close'],
          ].map(([key, label]) => (
            <span
              key={key}
              className="flex items-center gap-1"
              style={{ fontSize: '10px', color: 'var(--re-text-tertiary)' }}
            >
              <kbd
                className="rounded px-1"
                style={{
                  fontSize: '10px',
                  border: '1px solid var(--re-border-default)',
                  background: 'var(--re-bg-surface)',
                  fontFamily: 'inherit',
                }}
              >
                {key}
              </kbd>
              <span>{label}</span>
            </span>
          ))}
          <span
            className="ml-auto"
            style={{ fontSize: '10px', color: 'var(--re-text-tertiary)' }}
          >
            {filtered.length} result{filtered.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>
    </div>
  );
}
