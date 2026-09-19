'use client';

/**
 * HomeScreen — session selection/creation plus the engine's capture requests.
 *
 * First stop in the field: pick up an open session (or a session the engine has
 * asked for more evidence in), import a task file / session bundle, or start a
 * new capture. Large touch targets, minimal typing.
 */
import { useRef, useState } from 'react';
import { Plus, ChevronRight, Camera, FolderCheck, Search, ClipboardList, Upload, Images, AlertTriangle } from 'lucide-react';
import { openTasksFor, summarizeSession, useMobileStore } from './store';
import type { CaptureTask, FieldSession } from './types';

const STATUS_STYLE: Record<FieldSession['status'], string> = {
  CAPTURING: 'text-[#3d8ef7]',
  READY_TO_SYNC: 'text-[#f1c40f]',
  SYNCED: 'text-[#2ecc71]',
  ARCHIVED: 'text-[#9296a6]',
};

export function HomeScreen({
  onOpenCapture,
  onOpenReview,
}: {
  onOpenCapture: () => void;
  onOpenReview: () => void;
}) {
  const sessions = useMobileStore((s) => s.sessions);
  const activeId = useMobileStore((s) => s.activeSessionId);
  const selectSession = useMobileStore((s) => s.selectSession);
  const setActiveTask = useMobileStore((s) => s.setActiveTask);
  const tasks = useMobileStore((s) => s.tasks);
  const importTaskFile = useMobileStore((s) => s.importTaskFile);
  const importBundle = useMobileStore((s) => s.importBundle);
  const [name, setName] = useState('');
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState('');
  const [notice, setNotice] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null);
  const taskFileRef = useRef<HTMLInputElement>(null);
  const bundleFileRef = useRef<HTMLInputElement>(null);

  const filtered = sessions.filter((s) => s.name.toLowerCase().includes(query.toLowerCase()));
  const requested = openTasksFor(tasks, activeId);

  const startCapture = () => {
    const trimmed = name.trim() || `Site capture ${new Date().toLocaleDateString()}`;
    useMobileStore.getState().createSession(trimmed, 'field');
    setName('');
    setCreating(false);
    onOpenCapture();
  };

  /**
   * Servicing a request for a session this device does not hold yet creates the
   * real session locally under the engine's own session id, so the frames the
   * operator captures land where the request came from.
   */
  const shootForTask = (task: CaptureTask) => {
    const existing = sessions.find((s) => s.sessionId === task.sessionId);
    if (existing) {
      selectSession(task.sessionId);
    } else {
      const now = new Date().toISOString();
      useMobileStore.setState((s) => ({
        sessions: [
          {
            sessionId: task.sessionId,
            name: `Requested session ${task.sessionId.slice(0, 8)}`,
            intent: 'requested capture',
            createdAt: now,
            updatedAt: now,
            status: 'CAPTURING' as const,
            frames: [],
          },
          ...s.sessions,
        ],
        activeSessionId: task.sessionId,
        activeTaskId: null,
      }));
    }
    setActiveTask(task.taskId);
    onOpenCapture();
  };

  const readFile = (file: File, kind: 'tasks' | 'bundle') => {
    void file
      .text()
      .then(async (text) => {
        if (kind === 'tasks') {
          const count = importTaskFile(text);
          setNotice(
            count > 0
              ? { tone: 'ok', text: `${count} capture request${count === 1 ? '' : 's'} imported` }
              : { tone: 'error', text: 'Task file contained no usable tasks' },
          );
          return;
        }
        const result = await importBundle(text);
        const parts = [`Bundle imported: ${result.framesStored} frame(s) verified into device storage`];
        if (result.integrityFailures > 0) parts.push(`${result.integrityFailures} failed sha256 verification and were NOT accepted`);
        if (result.skippedFrames > 0) parts.push(`${result.skippedFrames} frame(s) carried no pixels in the file`);
        setNotice({ tone: result.integrityFailures > 0 ? 'error' : 'ok', text: parts.join(' · ') });
      })
      .catch((err: unknown) => setNotice({ tone: 'error', text: (err as Error).message }));
  };

  const missingFor = (s: FieldSession) => s.frames.filter((f) => f.localCopy === 'missing').length;

  return (
    <div className="flex h-full flex-col bg-[#0d0e12] text-[#f0f1f6]">
      <div className="flex items-start justify-between border-b border-[#1f222b] px-5 pb-3 pt-5">
        <div>
          <div className="text-lg font-bold tracking-tight">Reality Capture</div>
          <div className="font-mono text-[10px] text-[#9296a6]">FIELD MODE · evidence-grade capture</div>
        </div>
        <div className="mt-1 flex gap-2">
          <button
            type="button"
            onClick={() => taskFileRef.current?.click()}
            className="flex items-center gap-1 rounded-lg border border-[#2b3040] bg-[#171922] px-2 py-1.5 text-[10px] font-semibold text-[#9296a6]"
          >
            <ClipboardList className="h-3.5 w-3.5" /> Requests
          </button>
          <button
            type="button"
            onClick={() => bundleFileRef.current?.click()}
            className="flex items-center gap-1 rounded-lg border border-[#2b3040] bg-[#171922] px-2 py-1.5 text-[10px] font-semibold text-[#9296a6]"
          >
            <Upload className="h-3.5 w-3.5" /> Bundle
          </button>
        </div>
      </div>

      <input
        ref={taskFileRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) readFile(f, 'tasks');
          e.target.value = '';
        }}
      />
      <input
        ref={bundleFileRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) readFile(f, 'bundle');
          e.target.value = '';
        }}
      />
<div className="min-h-0 flex-1 overflow-y-auto p-4">
        {notice && (
          <div
            className={`mb-3 flex items-start gap-2 rounded-xl border p-3 text-[11px] ${
              notice.tone === 'ok'
                ? 'border-[#2ecc71]/40 bg-[#2ecc71]/10 text-[#c0f0d4]'
                : 'border-[#e74c3c]/40 bg-[#e74c3c]/10 text-[#f0c4c0]'
            }`}
          >
            {notice.tone === 'error' && <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
            <span>{notice.text}</span>
          </div>
        )}

        {requested.length > 0 && (
          <div className="mb-4">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-[#f1c40f]">
              requested captures · {requested.length}
            </div>
            <ul className="space-y-2">
              {requested.map((task) => (
                <li key={task.taskId} className="rounded-2xl border border-[#f1c40f]/40 bg-[#f1c40f]/10 p-3">
                  <p className="text-xs font-medium leading-snug text-[#f0f1f6]">{task.guidance}</p>
                  <p className="mt-1 font-mono text-[9px] leading-relaxed text-[#9296a6]">
                    {task.taskId}
                    {task.targetFrameIds.length > 0 && ` · because ${task.targetFrameIds.join(', ')}`}
                    {task.reasons.length > 0 && ` · ${task.reasons.join(', ')}`}
                  </p>
                  <button
                    type="button"
                    onClick={() => shootForTask(task)}
                    className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-[#f1c40f] py-2 text-xs font-bold text-black"
                  >
                    <Camera className="h-4 w-4" /> Shoot this
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mb-3 flex items-center gap-2 rounded-xl border border-[#2b3040] bg-[#171922] px-3 py-2">
          <Search className="h-4 w-4 shrink-0 text-[#9296a6]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter sessions"
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[#4a4f60]"
          />
        </div>
{filtered.length === 0 ? (
          <div className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-[#2b3040] p-6 text-center">
            <FolderCheck className="h-6 w-6 text-[#4a4f60]" />
            <p className="text-sm text-[#9296a6]">
              {sessions.length === 0 ? 'No sessions on this device yet.' : 'No session matches that filter.'}
            </p>
            <p className="text-[11px] text-[#9296a6]">Start a capture to create one, or import a requested-capture file.</p>
          </div>
        ) : (
          <ul className="space-y-2">
            {filtered.map((session) => {
              const summary = summarizeSession(session);
              const sessionTasks = openTasksFor(tasks, session.sessionId).length;
              const missing = missingFor(session);
              const active = session.sessionId === activeId;
              return (
                <li
                  key={session.sessionId}
                  className={`overflow-hidden rounded-2xl border ${
                    active ? 'border-[#3d8ef7]/60 bg-[#3d8ef7]/5' : 'border-[#2b3040] bg-[#171922]'
                  }`}
                >
                  <div className="flex items-stretch">
                    <button
                      type="button"
                      onClick={() => {
                        selectSession(session.sessionId);
                        onOpenCapture();
                      }}
                      className="min-w-0 flex-1 px-3 py-3 text-left"
                    >
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-semibold">{session.name}</span>
                        <span className={`font-mono text-[9px] ${STATUS_STYLE[session.status]}`}>{session.status}</span>
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-[#9296a6]">
                        {summary.total} frames · {summary.useful} useful · {summary.rejected} rejected · {summary.redundant}{' '}
                        redundant
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[9px]">
                        <span className="text-[#4a4f60]">{session.sessionId.slice(0, 8)}</span>
                        <span className="text-[#9296a6]">
                          {summary.headingSpreadDeg == null
                            ? 'heading spread unknown'
                            : `${Math.round(summary.headingSpreadDeg)}° heading spread`}
                        </span>
                        {sessionTasks > 0 && <span className="text-[#f1c40f]">{sessionTasks} requested</span>}
                        {missing > 0 && <span className="text-[#e74c3c]">{missing} no local copy</span>}
                        {summary.total > 0 && summary.missingLocalCopy === 0 && (
                          <span className="text-[#2ecc71]">all pixels on device</span>
                        )}
                      </div>
                    </button>
                    <button
                      type="button"
                      aria-label={`Evidence for ${session.name}`}
                      onClick={() => {
                        selectSession(session.sessionId);
                        onOpenReview();
                      }}
                      className="flex w-12 shrink-0 flex-col items-center justify-center gap-1 border-l border-[#2b3040] text-[#9296a6]"
                    >
                      <Images className="h-4 w-4" />
                      <span className="font-mono text-[8px]">{summary.total}</span>
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <div className="border-t border-[#1f222b] bg-[#0d0e12] p-4">
        {creating ? (
          <div className="space-y-2">
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Session name (optional)"
              className="w-full rounded-xl border border-[#2b3040] bg-[#171922] px-4 py-3 text-sm outline-none placeholder:text-[#4a4f60]"
            />
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setCreating(false)}
                className="rounded-xl border border-[#2b3040] py-3 text-sm font-semibold text-[#9296a6]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={startCapture}
                className="flex items-center justify-center gap-2 rounded-xl bg-[#3d8ef7] py-3 text-sm font-bold text-white"
              >
                <Camera className="h-4 w-4" /> Start capture
              </button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setCreating(true)}
            className="flex w-full items-center justify-center gap-2 rounded-2xl bg-[#3d8ef7] py-4 text-sm font-bold text-white"
          >
            <Plus className="h-5 w-5" /> New capture session
            <ChevronRight className="h-4 w-4 opacity-70" />
          </button>
        )}
      </div>
    </div>
  );
}