'use client';

/**
 * MobileFieldApp — the Reality Engine field product.
 *
 * A distinct mobile application (not a responsive desktop Studio):
 * OPEN → SELECT/CREATE SESSION → CAPTURE → LIVE ASSESSMENT → REVIEW →
 * FINISH → EXPORT/HAND-OFF. Bottom tab bar, one-hand reach, glanceable state.
 */
import { useEffect, useState } from 'react';
import { Home, Camera, Images, Send, AlertTriangle } from 'lucide-react';
import { HomeScreen } from './HomeScreen';
import { CaptureScreen } from './CaptureScreen';
import { ReviewScreen } from './ReviewScreen';
import { SyncScreen } from './SyncScreen';
import { openTasksFor, useMobileStore } from './store';

type Tab = 'home' | 'capture' | 'review' | 'sync';

const TABS: { id: Tab; label: string; icon: typeof Home }[] = [
  { id: 'home', label: 'Sessions', icon: Home },
  { id: 'capture', label: 'Capture', icon: Camera },
  { id: 'review', label: 'Evidence', icon: Images },
  { id: 'sync', label: 'Hand off', icon: Send },
];

export default function MobileFieldApp() {
  const [tab, setTab] = useState<Tab>('home');
  const rehydrateFromVault = useMobileStore((s) => s.rehydrateFromVault);
  const rehydrated = useMobileStore((s) => s.rehydrated);
  const activeSession = useMobileStore((s) => s.sessions.find((x) => x.sessionId === s.activeSessionId) ?? null);
  const activeId = useMobileStore((s) => s.activeSessionId);
  const openTaskCount = useMobileStore((s) => openTasksFor(s.tasks, s.activeSessionId).length);
  const missingCopies = useMobileStore(
    (s) =>
      s.sessions
        .find((x) => x.sessionId === s.activeSessionId)
        ?.frames.filter((f) => f.localCopy === 'missing').length ?? 0,
  );

  // Resolve persisted frames against the vault before the operator trusts them.
  useEffect(() => {
    void rehydrateFromVault();
  }, [rehydrateFromVault]);

  return (
    <div className="relative mx-auto flex h-dvh w-full max-w-md flex-col overflow-hidden bg-[#0d0e12] text-[#f0f1f6] antialiased">
      <main className="min-h-0 flex-1">
        {tab === 'home' && <HomeScreen onOpenCapture={() => setTab('capture')} onOpenReview={() => setTab('review')} />}
        {tab === 'capture' && <CaptureScreen onExit={() => setTab('home')} />}
        {tab === 'review' && <ReviewScreen onExit={() => setTab('home')} />}
        {tab === 'sync' && <SyncScreen onExit={() => setTab('home')} />}
      </main>

      {/* Bottom tab bar — thumb zone. Capture is always reachable. */}
      <nav className="grid grid-cols-4 border-t border-[#1f222b] bg-[#0d0e12]" aria-label="Field navigation">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id;
          const disabled = id !== 'home' && !activeId;
          const badge = id === 'capture' ? openTaskCount : id === 'sync' ? missingCopies : 0;
          return (
            <button
              key={id}
              type="button"
              disabled={disabled}
              onClick={() => setTab(id)}
              className={`relative flex flex-col items-center gap-0.5 py-3 text-[10px] font-medium transition-colors ${
                active ? 'text-[#3d8ef7]' : disabled ? 'text-[#4a4f60]' : 'text-[#9296a6]'
              }`}
            >
              <Icon className="h-5 w-5" />
              {label}
              {badge > 0 && (
                <span
                  className={`absolute right-1/2 top-1.5 translate-x-4 rounded-full px-1 font-mono text-[9px] font-bold ${
                    id === 'sync' ? 'bg-[#e74c3c] text-black' : 'bg-[#f1c40f] text-black'
                  }`}
                >
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {!rehydrated && (
        <div className="pointer-events-none absolute inset-x-0 top-0 z-30 bg-[#171922]/95 px-4 py-1.5 text-center font-mono text-[10px] text-[#9296a6]">
          checking device storage…
        </div>
      )}

      {/* Session context strip when capturing without a session selected */}
      {activeSession === null && tab !== 'home' && (
        <div className="absolute inset-x-0 top-0 z-30 bg-[#f1c40f] px-4 py-2 text-center text-xs font-semibold text-black">
          No active session — pick one in Sessions first.
        </div>
      )}

      {/* Durability warning: pixels that cannot be handed off must be visible. */}
      {activeSession !== null && missingCopies > 0 && tab !== 'sync' && (
        <div className="absolute inset-x-0 top-0 z-30 flex items-center justify-center gap-2 bg-[#e74c3c] px-4 py-1.5 text-center text-[11px] font-semibold text-black">
          <AlertTriangle className="h-3.5 w-3.5" />
          {missingCopies} frame{missingCopies === 1 ? '' : 's'} have no local copy — those pixels cannot be handed off
        </div>
      )}
    </div>
  );
}
