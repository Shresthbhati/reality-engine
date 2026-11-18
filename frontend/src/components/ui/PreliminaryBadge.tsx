/** Marks a specialist tool's output as a screening aid, never a certification (RE-UX-000006). */
export const PRELIMINARY_NOTE =
  "Preliminary screening aid built from reconstructed geometry. Not a certification, code-compliance approval or stamped calculation; have a qualified professional verify before relying on it.";

export default function PreliminaryBadge() {
  return (
    <span
      title={PRELIMINARY_NOTE}
      aria-label={`Preliminary. ${PRELIMINARY_NOTE}`}
      className="ml-2 inline-flex items-center h-5 px-2 rounded text-[10px] font-semibold uppercase tracking-wider align-middle"
      style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}
    >
      Preliminary
    </span>
  );
}
