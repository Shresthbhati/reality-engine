export default function PageHeader({
  title,
  action,
}: {
  title: string;
  action?: React.ReactNode;
}) {
  return (
    <div
      className="flex items-center justify-between px-6 h-14 border-b shrink-0"
      style={{ borderColor: "var(--border)" }}
    >
      <h1 className="text-sm font-semibold tracking-wide" style={{ color: "var(--text-primary)" }}>
        {title}
      </h1>
      {action}
    </div>
  );
}
