import Link from "next/link";
import PageHeader from "@/components/ui/PageHeader";
import ProposalsPanel from "@/components/proposals/ProposalsPanel";

/** Change proposals: review a candidate world against the current version, then accept, override or reject. */
export default async function WorldProposalsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Change proposals"
        action={
          <Link href={`/worlds/${encodeURIComponent(id)}`} className="text-xs hover:underline" style={{ color: "var(--accent)" }}>
            Back to workspace
          </Link>
        }
      />
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
        <ProposalsPanel worldId={id} />
      </div>
    </div>
  );
}
