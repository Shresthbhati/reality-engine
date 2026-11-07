import Link from "next/link";
import PageHeader from "@/components/ui/PageHeader";
import AdoptionReviewPanel from "@/components/adoptions/AdoptionReviewPanel";

/**
 * World Adoptions: review candidate capture sessions, align relative geometry,
 * inspect 5-state semantic change sets, and commit atomic version updates (Task U.7).
 */
export default async function WorldAdoptionsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Session Adoption Review"
        action={
          <Link
            href={`/worlds/${encodeURIComponent(id)}`}
            className="text-xs hover:underline"
            style={{ color: "var(--accent)" }}
          >
            Back to workspace
          </Link>
        }
      />
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
        <AdoptionReviewPanel worldId={id} />
      </div>
    </div>
  );
}
