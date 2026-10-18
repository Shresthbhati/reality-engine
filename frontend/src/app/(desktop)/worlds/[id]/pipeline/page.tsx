import Link from "next/link";
import PageHeader from "@/components/ui/PageHeader";
import PipelineGraph from "@/components/pipeline/PipelineGraph";

/** Pipeline view: the real stages of the world's newest reconstruction run, plus an honest summary. */
export default async function WorldPipelinePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Pipeline"
        action={
          <Link href={`/worlds/${encodeURIComponent(id)}`} className="text-xs hover:underline" style={{ color: "var(--accent)" }}>
            Back to workspace
          </Link>
        }
      />
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
        <PipelineGraph worldId={id} />
      </div>
    </div>
  );
}
