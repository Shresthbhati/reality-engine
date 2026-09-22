import { notFound } from "next/navigation";
import { getReport, getResult } from "@/lib/data";
import EditorClient from "./EditorClient";

export default async function ReportEditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const report = getReport(id);
  if (!report) notFound();

  const results = report.resultIds
    .map((resultId) => getResult(resultId))
    .filter((r): r is NonNullable<typeof r> => r != null);

  return <EditorClient report={report} results={results} />;
}
