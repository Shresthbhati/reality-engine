/**
 * Analysis, Results and Reports have no backend resource in this build.
 *
 * Rather than a mock array, every entry point here returns an explicit
 * `unsupportedResource` so a page can render a reasoned empty state
 * ("Analysis is not served by the Reality Engine API yet …") instead of
 * pretending data exists. The row shapes are still typed so screens compile
 * against the real UI model even though the data does not exist yet.
 */
import { unsupportedResource, type UnsupportedResource } from "./unsupported";
import type { AnalysisRow, ReportRow, ResultRow } from "@/lib/types";

export async function getAnalysis(): Promise<{
  items: AnalysisRow[];
  resource: UnsupportedResource;
}> {
  const r = unsupportedResource("Analysis");
  return { items: r.items as AnalysisRow[], resource: r };
}

export async function getAnalysisDetail(id: string): Promise<{
  row: AnalysisRow | null;
  resource: UnsupportedResource;
}> {
  void id;
  const r = unsupportedResource("Analysis");
  return { row: null, resource: r };
}

export async function getResults(): Promise<{
  items: ResultRow[];
  resource: UnsupportedResource;
}> {
  const r = unsupportedResource("Results");
  return { items: r.items as ResultRow[], resource: r };
}

export async function getResultDetail(id: string): Promise<{
  row: ResultRow | null;
  resource: UnsupportedResource;
}> {
  void id;
  const r = unsupportedResource("Results");
  return { row: null, resource: r };
}

export async function getReports(): Promise<{
  items: ReportRow[];
  resource: UnsupportedResource;
}> {
  const r = unsupportedResource("Reports");
  return { items: r.items as ReportRow[], resource: r };
}

export async function getReportDetail(id: string): Promise<{
  row: ReportRow | null;
  resource: UnsupportedResource;
}> {
  void id;
  const r = unsupportedResource("Reports");
  return { row: null, resource: r };
}
