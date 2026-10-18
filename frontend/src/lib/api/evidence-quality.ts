/** Evidence Quality Firewall endpoints (see apps/api/routes_evidence_quality.py). */
import { apiGet, apiPut } from "./client";
import type { EvidenceQualityItem, QualityDecision, QualityPolicy, QualityResult } from "./evidence-quality-model";

export interface EvidenceQualityList {
  items: EvidenceQualityItem[];
  counts: Record<QualityResult, number> & { total: number };
  not_screened_non_photo: number;
  policy: QualityPolicy;
}

export function listEvidenceQuality(): Promise<EvidenceQualityList> {
  return apiGet<EvidenceQualityList>("/api/evidence/quality");
}

export function getEvidenceQuality(id: string): Promise<EvidenceQualityItem & { policy: QualityPolicy }> {
  return apiGet(`/api/evidence/${encodeURIComponent(id)}/quality`);
}

export interface DecisionRecord {
  evidence_id: string;
  decision: QualityDecision;
  reason: string | null;
  decided_at: string | null;
}

export function putEvidenceDecision(id: string, decision: QualityDecision, reason: string | null): Promise<DecisionRecord> {
  return apiPut<DecisionRecord>(`/api/evidence/${encodeURIComponent(id)}/decision`, { decision, reason: reason || null });
}
