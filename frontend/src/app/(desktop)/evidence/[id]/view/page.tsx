import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import InspectorDecision from "@/components/evidence/InspectorDecision";
import {
  CheckRows,
  DECISION_NOTE,
  EXPLANATION_NOTE,
  Meta,
  Panel,
  ResultChip,
  ScoreRing,
  TONE_COLOR,
} from "@/components/evidence/QualityParts";
import { evidenceArtifactUrl, getEvidenceQuality, isApiError } from "@/lib/api";
import {
  RESULT_LABEL,
  barPercent,
  contributionSummary,
  formatBytes,
  formatDimensions,
  resultOf,
  type EvidenceQualityItem,
  type QualityPolicy,
} from "@/lib/api/evidence-quality-model";

/** A real measured value on a 0..max bar with the fail/warn thresholds marked. */
function MeterBar({ label, valueText, value, fail, warn, max }: { label: string; valueText: string; value: number; fail: number; warn: number; max: number }) {
  const color = value < fail ? TONE_COLOR.error : value < warn ? TONE_COLOR.warning : TONE_COLOR.success;
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="font-mono-num" style={{ color: "var(--text-primary)" }}>{valueText}</span>
      </div>
      <div className="relative h-2 rounded" style={{ background: "var(--border)" }}>
        <div className="h-2 rounded" style={{ width: `${barPercent(value, max)}%`, background: color }} />
        {[fail, warn].map((t) => (
          <span key={t} className="absolute top-[-2px] h-3 w-px" style={{ left: `${barPercent(t, max)}%`, background: "var(--text-secondary)" }} />
        ))}
      </div>
      <div className="text-[11px] mt-0.5" style={{ color: "var(--text-tertiary)" }}>
        Fail below {fail}, warn below {warn}
      </div>
    </div>
  );
}

function QualityBars({ item, policy }: { item: EvidenceQualityItem; policy: QualityPolicy }) {
  if (!item.inspected || !item.width || !item.height) return null;
  const mp = (item.width * item.height) / 1e6;
  return (
    <>
      <MeterBar
        label="Resolution"
        valueText={`${mp.toFixed(2)} MP`}
        value={mp}
        fail={policy.resolution_fail_mp}
        warn={policy.resolution_warn_mp}
        max={policy.resolution_warn_mp * 3}
      />
      {item.sharpness !== null && (
        <MeterBar
          label="Sharpness (Laplacian variance, relative)"
          valueText={item.sharpness.toFixed(1)}
          value={item.sharpness}
          fail={policy.sharpness_fail}
          warn={policy.sharpness_warn}
          max={policy.sharpness_warn * 3}
        />
      )}
      <p className="text-xs mb-2" style={{ color: "var(--text-tertiary)" }}>
        Exposure is not measured by the ingest checks, so it is not shown.
      </p>
    </>
  );
}

export default async function EvidenceInspectorPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  let item: EvidenceQualityItem & { policy: QualityPolicy };
  try {
    item = await getEvidenceQuality(id);
  } catch (e) {
    if (isApiError(e) && e.code === "not_found") notFound();
    throw e;
  }
  const result = resultOf(item);
  const c = item.contribution;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex flex-1 min-h-0 flex-col lg:flex-row">
        <div className="flex-1 relative flex flex-col min-h-[300px]" style={{ background: "var(--bg-base)" }}>
          <Link
            href={`/evidence/${item.id}`}
            className="absolute top-4 left-4 z-10 inline-flex items-center gap-1.5 px-3 h-8 rounded-md text-sm font-medium"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back
          </Link>
          <div className="flex-1 flex items-center justify-center p-4 pt-14 min-h-0">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={evidenceArtifactUrl(item.id)} alt={item.name} className="max-h-full max-w-full object-contain" />
          </div>
          <div className="px-4 py-2 text-xs" style={{ color: "var(--text-tertiary)", borderTop: "1px solid var(--border)" }}>
            Original image as stored. No analysis overlay is shown: the pipeline does not store feature locations or heatmaps for this file.
          </div>
        </div>

        <aside
          className="lg:w-[420px] shrink-0 border-l overflow-y-auto p-4 flex flex-col gap-3"
          style={{ borderColor: "var(--border)", background: "var(--bg-base)" }}
        >
          <div className="flex items-start justify-between gap-2">
            <h2 className="text-sm font-semibold break-all" style={{ color: "var(--text-primary)" }}>{item.name}</h2>
            <ResultChip result={result} />
          </div>

          <Panel title="Quality">
            <div className="flex items-center gap-3 mb-3">
              <ScoreRing score={item.score} size={48} />
              <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                Score summarises the checks (100 minus {item.policy.warn_penalty} per warning and {item.policy.fail_penalty} per failure). Not a calibrated measure.
              </span>
            </div>
            <QualityBars item={item} policy={item.policy} />
            <CheckRows checks={item.checks} />
          </Panel>

          <Panel title="Spatial value">
            <p className="text-sm" style={{ color: "var(--text-primary)" }}>{contributionSummary(c)}</p>
            {c.available && c.median_shift_fraction != null && (
              <Meta label="Median feature shift" value={`${(c.median_shift_fraction * 100).toFixed(1)}% of image diagonal`} />
            )}
            {c.available && c.basis && <p className="text-xs mt-1" style={{ color: "var(--text-tertiary)" }}>{c.basis}</p>}
            {c.available && c.best_partner_id && (
              <div className="mt-3">
                <div className="text-xs mb-1" style={{ color: "var(--text-tertiary)" }}>
                  Best-overlapping photo ({c.verified_matches} verified matches)
                </div>
                <Link href={`/evidence/${c.best_partner_id}/view`} className="block">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={evidenceArtifactUrl(c.best_partner_id)}
                    alt={c.best_partner_name ?? "Best-overlapping photo"}
                    loading="lazy"
                    className="w-full max-h-32 object-cover rounded"
                    style={{ border: "1px solid var(--border)" }}
                  />
                  <span className="text-xs" style={{ color: "var(--accent)" }}>{c.best_partner_name ?? c.best_partner_id}</span>
                </Link>
              </div>
            )}
          </Panel>

          <Panel title="Decision and reason">
            <Meta label="Automatic result" value={item.auto_result ? RESULT_LABEL[item.auto_result] : "Not inspected"} />
            <Meta label="Your decision" value={item.decision ? RESULT_LABEL[item.decision.decision] : "None yet"} />
            {item.decision?.reason && <Meta label="Reason" value={item.decision.reason} />}
            <p className="text-sm my-2" style={{ color: "var(--text-primary)" }}>{item.explanation}</p>
            <p className="text-xs mb-3" style={{ color: "var(--text-tertiary)" }}>{EXPLANATION_NOTE}</p>
            <InspectorDecision
              evidenceId={item.id}
              current={item.decision?.decision ?? null}
              initialReason={item.decision?.reason ?? null}
            />
            <p className="text-xs mt-2" style={{ color: "var(--text-tertiary)" }}>{DECISION_NOTE}</p>
          </Panel>

          <Panel title="Metadata">
            <Meta label="Source" value={item.source} />
            <Meta label="Resolution" value={formatDimensions(item)} />
            <Meta label="File size" value={formatBytes(item.size)} />
            <Meta label="Class" value={item.evidence_class ?? "Not inspected"} />
            {item.class_basis && <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>{item.class_basis}</p>}
            <Meta label="Location" value={item.gps ? `${item.gps[0].toFixed(4)}, ${item.gps[1].toFixed(4)}` : "No GPS in file"} />
            <Meta
              label="World"
              value={
                item.world_id ? (
                  <Link href={`/worlds/${item.world_id}`} style={{ color: "var(--accent)" }}>{item.world_id}</Link>
                ) : (
                  "Unattached"
                )
              }
            />
          </Panel>
        </aside>
      </div>
    </div>
  );
}
