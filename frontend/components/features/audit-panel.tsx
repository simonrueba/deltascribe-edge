"use client";

import { useState } from "react";
import type { AuditEntry, RetrievalEvidenceResult } from "@/types/api";

interface AuditPanelProps {
  auditTrail: AuditEntry[];
  retrievalEvidence: RetrievalEvidenceResult | null;
}

function ConfidenceBadge({ uncertainty }: { uncertainty: string }) {
  // uncertainty = model's uncertainty level; invert for display confidence
  const config =
    uncertainty === "low"
      ? { label: "High", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" }
      : uncertainty === "medium"
        ? { label: "Med", cls: "bg-amber-50 text-amber-700 border-amber-200" }
        : { label: "Low", cls: "bg-red-50 text-red-600 border-red-200" };
  return (
    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${config.cls}`}>
      {config.label}
    </span>
  );
}

export function AuditPanel({ auditTrail, retrievalEvidence }: AuditPanelProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  if (auditTrail.length === 0 && !retrievalEvidence) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-400 text-sm">
          Run analysis to see supporting evidence.
        </p>
      </div>
    );
  }

  const toggle = (key: string) =>
    setExpandedSection((prev) => (prev === key ? null : key));

  // Group audit entries by source type for cleaner display
  const imageEvidence = auditTrail.filter(
    (e) => e.source_type === "current_image" || e.source_type === "prior_image"
  );
  const retrievalAudit = auditTrail.filter(
    (e) => e.source_type === "similar_case" || e.source_type === "zero_shot"
  );

  // Filter zero-shot to only meaningful predictions (>15% confidence)
  const meaningfulPredictions = retrievalEvidence?.zero_shot_predictions.filter(
    (p) => p.confidence > 0.15 && !p.label.startsWith("delta:")
  ) ?? [];

  return (
    <div className="h-full overflow-y-auto">
      {/* Model findings with evidence */}
      {imageEvidence.length > 0 && (
        <div className="border-b border-slate-100">
          <button
            onClick={() => toggle("findings")}
            className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-slate-50/50 transition-colors"
          >
            <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
              Model Claims ({imageEvidence.length})
            </h3>
            <svg
              className={`w-3.5 h-3.5 text-slate-300 transition-transform ${expandedSection === "findings" ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expandedSection === "findings" && (
            <div className="px-5 pb-3">
              {imageEvidence.map((entry, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b border-slate-50 last:border-0">
                  <ConfidenceBadge uncertainty={entry.uncertainty} />
                  <p className="text-[12px] text-slate-600 leading-snug">{entry.claim}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Similar cases */}
      {retrievalEvidence && retrievalEvidence.similar_cases.length > 0 && (
        <div className="border-b border-slate-100">
          <button
            onClick={() => toggle("cases")}
            className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-slate-50/50 transition-colors"
          >
            <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
              Similar Cases ({retrievalEvidence.similar_cases.length})
            </h3>
            <svg
              className={`w-3.5 h-3.5 text-slate-300 transition-transform ${expandedSection === "cases" ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expandedSection === "cases" && (
            <div className="px-5 pb-3">
              {retrievalEvidence.similar_cases.map((c, i) => (
                <div key={i} className="flex items-center gap-2 py-1.5 border-b border-slate-50 last:border-0">
                  <span className="text-[10px] font-medium text-slate-400 tabular-nums shrink-0">
                    {Math.round(c.similarity_score * 100)}%
                  </span>
                  <span className="text-[12px] text-slate-600">
                    {c.known_findings.join(", ")} ({c.delta_label})
                  </span>
                  <span className="text-[10px] text-slate-400 ml-auto shrink-0">{c.source}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Zero-shot predictions (only meaningful ones) */}
      {meaningfulPredictions.length > 0 && (
        <div className="border-b border-slate-100">
          <button
            onClick={() => toggle("zeroshot")}
            className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-slate-50/50 transition-colors"
          >
            <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
              Independent Classification ({meaningfulPredictions.length})
            </h3>
            <svg
              className={`w-3.5 h-3.5 text-slate-300 transition-transform ${expandedSection === "zeroshot" ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expandedSection === "zeroshot" && (
            <div className="px-5 pb-3">
              {meaningfulPredictions.map((p, i) => (
                <div key={i} className="flex items-center gap-2 py-1.5 border-b border-slate-50 last:border-0">
                  <div className="flex-1 flex items-center gap-2">
                    <span className="text-[12px] text-slate-600 capitalize">{p.label}</span>
                    <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full"
                        style={{ width: `${Math.round(p.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-slate-400 tabular-nums shrink-0">
                      {Math.round(p.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))}
              <p className="text-[10px] text-slate-400 mt-1.5">
                MedSigLIP zero-shot classification (independent of MedGemma)
              </p>
            </div>
          )}
        </div>
      )}

      {/* Retrieval-based audit evidence */}
      {retrievalAudit.length > 0 && (
        <div className="border-b border-slate-100">
          <button
            onClick={() => toggle("retrievalAudit")}
            className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-slate-50/50 transition-colors"
          >
            <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
              Retrieval Evidence ({retrievalAudit.length})
            </h3>
            <svg
              className={`w-3.5 h-3.5 text-slate-300 transition-transform ${expandedSection === "retrievalAudit" ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expandedSection === "retrievalAudit" && (
            <div className="px-5 pb-3">
              {retrievalAudit.map((entry, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b border-slate-50 last:border-0">
                  <ConfidenceBadge uncertainty={entry.uncertainty} />
                  <div className="min-w-0">
                    <p className="text-[12px] text-slate-600 leading-snug">{entry.claim}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5 truncate">
                      {entry.source_ref}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
