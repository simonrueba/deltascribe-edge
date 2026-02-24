"use client";

import { useState } from "react";
import type { WorkflowStep, WorkflowStepUpdate } from "@/types/api";

const STEPS: { key: WorkflowStep; label: string; short: string }[] = [
  { key: "intake", label: "Receive images", short: "Intake" },
  { key: "preprocess", label: "Preprocess", short: "Preproc" },
  { key: "retrieve_context", label: "Retrieve context", short: "Context" },
  { key: "retrieve_evidence", label: "Retrieve evidence", short: "Evidence" },
  { key: "inference", label: "Run model", short: "Model" },
  { key: "validate", label: "Validate output", short: "Validate" },
  { key: "assemble", label: "Assemble report", short: "Assemble" },
];

function formatDuration(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

interface WorkflowTimelineProps {
  steps: WorkflowStepUpdate[];
  analyzing: boolean;
}

export function WorkflowTimeline({ steps, analyzing }: WorkflowTimelineProps) {
  const [expanded, setExpanded] = useState(false);

  if (steps.length === 0 && !analyzing) return null;

  const completedCount = STEPS.filter((s) =>
    steps.some((u) => u.step === s.key && u.status === "complete")
  ).length;
  const runningStep = STEPS.find((s) =>
    steps.some((u) => u.step === s.key && u.status === "running")
  );
  const hasError = steps.some((u) => u.status === "error");
  const totalMs = steps
    .filter((u) => u.status === "complete" && u.duration_ms != null)
    .reduce((sum, u) => sum + (u.duration_ms ?? 0), 0);

  return (
    <div className="border-b border-slate-200 bg-slate-50/50">
      {/* Compact single-line summary */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2 flex items-center gap-2 text-left hover:bg-slate-100/50 transition-colors"
      >
        <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 shrink-0">
          Workflow
        </span>

        {/* Step dots */}
        <div className="flex items-center gap-1 ml-1">
          {STEPS.map((s) => {
            const latest = [...steps].reverse().find((u) => u.step === s.key);
            const status = latest?.status ?? "pending";
            return (
              <div key={s.key} title={`${s.label}${latest?.duration_ms != null ? ` (${formatDuration(latest.duration_ms)})` : ""}`}>
                {status === "complete" ? (
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                ) : status === "running" ? (
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                ) : status === "error" ? (
                  <div className="w-2 h-2 rounded-full bg-red-500" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-slate-200" />
                )}
              </div>
            );
          })}
        </div>

        {/* Status text */}
        <span className="text-[11px] text-slate-400 ml-auto tabular-nums shrink-0">
          {hasError ? (
            <span className="text-red-500">Error</span>
          ) : runningStep ? (
            <span className="text-blue-600">{runningStep.short}...</span>
          ) : completedCount === STEPS.length ? (
            <span className="text-emerald-600">{formatDuration(totalMs)}</span>
          ) : completedCount > 0 ? (
            `${completedCount}/${STEPS.length}`
          ) : (
            "Starting..."
          )}
        </span>

        {/* Expand chevron */}
        <svg
          className={`w-3 h-3 text-slate-300 transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded detail view */}
      {expanded && (
        <div className="px-4 pb-2 flex flex-col gap-0.5">
          {STEPS.map((s) => {
            const latest = [...steps].reverse().find((u) => u.step === s.key);
            const status = latest?.status ?? "pending";
            const duration = latest?.duration_ms;

            return (
              <div key={s.key} className="flex items-center gap-2 py-0.5">
                <div className="w-4 h-4 flex items-center justify-center shrink-0">
                  {status === "complete" ? (
                    <svg className="w-3.5 h-3.5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : status === "running" ? (
                    <div className="w-3 h-3 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
                  ) : status === "error" ? (
                    <svg className="w-3.5 h-3.5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  ) : (
                    <div className="w-1.5 h-1.5 rounded-full bg-slate-200" />
                  )}
                </div>
                <span className={`text-[12px] ${
                  status === "complete" ? "text-slate-600" :
                  status === "running" ? "text-blue-700 font-medium" :
                  status === "error" ? "text-red-600" :
                  "text-slate-300"
                }`}>
                  {s.label}
                </span>
                {status === "complete" && duration != null && (
                  <span className="text-[11px] text-slate-300 ml-auto tabular-nums">
                    {formatDuration(duration)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
