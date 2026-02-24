"use client";

import type { Finding, DeltaStatus } from "@/types/api";

const DELTA: Record<DeltaStatus, { label: string; color: string; bg: string }> = {
  improved: { label: "Improved", color: "text-emerald-700", bg: "bg-emerald-50" },
  stable: { label: "Stable", color: "text-amber-700", bg: "bg-amber-50" },
  worsened: { label: "Worsened", color: "text-red-700", bg: "bg-red-50" },
  uncertain: { label: "Uncertain", color: "text-slate-500", bg: "bg-slate-50" },
};

interface FindingsPanelProps {
  findings: Finding[];
  onHighlight: (i: number | null) => void;
  highlighted: number | null;
}

export function FindingsPanel({ findings, onHighlight, highlighted }: FindingsPanelProps) {
  if (findings.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-400 text-sm">
          Select a patient case and run analysis to see findings.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      {findings.map((f, i) => {
        const d = DELTA[f.delta];
        const active = highlighted === i;
        return (
          <div
            key={i}
            className={`finding-row border-b border-slate-100 px-5 py-3.5 cursor-default transition-colors ${
              active ? "bg-blue-50/50" : "hover:bg-slate-50"
            }`}
            style={{ animationDelay: `${i * 0.05}s` }}
            onMouseEnter={() => onHighlight(i)}
            onMouseLeave={() => onHighlight(null)}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="text-[14px] font-medium text-slate-800 capitalize">
                  {f.label}
                </span>
                <span className={`text-[12px] font-medium px-2 py-0.5 rounded-full ${d.color} ${d.bg}`}>
                  {d.label}
                </span>
              </div>
              {f.uncertainty === "high" && (
                <span className="text-[11px] text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full">
                  Low confidence
                </span>
              )}
            </div>
            <p className="text-[13px] text-slate-600 mt-1.5 leading-relaxed">
              {f.rationale}
            </p>
          </div>
        );
      })}
    </div>
  );
}
