"use client";

import { useState, useRef, useEffect } from "react";
import type { DemoPatient } from "@/types/api";

interface CaseSelectorProps {
  patients: DemoPatient[];
  selected: DemoPatient | null;
  onSelect: (p: DemoPatient) => void;
}

const deltaColor: Record<string, string> = {
  improved: "text-[var(--color-delta-improved)]",
  worsened: "text-[var(--color-delta-worsened)]",
  stable: "text-[var(--color-delta-stable)]",
  uncertain: "text-[var(--color-delta-uncertain)]",
};

const deltaBadgeBg: Record<string, string> = {
  improved: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  worsened: "bg-red-50 text-red-700 ring-red-200",
  stable: "bg-amber-50 text-amber-700 ring-amber-200",
  uncertain: "bg-slate-50 text-slate-600 ring-slate-200",
};

export function CaseSelector({ patients, selected, onSelect }: CaseSelectorProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      {/* Trigger */}
      <button
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-2.5 pl-3 pr-2.5 py-1.5 rounded-md border transition-all text-left min-w-[260px] ${
          open
            ? "border-primary/40 ring-2 ring-primary/10 bg-white"
            : selected
              ? "border-slate-200 bg-white hover:border-slate-300"
              : "border-slate-200 bg-white hover:border-slate-300"
        }`}
      >
        {selected ? (
          <>
            <span className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" className="text-primary">
                <path d="M8 1a3 3 0 0 0-3 3v1H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-1V4a3 3 0 0 0-3-3Zm0 1.5A1.5 1.5 0 0 1 9.5 4v1h-3V4A1.5 1.5 0 0 1 8 2.5Z" fill="currentColor" />
              </svg>
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-medium text-slate-800 truncate">
                {selected.patient_id}
              </div>
              <div className="text-[11px] text-slate-400 truncate">
                {selected.description}
              </div>
            </div>
            <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ring-1 shrink-0 ${deltaBadgeBg[selected.expected_delta] ?? deltaBadgeBg.uncertain}`}>
              {selected.expected_delta}
            </span>
          </>
        ) : (
          <>
            <span className="w-6 h-6 rounded bg-slate-100 flex items-center justify-center shrink-0">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" className="text-slate-400">
                <path d="M8 1a3 3 0 0 0-3 3v1H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-1V4a3 3 0 0 0-3-3Zm0 1.5A1.5 1.5 0 0 1 9.5 4v1h-3V4A1.5 1.5 0 0 1 8 2.5Z" fill="currentColor" />
              </svg>
            </span>
            <span className="text-[13px] text-slate-400">Select a case...</span>
          </>
        )}
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          className={`text-slate-400 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute top-full left-0 mt-1.5 w-[380px] bg-white border border-slate-200 rounded-lg shadow-lg shadow-slate-200/50 z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-slate-100">
            <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Demo Cases ({patients.length})
            </span>
          </div>
          <div className="max-h-[320px] overflow-y-auto py-1">
            {patients.map((p) => {
              const isSelected = selected?.patient_id === p.patient_id;
              return (
                <button
                  key={p.patient_id}
                  onClick={() => { onSelect(p); setOpen(false); }}
                  className={`w-full text-left px-3 py-2.5 transition-colors ${
                    isSelected
                      ? "bg-primary/5"
                      : "hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-start gap-2.5">
                    <span className={`mt-0.5 w-5 h-5 rounded flex items-center justify-center shrink-0 text-[10px] font-bold ${
                      isSelected
                        ? "bg-primary text-white"
                        : "bg-slate-100 text-slate-500"
                    }`}>
                      {p.patient_id.replace(/\D/g, "").slice(-2)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`text-[13px] font-medium ${isSelected ? "text-primary" : "text-slate-800"}`}>
                          {p.patient_id}
                        </span>
                        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ring-1 ${deltaBadgeBg[p.expected_delta] ?? deltaBadgeBg.uncertain}`}>
                          {p.expected_delta}
                        </span>
                      </div>
                      <p className="text-[12px] text-slate-500 mt-0.5 leading-snug">
                        {p.description}
                      </p>
                      <div className="flex items-center gap-1 mt-1 text-[11px] text-slate-400">
                        <span>{p.prior.finding}</span>
                        <span className={deltaColor[p.expected_delta] ?? ""}>
                          {"\u2192"}
                        </span>
                        <span>{p.current.finding}</span>
                        <span className="text-slate-300 mx-1">{"\u00b7"}</span>
                        <span>{p.prior.study_index ?? "Prior"} {"\u2192"} {p.current.study_index ?? "Current"}</span>
                      </div>
                    </div>
                    {isSelected && (
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-primary shrink-0 mt-0.5">
                        <path d="M3.5 8.5l3 3 6-6.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
