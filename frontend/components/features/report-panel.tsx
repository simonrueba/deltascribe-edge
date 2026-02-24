"use client";

import { useState, useCallback } from "react";
import Markdown from "react-markdown";
import type { AnalysisResult } from "@/types/api";
import { exportFHIR } from "@/lib/api";

const mdComponents = {
  p: (props: React.ComponentProps<"p">) => (
    <p className="text-[14px] text-slate-700 leading-relaxed mb-2" {...props} />
  ),
  ul: (props: React.ComponentProps<"ul">) => (
    <ul className="text-[14px] text-slate-700 leading-relaxed list-disc pl-5 mb-2" {...props} />
  ),
  li: (props: React.ComponentProps<"li">) => (
    <li className="mb-0.5" {...props} />
  ),
  strong: (props: React.ComponentProps<"strong">) => (
    <strong className="font-semibold text-slate-800" {...props} />
  ),
};

interface ReportPanelProps {
  result: AnalysisResult | null;
  dictationTranscript?: string | null;
}

function EditableSection({
  label,
  content,
  onChange,
}: {
  label: string;
  content: string;
  onChange: (val: string) => void;
}) {
  const [editing, setEditing] = useState(false);

  if (editing) {
    return (
      <div className="px-5 py-4 border-b border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
            {label}
          </h3>
          <button
            onClick={() => setEditing(false)}
            className="text-[11px] font-medium text-primary hover:text-blue-800 transition-colors"
          >
            Done
          </button>
        </div>
        <textarea
          value={content}
          onChange={(e) => onChange(e.target.value)}
          className="w-full min-h-[120px] text-[14px] text-slate-700 leading-relaxed p-3 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 resize-y bg-white"
        />
      </div>
    );
  }

  return (
    <div className="px-5 py-4 border-b border-slate-100 group">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
          {label}
        </h3>
        <button
          onClick={() => setEditing(true)}
          className="text-[11px] text-slate-400 hover:text-primary transition-colors opacity-0 group-hover:opacity-100"
        >
          Edit
        </button>
      </div>
      <Markdown components={mdComponents}>{content}</Markdown>
    </div>
  );
}

export function ReportPanel({ result, dictationTranscript }: ReportPanelProps) {
  const [copied, setCopied] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [findings, setFindings] = useState<string | null>(null);
  const [impression, setImpression] = useState<string | null>(null);
  const [addendum, setAddendum] = useState("");

  const handleInsertDictation = useCallback(() => {
    if (dictationTranscript) {
      setAddendum((prev) =>
        prev ? `${prev}\n\n${dictationTranscript}` : dictationTranscript
      );
    }
  }, [dictationTranscript]);

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-400 text-sm">
          Run analysis to generate a draft report.
        </p>
      </div>
    );
  }

  // Initialize editable state from result
  const currentFindings = findings ?? result.draft_report.findings ?? "";
  const currentImpression = impression ?? result.draft_report.impression ?? "";

  const fullText = [
    `FINDINGS:\n${currentFindings}`,
    `\nIMPRESSION:\n${currentImpression}`,
    addendum ? `\nADDENDUM:\n${addendum}` : "",
  ].filter(Boolean).join("\n");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const blob = await exportFHIR(result.job_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `deltascribe-${result.job_id}.fhir.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    } finally {
      setExporting(false);
    }
  };

  const isModified = findings !== null || impression !== null || addendum.length > 0;

  return (
    <div className="h-full overflow-y-auto">
      {/* Model narrative (raw LLM output) — shown first */}
      {result.model_narrative && (
        <div className="px-5 py-4 border-b border-slate-100 bg-blue-50/40">
          <h3 className="text-[12px] font-semibold uppercase tracking-wide text-blue-500 mb-2">
            Model Narrative
            <span className="ml-2 text-[10px] font-normal text-blue-400">
              Raw MedGemma output (512 tokens)
            </span>
          </h3>
          <div className="text-[14px] text-slate-700 leading-relaxed italic whitespace-pre-line">
            {result.model_narrative}
          </div>
        </div>
      )}

      {/* Delta summary (read-only) */}
      <div className="px-5 py-4 border-b border-slate-100">
        <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400 mb-2">
          Summary of Changes
        </h3>
        <Markdown components={mdComponents}>{result.delta_summary}</Markdown>
      </div>

      {/* Editable findings */}
      <EditableSection
        label="Findings"
        content={currentFindings}
        onChange={setFindings}
      />

      {/* Editable impression */}
      <EditableSection
        label="Impression"
        content={currentImpression}
        onChange={setImpression}
      />

      {/* Addendum / clinician notes */}
      <div className="px-5 py-4 border-b border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-[12px] font-semibold uppercase tracking-wide text-slate-400">
            Clinician Addendum
          </h3>
          {dictationTranscript && (
            <button
              onClick={handleInsertDictation}
              className="text-[11px] font-medium text-primary hover:text-blue-800 transition-colors flex items-center gap-1"
            >
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M8 1a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0v-4A2.5 2.5 0 0 0 8 1Z" fill="currentColor" />
                <path d="M3.5 6.5a.5.5 0 0 1 1 0v1a3.5 3.5 0 1 0 7 0v-1a.5.5 0 0 1 1 0v1a4.5 4.5 0 0 1-4 4.473V13.5h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-1.527a4.5 4.5 0 0 1-4-4.473v-1Z" fill="currentColor" />
              </svg>
              Insert dictation
            </button>
          )}
        </div>
        <textarea
          value={addendum}
          onChange={(e) => setAddendum(e.target.value)}
          placeholder="Add clinical notes, corrections, or additional observations..."
          className="w-full min-h-[80px] text-[14px] text-slate-700 leading-relaxed p-3 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 resize-y bg-white placeholder:text-slate-300"
        />
      </div>

      {/* Actions */}
      <div className="px-5 py-3 flex items-center gap-3 border-b border-slate-100">
        <button
          onClick={handleCopy}
          className="text-[13px] text-slate-500 hover:text-slate-700 transition-colors px-3 py-1.5 border border-slate-200 rounded-md hover:bg-slate-50"
        >
          {copied ? "Copied!" : "Copy Report"}
        </button>
        <button
          onClick={handleExport}
          disabled={exporting}
          className="text-[13px] text-slate-500 hover:text-slate-700 transition-colors px-3 py-1.5 border border-slate-200 rounded-md hover:bg-slate-50 disabled:opacity-40"
        >
          {exporting ? "Exporting..." : "Export FHIR"}
        </button>
        {isModified && (
          <span className="text-[11px] text-amber-600 ml-auto">
            Modified by clinician
          </span>
        )}
      </div>
    </div>
  );
}
