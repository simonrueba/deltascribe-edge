"use client";

import { useRef } from "react";
import type { DemoPatient } from "@/types/api";
import { CaseSelector } from "@/components/features/case-selector";
import { DictationButton } from "@/components/features/dictation-button";

interface ControlBarProps {
  patients: DemoPatient[];
  selectedPatient: DemoPatient | null;
  onSelectPatient: (p: DemoPatient) => void;
  canAnalyze: boolean;
  analyzing: boolean;
  onAnalyze: () => void;
  onReset: () => void;
  hasResult: boolean;
  onUploadClick: () => void;
  fhirContext: Record<string, unknown> | null;
  onLoadFhir: (data: Record<string, unknown>) => void;
  dictationAudio: string | null;
  onTranscript: (text: string, audioBase64: string) => void;
}

export function ControlBar({
  patients,
  selectedPatient,
  onSelectPatient,
  canAnalyze,
  analyzing,
  onAnalyze,
  onReset,
  hasResult,
  onUploadClick,
  fhirContext,
  onLoadFhir,
  onTranscript,
}: ControlBarProps) {
  const fhirInputRef = useRef<HTMLInputElement>(null);

  const handleFhirFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result as string);
        onLoadFhir(parsed);
      } catch {
        // invalid JSON — silently ignore
      }
    };
    reader.readAsText(file);
    // Reset so re-selecting the same file triggers onChange
    e.target.value = "";
  };

  return (
    <div className="h-14 flex items-center gap-4 px-5 border-b border-slate-200 bg-white shrink-0">
      {/* Patient selector */}
      <CaseSelector
        patients={patients}
        selected={selectedPatient}
        onSelect={onSelectPatient}
      />

      <div className="flex items-center gap-1.5">
        <button
          onClick={onUploadClick}
          className="text-[13px] text-slate-500 hover:text-slate-700 transition-colors px-3 py-2 border border-slate-200 rounded-md hover:bg-slate-50"
        >
          Upload
        </button>

        {/* FHIR context import */}
        <input
          ref={fhirInputRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={handleFhirFile}
        />
        <button
          onClick={() => fhirInputRef.current?.click()}
          className={`text-[13px] transition-colors px-3 py-2 border rounded-md ${
            fhirContext
              ? "bg-green-50 border-green-300 text-green-700"
              : "text-slate-500 hover:text-slate-700 border-slate-200 hover:bg-slate-50"
          }`}
        >
          {fhirContext ? "FHIR loaded" : "FHIR"}
        </button>

        {/* Dictation */}
        <DictationButton onTranscript={onTranscript} disabled={analyzing} />
      </div>

      <div className="w-px h-7 bg-slate-200" />

      {/* Analyze */}
      <button
        disabled={!canAnalyze}
        onClick={onAnalyze}
        className={`px-6 py-2 text-[13px] font-semibold rounded-md transition-all ${
          analyzing
            ? "bg-blue-50 text-blue-700 border border-blue-200"
            : canAnalyze
              ? "bg-primary text-white hover:bg-blue-900 shadow-sm"
              : "bg-slate-100 text-slate-400 cursor-not-allowed"
        }`}
      >
        {analyzing ? (
          <span className="flex items-center gap-2">
            <span className="w-4 h-4 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin" />
            Analyzing...
          </span>
        ) : (
          "Run Analysis"
        )}
      </button>

      {hasResult && (
        <button
          onClick={onReset}
          className="text-[13px] text-slate-400 hover:text-slate-600 transition-colors"
        >
          Reset
        </button>
      )}

    </div>
  );
}
