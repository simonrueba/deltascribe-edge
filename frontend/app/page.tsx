"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Header } from "@/components/features/header";
import { ImageViewer } from "@/components/features/image-viewer";
import { ControlBar } from "@/components/features/control-bar";
import { FindingsPanel } from "@/components/features/findings-panel";
import { ReportPanel } from "@/components/features/report-panel";
import { AuditPanel } from "@/components/features/audit-panel";
import { UploadDialog } from "@/components/features/upload-zone";
import { WorkflowTimeline } from "@/components/features/workflow-timeline";
import {
  startAnalysis,
  getDemoPatientImages,
  getAnalysisStatus,
  getAnalysisResult,
  createAnalysisWebSocket,
  listDemoPatients,
} from "@/lib/api";
import type {
  DemoPatient,
  AnalysisResult,
  AnalysisStatus,
  WorkflowStepUpdate,
} from "@/types/api";

type Tab = "findings" | "report" | "audit";

export default function Home() {
  // Data
  const [patients, setPatients] = useState<DemoPatient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<DemoPatient | null>(null);
  const [priorImage, setPriorImage] = useState<string | null>(null);
  const [currentImage, setCurrentImage] = useState<string | null>(null);

  // Analysis
  const [analyzing, setAnalyzing] = useState(false);
  const [steps, setSteps] = useState<WorkflowStepUpdate[]>([]);
  const [stepsTarget, setStepsTarget] = useState<WorkflowStepUpdate[]>([]);
  const [_progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // P2 features
  const [dictationAudio, setDictationAudio] = useState<string | null>(null);
  const [dictationTranscript, setDictationTranscript] = useState<string | null>(null);
  const [fhirContext, setFhirContext] = useState<Record<string, unknown> | null>(null);

  // UI
  const [tab, setTab] = useState<Tab>("findings");
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stagger step display: drip stepsTarget → steps at 120ms intervals
  // so batched updates (from late WebSocket connect) animate one at a time.
  useEffect(() => {
    if (steps.length >= stepsTarget.length) return;
    const timer = setTimeout(() => {
      setSteps(stepsTarget.slice(0, steps.length + 1));
    }, 120);
    return () => clearTimeout(timer);
  }, [steps.length, stepsTarget]);

  // Load patients on mount
  useEffect(() => {
    listDemoPatients().then(setPatients).catch(() => {});
  }, []);

  // Select patient
  const handleSelectPatient = useCallback(async (p: DemoPatient) => {
    setSelectedPatient(p);
    setResult(null);
    setError(null);
    setSteps([]);
    setStepsTarget([]);
    setProgress(0);
    setDictationAudio(null);
    setDictationTranscript(null);
    setFhirContext(null);
    try {
      const imgs = await getDemoPatientImages(p.patient_id);
      setPriorImage(imgs.prior_image);
      setCurrentImage(imgs.current_image);
    } catch (e) {
      setError(`Failed to load images: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  // Upload handler
  const handleUpload = useCallback((prior: string, current: string) => {
    setSelectedPatient(null);
    setPriorImage(prior);
    setCurrentImage(current);
    setResult(null);
    setError(null);
    setSteps([]);
    setStepsTarget([]);
    setProgress(0);
    setDictationAudio(null);
    setDictationTranscript(null);
    setFhirContext(null);
  }, []);

  // Poll fallback
  const startPolling = useCallback((id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s: AnalysisStatus = await getAnalysisStatus(id);
        setStepsTarget(s.steps);
        setProgress(s.progress_percent);
        if (s.status === "complete") {
          clearInterval(pollRef.current!);
          const r = await getAnalysisResult(id);
          setResult(r);
          setAnalyzing(false);
          setTab("findings");
        } else if (s.status === "error") {
          clearInterval(pollRef.current!);
          setError(s.error_message ?? "Analysis failed");
          setAnalyzing(false);
        }
      } catch { /* keep polling */ }
    }, 500);
  }, []);

  // Analyze
  const handleAnalyze = useCallback(async () => {
    if (!priorImage || !currentImage) return;
    setAnalyzing(true);
    setResult(null);
    setError(null);
    setSteps([]);
    setStepsTarget([]);
    setProgress(0);

    try {
      const status = await startAnalysis({
        prior_image: priorImage,
        current_image: currentImage,
        patient_id: selectedPatient?.patient_id ?? "uploaded",
        dictation_audio: dictationAudio,
        fhir_context: fhirContext,
      });

      try {
        const ws = createAnalysisWebSocket(status.job_id);
        wsRef.current = ws;
        ws.onmessage = async (ev) => {
          const d: AnalysisStatus = JSON.parse(ev.data);
          setStepsTarget(d.steps);
          setProgress(d.progress_percent);
          if (d.status === "complete") {
            const r = await getAnalysisResult(d.job_id);
            setResult(r);
            setAnalyzing(false);
            setTab("findings");
            ws.close();
          } else if (d.status === "error") {
            setError(d.error_message ?? "Analysis failed");
            setAnalyzing(false);
            ws.close();
          }
        };
        ws.onerror = () => { ws.close(); startPolling(status.job_id); };
      } catch {
        startPolling(status.job_id);
      }
    } catch (e) {
      setError(`Analysis failed: ${e instanceof Error ? e.message : String(e)}`);
      setAnalyzing(false);
    }
  }, [priorImage, currentImage, selectedPatient, dictationAudio, fhirContext, startPolling]);

  // Reset
  const handleReset = useCallback(() => {
    wsRef.current?.close();
    if (pollRef.current) clearInterval(pollRef.current);
    setSelectedPatient(null);
    setPriorImage(null);
    setCurrentImage(null);
    setResult(null);
    setError(null);
    setAnalyzing(false);
    setSteps([]);
    setStepsTarget([]);
    setProgress(0);
    setHighlighted(null);
    setTab("findings");
    setDictationAudio(null);
    setDictationTranscript(null);
    setFhirContext(null);
  }, []);

  useEffect(() => () => {
    wsRef.current?.close();
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  const canAnalyze = !!(priorImage && currentImage && !analyzing);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      <Header />

      <ControlBar
        patients={patients}
        selectedPatient={selectedPatient}
        onSelectPatient={handleSelectPatient}
        canAnalyze={canAnalyze}
        analyzing={analyzing}
        onAnalyze={handleAnalyze}
        onReset={handleReset}
        hasResult={!!result || !!error}
        onUploadClick={() => setShowUpload(true)}
        fhirContext={fhirContext}
        onLoadFhir={setFhirContext}
        dictationAudio={dictationAudio}
        onTranscript={(text, audio) => {
          setDictationTranscript(text);
          setDictationAudio(audio);
        }}
      />

      {/* Dictation transcript banner */}
      {dictationTranscript && (
        <div className="flex items-start gap-3 px-5 py-2.5 bg-violet-50 border-b border-violet-200 shrink-0">
          <svg
            width="14"
            height="14"
            viewBox="0 0 16 16"
            fill="none"
            className="mt-0.5 text-violet-500 shrink-0"
          >
            <path
              d="M8 1a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0v-4A2.5 2.5 0 0 0 8 1Z"
              fill="currentColor"
            />
            <path
              d="M3.5 6.5a.5.5 0 0 1 1 0v1a3.5 3.5 0 1 0 7 0v-1a.5.5 0 0 1 1 0v1a4.5 4.5 0 0 1-4 4.473V13.5h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-1.527a4.5 4.5 0 0 1-4-4.473v-1Z"
              fill="currentColor"
            />
          </svg>
          <p className="flex-1 text-[13px] text-violet-800 leading-relaxed">
            <span className="font-medium">Dictation:</span>{" "}
            {dictationTranscript}
          </p>
          <button
            onClick={() => setDictationTranscript(null)}
            className="text-violet-400 hover:text-violet-600 transition-colors mt-0.5 shrink-0"
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path
                d="M4.5 4.5l7 7m0-7l-7 7"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Main content: images left, results right */}
      <div className="flex-1 flex min-h-0">
        {/* Left: Image viewer */}
        <div className="flex-1 min-w-0 min-h-0 p-2 flex flex-col">
          <ImageViewer
            priorImage={priorImage}
            currentImage={currentImage}
            priorDate={selectedPatient?.prior.study_index}
            currentDate={selectedPatient?.current.study_index}
            priorFinding={selectedPatient?.prior.finding}
            currentFinding={selectedPatient?.current.finding}
            findings={result?.findings ?? []}
            highlighted={highlighted}
            analyzing={analyzing}
          />
        </div>

        {/* Right: Results panel */}
        <div className="w-[420px] shrink-0 flex flex-col border-l border-slate-200 bg-white">
          {/* Workflow timeline */}
          <WorkflowTimeline steps={steps} analyzing={analyzing} />

          {/* Tabs */}
          <div className="flex items-center border-b border-slate-200 shrink-0">
            {([
              { key: "findings" as Tab, label: "Findings" },
              { key: "report" as Tab, label: "Report" },
              { key: "audit" as Tab, label: "Evidence" },
            ]).map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`flex-1 px-3 py-2.5 text-[13px] font-medium transition-colors relative ${
                  tab === t.key
                    ? "text-slate-800"
                    : "text-slate-400 hover:text-slate-600"
                }`}
              >
                {t.label}
                {t.key === "findings" && result && (
                  <span className="ml-1 text-[12px] text-slate-400">({result.findings.length})</span>
                )}
                {tab === t.key && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                )}
              </button>
            ))}
          </div>

          {error && (
            <div className="px-4 py-2 text-[13px] text-red-500 bg-red-50 border-b border-red-100">
              {error}
            </div>
          )}

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            {tab === "findings" && (
              <FindingsPanel
                findings={result?.findings ?? []}
                onHighlight={setHighlighted}
                highlighted={highlighted}
              />
            )}
            {tab === "report" && <ReportPanel result={result} dictationTranscript={dictationTranscript} />}
            {tab === "audit" && (
              <AuditPanel
                auditTrail={result?.audit_trail ?? []}
                retrievalEvidence={result?.retrieval_evidence ?? null}
              />
            )}
          </div>

          {/* Persistent disclaimer */}
          {result && (
            <div className="shrink-0 px-4 py-2.5 bg-amber-50 border-t border-amber-200">
              <p className="text-[12px] text-amber-800 leading-relaxed">
                {result.disclaimer}
              </p>
            </div>
          )}
        </div>
      </div>

      <UploadDialog
        open={showUpload}
        onClose={() => setShowUpload(false)}
        onUpload={handleUpload}
      />
    </div>
  );
}
