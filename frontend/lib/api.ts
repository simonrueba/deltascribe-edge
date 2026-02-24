/** API client for DeltaScribe Edge backend. */

import type {
  AnalysisRequest,
  AnalysisResult,
  AnalysisStatus,
  DemoPatient,
  DemoPatientImages,
  HealthResponse,
  TranscriptionResponse,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API = `${API_BASE}/api`;

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function healthCheck(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>(`${API}/health`);
}

export async function startAnalysis(
  request: AnalysisRequest,
): Promise<AnalysisStatus> {
  return fetchJSON<AnalysisStatus>(`${API}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function getAnalysisStatus(
  jobId: string,
): Promise<AnalysisStatus> {
  return fetchJSON<AnalysisStatus>(`${API}/analyze/${jobId}/status`);
}

export async function getAnalysisResult(
  jobId: string,
): Promise<AnalysisResult> {
  return fetchJSON<AnalysisResult>(`${API}/analyze/${jobId}/result`);
}

export async function exportFHIR(jobId: string): Promise<Blob> {
  const res = await fetch(`${API}/export/fhir/${jobId}`);
  if (!res.ok) throw new Error(`FHIR export failed: ${res.statusText}`);
  return res.blob();
}

export async function listDemoPatients(): Promise<DemoPatient[]> {
  return fetchJSON<DemoPatient[]>(`${API}/demo/patients`);
}

export async function getDemoPatientImages(
  patientId: string,
): Promise<DemoPatientImages> {
  return fetchJSON<DemoPatientImages>(
    `${API}/demo/patients/${patientId}/images`,
  );
}

export function createAnalysisWebSocket(jobId: string): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/api/ws/analyze/${jobId}`);
}

export async function transcribeAudio(
  audioBase64: string,
): Promise<TranscriptionResponse> {
  return fetchJSON<TranscriptionResponse>(`${API}/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio: audioBase64 }),
  });
}

