/** TypeScript types matching backend Pydantic schemas. */

export type FindingLabel =
  | "consolidation"
  | "effusion"
  | "pneumothorax"
  | "cardiomegaly"
  | "edema"
  | "nodule"
  | "atelectasis"
  | "normal";

export type DeltaStatus = "improved" | "stable" | "worsened" | "uncertain";

export type UncertaintyLevel = "low" | "medium" | "high";

export type EvidenceSourceType =
  | "prior_image"
  | "current_image"
  | "fhir_context"
  | "guideline"
  | "dictation"
  | "similar_case"
  | "zero_shot";

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Finding {
  label: FindingLabel;
  delta: DeltaStatus;
  rationale: string;
  evidence_refs: EvidenceSourceType[];
  bounding_box: BoundingBox | null;
  uncertainty: UncertaintyLevel;
}

export interface DraftReport {
  findings: string;
  impression: string;
}

export interface AuditEntry {
  claim: string;
  source_type: EvidenceSourceType;
  source_ref: string;
  uncertainty: UncertaintyLevel;
}

export type WorkflowStep =
  | "intake"
  | "preprocess"
  | "retrieve_context"
  | "retrieve_evidence"
  | "inference"
  | "validate"
  | "assemble";

export type StepStatus = "pending" | "running" | "complete" | "error";

export interface WorkflowStepUpdate {
  step: WorkflowStep;
  status: StepStatus;
  timestamp: string;
  message: string | null;
  duration_ms: number | null;
}

export interface AnalysisRequest {
  prior_image: string;
  current_image: string;
  patient_id?: string;
  dictation_audio?: string | null;
  fhir_context?: Record<string, unknown> | null;
}

export interface AnalysisResult {
  job_id: string;
  patient_id: string;
  timestamp: string;
  findings: Finding[];
  delta_summary: string;
  draft_report: DraftReport;
  audit_trail: AuditEntry[];
  retrieval_evidence: RetrievalEvidenceResult | null;
  model_narrative: string | null;
  disclaimer: string;
}

export interface AnalysisStatus {
  job_id: string;
  status: "pending" | "processing" | "complete" | "error";
  current_step: WorkflowStep | null;
  steps: WorkflowStepUpdate[];
  progress_percent: number;
  error_message: string | null;
}

export interface HealthResponse {
  status: "healthy" | "unhealthy";
  version: string;
  model_loaded: boolean;
  device: string;
}

export interface SimilarCaseResult {
  case_id: string;
  similarity_score: number;
  known_findings: string[];
  delta_label: string;
  source: string;
  description: string | null;
}

export interface ZeroShotResult {
  label: string;
  confidence: number;
}

export interface GuidelineMatchResult {
  title: string;
  source: string;
  text: string;
  similarity_score: number;
  finding_type: string;
  citation: string;
}

export interface RetrievalEvidenceResult {
  similar_cases: SimilarCaseResult[];
  zero_shot_predictions: ZeroShotResult[];
  guideline_matches: GuidelineMatchResult[];
  case_retrieval_confidence: number;
  overall_confidence: number;
  evidence_summary: string;
  classification_agreement: string | null;
}

export interface DemoPatientImageInfo {
  filename: string;
  finding: string;
  study_index?: string;
}

export interface DemoPatient {
  patient_id: string;
  description: string;
  prior: DemoPatientImageInfo;
  current: DemoPatientImageInfo;
  expected_delta: string;
  note?: string;
}

export interface DemoPatientImages {
  patient_id: string;
  prior_image: string;
  current_image: string;
}

export interface TranscriptionTimestamp {
  start: number;
  end: number;
  text: string;
}

export interface TranscriptionResponse {
  text: string;
  confidence: number;
  timestamps: TranscriptionTimestamp[];
  source: string;
}
