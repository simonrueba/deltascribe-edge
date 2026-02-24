"""FHIR R4 bundle builder for interoperability export."""

from typing import Any
from uuid import uuid4

from app.schemas.analysis import AnalysisResult, DeltaStatus, Finding, FindingLabel


def build_fhir_bundle(result: AnalysisResult) -> dict[str, Any]:
    """
    Build a FHIR R4 Bundle containing DiagnosticReport and Observations.

    Args:
        result: Complete analysis result

    Returns:
        FHIR R4 Bundle as dict (JSON-serializable)
    """
    bundle_id = str(uuid4())
    timestamp = result.timestamp.isoformat()

    # Create DiagnosticReport
    report_id = str(uuid4())
    diagnostic_report = _build_diagnostic_report(result, report_id, timestamp)

    # Create Observations for each finding
    observations = []
    observation_refs = []
    for finding in result.findings:
        obs_id = str(uuid4())
        observation = _build_observation(finding, obs_id, timestamp, result.patient_id)
        observations.append(observation)
        observation_refs.append({"reference": f"Observation/{obs_id}"})

    # Add observation references to report
    diagnostic_report["result"] = observation_refs

    # Build bundle
    bundle = {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": timestamp,
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"],
            "tag": [
                {
                    "system": "http://deltascribe.ai/tags",
                    "code": "ai-generated",
                    "display": "AI-Generated Draft",
                }
            ],
        },
        "entry": [
            {"fullUrl": f"urn:uuid:{report_id}", "resource": diagnostic_report},
            *[{"fullUrl": f"urn:uuid:{obs['id']}", "resource": obs} for obs in observations],
        ],
    }

    return bundle


def _build_diagnostic_report(
    result: AnalysisResult, report_id: str, timestamp: str
) -> dict[str, Any]:
    """Build FHIR DiagnosticReport resource."""
    return {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "meta": {
            "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note"],
        },
        "status": "preliminary",  # Always preliminary - requires clinician review
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "36643-5",
                    "display": "XR Chest 2 Views",
                }
            ],
            "text": "Chest X-Ray Comparison",
        },
        "subject": {
            "reference": f"Patient/{result.patient_id}",
            "display": result.patient_id,
        },
        "effectiveDateTime": timestamp,
        "issued": timestamp,
        "conclusion": result.draft_report.impression,
        "conclusionCode": _get_conclusion_codes(result.findings),
        "presentedForm": [
            {
                "contentType": "text/plain",
                "data": _encode_report_text(result),
            }
        ],
        "extension": [
            {
                "url": "http://deltascribe.ai/fhir/StructureDefinition/ai-disclaimer",
                "valueString": result.disclaimer,
            },
            {
                "url": "http://deltascribe.ai/fhir/StructureDefinition/delta-summary",
                "valueString": result.delta_summary,
            },
        ],
    }


def _build_observation(
    finding: Finding, obs_id: str, timestamp: str, patient_id: str
) -> dict[str, Any]:
    """Build FHIR Observation resource for a finding."""
    # Map finding label to SNOMED code
    snomed_code = _get_snomed_code(finding.label)

    # Map delta to interpretation code
    interpretation = _get_interpretation_code(finding.delta)

    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [snomed_code],
            "text": finding.label.value.title(),
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "effectiveDateTime": timestamp,
        "interpretation": [interpretation],
        "note": [
            {
                "text": finding.rationale,
            }
        ],
        "extension": [
            {
                "url": "http://deltascribe.ai/fhir/StructureDefinition/delta-status",
                "valueCode": finding.delta.value,
            },
            {
                "url": "http://deltascribe.ai/fhir/StructureDefinition/uncertainty-level",
                "valueCode": finding.uncertainty.value,
            },
        ],
    }


def _get_snomed_code(label: FindingLabel) -> dict[str, str]:
    """Get SNOMED CT code for a finding label."""
    snomed_map = {
        FindingLabel.CONSOLIDATION: {
            "system": "http://snomed.info/sct",
            "code": "233709007",
            "display": "Pulmonary consolidation",
        },
        FindingLabel.EFFUSION: {
            "system": "http://snomed.info/sct",
            "code": "60046008",
            "display": "Pleural effusion",
        },
        FindingLabel.PNEUMOTHORAX: {
            "system": "http://snomed.info/sct",
            "code": "36118008",
            "display": "Pneumothorax",
        },
        FindingLabel.CARDIOMEGALY: {
            "system": "http://snomed.info/sct",
            "code": "8186001",
            "display": "Cardiomegaly",
        },
        FindingLabel.EDEMA: {
            "system": "http://snomed.info/sct",
            "code": "19242006",
            "display": "Pulmonary edema",
        },
        FindingLabel.NODULE: {
            "system": "http://snomed.info/sct",
            "code": "427359005",
            "display": "Pulmonary nodule",
        },
        FindingLabel.ATELECTASIS: {
            "system": "http://snomed.info/sct",
            "code": "46621007",
            "display": "Atelectasis",
        },
        FindingLabel.NORMAL: {
            "system": "http://snomed.info/sct",
            "code": "17621005",
            "display": "Normal",
        },
    }
    return snomed_map.get(label, snomed_map[FindingLabel.NORMAL])


def _get_interpretation_code(delta: DeltaStatus) -> dict[str, Any]:
    """Get FHIR interpretation code for delta status."""
    interpretation_map = {
        DeltaStatus.IMPROVED: {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "D",
                    "display": "Decreased",
                }
            ],
            "text": "Improved from prior",
        },
        DeltaStatus.STABLE: {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "N",
                    "display": "Normal",
                }
            ],
            "text": "Stable/Unchanged",
        },
        DeltaStatus.WORSENED: {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "U",
                    "display": "Increased",
                }
            ],
            "text": "Worsened from prior",
        },
        DeltaStatus.UNCERTAIN: {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "IND",
                    "display": "Indeterminate",
                }
            ],
            "text": "Comparison uncertain",
        },
    }
    return interpretation_map.get(delta, interpretation_map[DeltaStatus.UNCERTAIN])


def _get_conclusion_codes(findings: list[Finding]) -> list[dict[str, Any]]:
    """Get conclusion codes from findings."""
    codes = []
    for finding in findings:
        if finding.label != FindingLabel.NORMAL:
            snomed = _get_snomed_code(finding.label)
            codes.append({"coding": [snomed]})
    return codes


def _encode_report_text(result: AnalysisResult) -> str:
    """Encode report text as base64 for presentedForm."""
    import base64

    report_text = f"""CHEST X-RAY COMPARISON REPORT (AI-GENERATED DRAFT)

{result.draft_report.findings}

IMPRESSION:
{result.draft_report.impression}

DELTA SUMMARY:
{result.delta_summary}

---
{result.disclaimer}
"""
    return base64.b64encode(report_text.encode()).decode()


def extract_minimal_context(fhir_bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Extract minimal context from a FHIR bundle for inference.

    Only extracts:
    - Active problems/conditions
    - Current medications
    - Prior report impressions
    """
    context: dict[str, Any] = {
        "problems": [],
        "medications": [],
        "prior_impression": None,
    }

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Condition":
            # Extract condition name
            code = resource.get("code", {})
            text = code.get("text") or code.get("coding", [{}])[0].get("display", "")
            clinical_code = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")
            if text and clinical_code == "active":
                context["problems"].append(text)

        elif resource_type == "MedicationStatement":
            # Extract medication name
            med = resource.get("medicationCodeableConcept", {})
            text = med.get("text") or med.get("coding", [{}])[0].get("display", "")
            if text and resource.get("status") == "active":
                context["medications"].append(text)

        elif resource_type == "DiagnosticReport":
            # Extract prior impression
            conclusion = resource.get("conclusion")
            if conclusion:
                context["prior_impression"] = conclusion

    return context
