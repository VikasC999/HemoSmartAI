"""
HemoSmart - Canonical Patient Schema
----------------------------------------
This is THE single source of truth for what a "patient record" looks like
inside HemoSmart, regardless of which hospital or format it came from.

Every input format (PDF report, manual entry, CSV export, future HL7/FHIR
feed) must be converted into this schema by an adapter BEFORE it reaches
the ML model, the RAG engine, or the backend API.

    Hospital A (HL7)  ---\
    Hospital B (PDF)  ----\
    Hospital C (CSV)  -----> adapters/*.py --> PatientRecord --> XGBoost model
    Hospital D (API)  ----/
    Manual entry form -/

Nothing downstream should ever read a raw hospital format directly.
If you need a new hospital format supported, write ONE new adapter that
outputs a PatientRecord -- nothing else in the system needs to change.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


VALID_SURGERY_TYPES = ("Cardiac", "Orthopedic", "General", "Emergency")


class PatientRecord(BaseModel):
    """
    The canonical, validated shape of a single patient's data.
    All fields are required and range-checked against real physiological
    limits, so a bad adapter (or bad OCR read) fails loudly here instead
    of silently poisoning a prediction later.
    """

    hemoglobin: float = Field(..., ge=2.0, le=22.0,
                               description="g/dL")
    platelets: int = Field(..., ge=1000, le=1_000_000,
                            description="per microliter")
    INR: float = Field(..., ge=0.5, le=10.0,
                        description="International Normalized Ratio")
    age: int = Field(..., ge=0, le=120)
    surgery_type: str

    # Optional metadata -- not used by the model, but useful for
    # traceability (which hospital/format this record came from).
    source_format: str = Field(default="unknown",
                                description="e.g. 'pdf', 'manual', 'csv'")
    source_hospital: str = Field(default="unknown")

    @field_validator("surgery_type")
    @classmethod
    def check_surgery_type(cls, v: str) -> str:
        # Normalize case/whitespace differences between hospitals
        # ("cardiac", " Cardiac ", "CARDIAC" all mean the same thing)
        cleaned = v.strip().title()
        if cleaned not in VALID_SURGERY_TYPES:
            raise ValueError(
                f"surgery_type must be one of {VALID_SURGERY_TYPES}, "
                f"got '{v}'. Map unrecognized hospital-specific surgery "
                f"codes to the nearest category inside the adapter."
            )
        return cleaned

    def to_model_features(self, label_encoder) -> list:
        """
        Converts this record into the exact ordered feature list the
        trained XGBoost model expects. Keeping this conversion in ONE
        place (not duplicated in the backend, the agents, etc.) avoids
        feature-order bugs.
        """
        return [
            self.hemoglobin,
            self.platelets,
            self.INR,
            self.age,
            int(label_encoder.transform([self.surgery_type])[0]),
        ]


class PredictionResult(BaseModel):
    """
    The canonical shape of what the ML model + RAG engine hand back
    to the backend / frontend. Person C's FastAPI layer and Person B's
    RAG output both target this shape.
    """
    transfusion_needed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = ""
    source_format: str = "unknown"