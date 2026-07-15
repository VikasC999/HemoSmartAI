"""
Base Adapter Interface
------------------------
Every format adapter (PDF, manual entry, CSV, future HL7/FHIR) implements
this same interface: some format-specific input in, one or more
PatientRecord objects out.

Keeping a common interface means the backend can call any adapter the
same way, without knowing or caring which hospital format is behind it.
"""

from abc import ABC, abstractmethod
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema.patient_schema import PatientRecord


class BaseAdapter(ABC):
    """All adapters must implement `parse()` and return PatientRecord(s)."""

    source_format: str = "unknown"

    @abstractmethod
    def parse(self, raw_input) -> List[PatientRecord]:
        """Convert raw hospital-format input into a list of PatientRecords."""
        raise NotImplementedError

    def safe_parse(self, raw_input) -> List[PatientRecord]:
        """
        Wraps parse() so a malformed record from one hospital's export
        never crashes the whole batch -- it's skipped and logged instead.
        This matters because hospital data is messy in practice: missing
        fields, out-of-range OCR misreads, inconsistent surgery-type
        naming, etc.
        """
        try:
            return self.parse(raw_input)
        except Exception as exc:
            print(f"[{self.source_format} adapter] Failed to parse record: {exc}")
            return []