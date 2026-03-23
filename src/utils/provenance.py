"""
src/utils/provenance.py

Backward-compatible shim. Canonical implementation is at src/provenance/data_auditor.py
"""
from src.provenance.data_auditor import DataAuditor  # noqa: F401

__all__ = ["DataAuditor"]
 