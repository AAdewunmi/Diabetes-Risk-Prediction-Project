"""
Dashboard package exports.

Re-export selected symbols for convenient imports like:
    from dashboard import ModelWrapper, find_model, REPORTS_EXPLAIN_DIR
"""

from .predict import (
    REPORTS_EXPLAIN_DIR as REPORTS_EXPLAIN_DIR,  # explicit re-exports to satisfy Ruff F401
)
from .predict import ModelWrapper as ModelWrapper
from .predict import find_model as find_model

__all__ = ["REPORTS_EXPLAIN_DIR", "ModelWrapper", "find_model"]
