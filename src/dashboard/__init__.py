"""
Dashboard package initializer.

Re-exports commonly-used items so tests can do:
    from dashboard import ModelWrapper, find_model
"""

from .predict import REPORTS_EXPLAIN_DIR, ModelWrapper, find_model  # noqa: F401
