"""
src/dashboard/utils.py

Utility helpers for dashboard prediction endpoints.

Functions:
- safe_prepare_df(df, expected_features, fill_method='median'):
    Ensure the dataframe contains exactly expected_features in that order.
    Drops unexpected columns, fills missing columns with median from provided df
    (or zeros if df empty), coerces numeric types.

- artifact_name(prefix, ext): returns a timestamped artifact filename.

- list_explain_files(dirpath, extensions): list files with mtimes (for /api/explain_files)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import pandas as pd


def safe_prepare_df(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Given an input DataFrame and expected feature list:
      - Drop any 'Outcome' column if present (common uploaded mistake).
      - Keep only expected features; if some are missing, create them filled with median of df
        when possible, otherwise with zero.
      - Coerce to numeric where possible.

    Returns a new DataFrame with columns in expected_features order.
    """
    df_copy = df.copy()
    if "Outcome" in df_copy.columns:
        df_copy = df_copy.drop(columns=["Outcome"])

    # coerce numeric-ish columns
    for col in df_copy.columns:
        try:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        except Exception:
            pass

    # prepare missing features
    missing = [c for c in expected_features if c not in df_copy.columns]
    for c in missing:
        # attempt median from df if any numeric columns exist
        try:
            med = df_copy.median(numeric_only=True).median()
            fill_val = float(med) if pd.notna(med) else 0.0
        except Exception:
            fill_val = 0.0
        df_copy[c] = fill_val

    # drop extras not expected
    extras = [c for c in df_copy.columns if c not in expected_features]
    if extras:
        df_copy = df_copy.drop(columns=extras)

    # reorder
    df_copy = df_copy[expected_features]

    # final numeric coercion and fill NaN with medians/0
    for col in df_copy.columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        if df_copy[col].isna().any():
            # fill with median of that column or 0
            try:
                m = df_copy[col].median()
                fill = 0.0 if pd.isna(m) else float(m)
            except Exception:
                fill = 0.0
            df_copy[col] = df_copy[col].fillna(fill)

    return df_copy


def artifact_name(prefix: str, ext: str) -> str:
    """Return a timestamped filename: <prefix>_<ts>.<ext>"""
    ts = int(time.time())
    return f"{prefix}_{ts}.{ext}"


def list_files_with_mtime(dirpath: str, extensions=None) -> List[Dict]:
    """Return list of files in dirpath filtered by extensions with their mtimes (int)."""
    if not os.path.isdir(dirpath):
        return []
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".html", ".csv"]
    out = []
    for name in os.listdir(dirpath):
        if not any(name.lower().endswith(ext) for ext in extensions):
            continue
        try:
            p = os.path.join(dirpath, name)
            m = int(os.path.getmtime(p))
        except Exception:
            m = 0
        out.append({"filename": name, "mtime": m})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out
