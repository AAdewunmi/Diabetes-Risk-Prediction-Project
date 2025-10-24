# data_exploration.py
"""
Exploratory Data Analysis utilities for the Diabetes Risk Prediction Project.

Functions:
- explore_data: high-level orchestrator that writes summary CSVs and basic visualizations.
- generate_basic_summary: returns data types, non-null counts, and descriptive stats.
- missing_value_report: returns missingness per column.
- detect_time_columns: heuristically finds possible date/time columns for time-series analysis.

Saves summary artifacts under the `reports/` directory by default.
"""

from typing import Optional, List, Tuple
import os
import logging
import pandas as pd
import numpy as np

from data_visualisation import (
    plot_outcome_distribution,
    plot_correlation_heatmap
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_reports_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a table with dtypes, non-null counts and percent missing for each column.

    Returns:
        pandas.DataFrame: summary table
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.notnull().sum(),
        "missing_count": df.isnull().sum(),
        "pct_missing": (df.isnull().mean() * 100).round(2)
    })
    return summary


def missing_value_report(df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute and optionally save missingness report.

    Args:
        df: input DataFrame
        output_path: where to save CSV (optional)

    Returns:
        pd.DataFrame: missingness report
    """
    report = generate_basic_summary(df)
    if output_path:
        report.to_csv(output_path)
        logging.info("Saved missing value report to %s", output_path)
    return report


def detect_time_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically detect columns that look like datetime columns.

    Looks for common names and attempts to parse dtype.

    Returns:
        list of candidate datetime column names (may be empty)
    """
    possible_names = {"date", "datetime", "admission_date", "discharge_date", "visit_date", "timestamp"}
    found = [c for c in df.columns if c.lower() in possible_names]
    # Also detect columns that parse to datetime
    for c in df.columns:
        if c in found:
            continue
        if df[c].dtype == object:
            try:
                _ = pd.to_datetime(df[c], errors="coerce")
                if _.notna().sum() / max(1, len(df)) > 0.5:
                    found.append(c)
            except Exception:
                pass
    return found


def explore_data(df: pd.DataFrame, output_dir: str = "reports") -> None:
    """
    High-level EDA orchestration. Produces:
      - basic info printed
      - saved CSVs for summary and missingness
      - some default visualizations saved under reports/

    Args:
        df: DataFrame to analyze
        output_dir: directory where CSVs/plots are saved
    """
    ensure_reports_dir(output_dir)

    if df is None or df.shape[0] == 0:
        logging.error("No data provided to explore_data.")
        return

    # 1. Basic info & summary
    logging.info("Data shape: %s", df.shape)
    summary = generate_basic_summary(df)
    summary_csv = os.path.join(output_dir, "eda_summary.csv")
    summary.to_csv(summary_csv)
    logging.info("Saved dataset summary to %s", summary_csv)

    # 2. Missingness
    missing_csv = os.path.join(output_dir, "eda_missingness.csv")
    missing_value_report(df, missing_csv)

    # 3. Target distribution (if 'Outcome' or 'readmitted' present)
    target_cols = [c for c in ("Outcome", "readmitted") if c in df.columns]
    if target_cols:
        target = target_cols[0]
        plot_outcome_distribution(df, target=target, output_path=os.path.join(output_dir, "outcome_distribution.png"))
    else:
        logging.info("No target column ('Outcome' or 'readmitted') found; skipping outcome distribution plot.")

    # 4. Correlation heatmap of numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr_path = os.path.join(output_dir, "correlation_heatmap.png")
        plot_correlation_heatmap(df[num_cols], output_path=corr_path)
    else:
        logging.info("Not enough numeric columns for correlation heatmap.")

    # 5. Time-series detection
    time_cols = detect_time_columns(df)
    if time_cols:
        logging.info("Detected possible time columns: %s", time_cols)
        # Save a small sample with parsed datetime for follow-up analysis
        for c in time_cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            sample = df[[c]].copy()
            sample[c] = parsed
            sample_csv = os.path.join(output_dir, f"time_column_sample_{c}.csv")
            sample.head(200).to_csv(sample_csv, index=False)
            logging.info("Saved sample parsed dates for %s to %s", c, sample_csv)
    else:
        logging.info("No obvious time-series columns detected.")


if __name__ == "__main__":
    # Example usage: python data_exploration.py
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA for diabetes dataset and save reports.")
    parser.add_argument("--data", type=str, default="./data/diabetes.csv", help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default="reports", help="Output reports directory")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logging.error("File not found: %s", args.data)
        raise

    explore_data(df, output_dir=args.out)