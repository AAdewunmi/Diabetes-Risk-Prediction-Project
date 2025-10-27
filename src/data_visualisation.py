# data_visualisation.py
"""
Visualization utilities used by EDA and reporting.

Functions:
 - plot_outcome_distribution
 - plot_correlation_heatmap
 - plot_pairplot_sample
 - plot_violin
 - plot_time_series_if_present
Each function saves figures to disk and returns the matplotlib Figure object.

Note: pairplots or very large plots sample the data to avoid memory/time issues.
"""

import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure seaborn aesthetics (do not set global styles that override user's settings)
sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_outcome_distribution(
    df: pd.DataFrame,
    target: str = "Outcome",
    output_path: str = "reports/outcome_distribution.png",
) -> plt.Figure:
    """
    Plot and save the count distribution of a binary/multi-class target column.
    """
    ensure_dir(output_path)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=target, data=df)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved outcome distribution to %s", output_path)
    return plt.gcf()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str = "reports/correlation_heatmap.png",
    annot: bool = True,
) -> plt.Figure:
    """
    Plot and save correlation heatmap for numeric DataFrame df.
    """
    ensure_dir(output_path)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr, annot=annot, cmap="coolwarm", fmt=".2f")
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved correlation heatmap to %s", output_path)
    return plt.gcf()


def plot_pairplot_sample(
    df: pd.DataFrame,
    cols: List[str],
    hue: Optional[str] = None,
    sample_n: int = 1000,
    output_path: str = "reports/pairplot.png",
) -> None:
    """
    Create a seaborn pairplot on a sampled subset (to reduce memory/time).
    """
    ensure_dir(output_path)
    if len(df) > sample_n:
        data = df[cols].sample(n=sample_n, random_state=42)
    else:
        data = df[cols]
    try:
        sns.pairplot(data, hue=hue, diag_kind="kde", corner=True)
        plt.savefig(output_path)
        plt.close()
        logging.info("Saved pairplot to %s", output_path)
    except Exception as e:
        logging.exception("Pairplot failed: %s", e)


def plot_violin(
    df: pd.DataFrame, x: str, y: str, output_path: str = "reports/violin.png"
) -> plt.Figure:
    """
    Violin plot (distribution) of y by categories in x.
    """
    ensure_dir(output_path)
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=x, y=y, data=df)
    plt.title(f"{y} distribution by {x}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved violin plot to %s", output_path)
    return plt.gcf()


def plot_time_series_if_present(
    df: pd.DataFrame,
    date_col_candidates: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    output_path: str = "reports/time_series.png",
) -> None:
    """
    If a date-like column exists, aggregate and plot time-series trends for selected numeric columns.

    Args:
      df: input dataframe
      date_col_candidates: list of possible date column names to check (if None, common names will be checked)
      value_cols: list of numeric columns to plot (if None, takes first 3 numerics)
    """
    ensure_dir(output_path)
    # heuristics for date columns:
    candidates = date_col_candidates or [
        "date",
        "datetime",
        "admission_date",
        "visit_date",
        "timestamp",
    ]
    date_col = None
    for c in candidates:
        if c in df.columns:
            # try parse
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    date_col = c
                    df = df.copy()
                    df["_parsed_date"] = parsed
                    break
            except Exception:
                continue

    if date_col is None:
        logging.info("No parseable date column found; skipping time-series plot.")
        return

    numerics = value_cols or df.select_dtypes(include="number").columns.tolist()
    numerics = [c for c in numerics if c != "_parsed_date"]
    if not numerics:
        logging.info("No numeric columns to plot in time-series.")
        return

    # choose up to 3 columns to avoid clutter
    plot_cols = numerics[:3]
    ts = (
        df.dropna(subset=["_parsed_date"] + plot_cols)
        .set_index("_parsed_date")
        .sort_index()
    )
    if ts.empty:
        logging.info("No rows after parsing dates; skipping time-series.")
        return

    agg = ts[plot_cols].resample("M").mean()  # monthly mean
    plt.figure(figsize=(10, 6))
    for col in plot_cols:
        plt.plot(agg.index, agg[col], label=col)
    plt.legend()
    plt.title("Monthly mean time-series of selected features")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved time-series plot to %s", output_path)


if __name__ == "__main__":
    # quick demo if run directly
    import argparse

    parser = argparse.ArgumentParser(
        description="Create visual reports for the diabetes dataset"
    )
    parser.add_argument(
        "--data", type=str, default="./data/diabetes.csv", help="Path to dataset CSV"
    )
    parser.add_argument(
        "--out", type=str, default="reports", help="Output directory for plots"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logging.error("File not found: %s", args.data)
        raise

    # Example usage of the plotting functions
    plot_outcome_distribution(
        df, target="Outcome", output_path=f"{args.out}/outcome_distribution.png"
    )
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) > 1:
        plot_correlation_heatmap(
            df[num_cols], output_path=f"{args.out}/correlation_heatmap.png"
        )
    # pairplot for a few columns (limit to safe number)
    pair_cols = [
        c
        for c in ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age", "Outcome"]
        if c in df.columns
    ]
    if pair_cols:
        plot_pairplot_sample(
            df,
            cols=pair_cols,
            hue="Outcome" if "Outcome" in df.columns else None,
            output_path=f"{args.out}/pairplot.png",
        )
    plot_violin(
        df,
        x="Outcome" if "Outcome" in df.columns else df.columns[0],
        y="BMI" if "BMI" in df.columns else num_cols[0],
        output_path=f"{args.out}/violin.png",
    )
    plot_time_series_if_present(df, output_path=f"{args.out}/time_series.png")
