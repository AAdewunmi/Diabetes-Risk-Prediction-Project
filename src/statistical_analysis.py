# statistical_analysis.py
"""
Advanced statistical analysis helpers.

Capabilities:
 - Two-sample t-test and Mann-Whitney U
 - Pearson/Spearman correlations
 - Chi-square test for independence
 - One-way ANOVA
 - Logistic regression (statsmodels) for variable association with outcome
 - Optional survival analysis (Kaplan-Meier, CoxPH) when 'time' & 'event' columns are present
   (requires the `lifelines` package; handled gracefully if missing)

Outputs results to console and returns result objects for programmatic use.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Optional import for survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def t_test_between_groups(
    df: pd.DataFrame, col: str, group_col: str = "Outcome"
) -> Dict[str, Any]:
    """
    Perform independent t-test between two groups identified by group_col (assumes binary).
    Returns test statistic and p-value.
    """
    groups = df.groupby(group_col)[col].apply(lambda s: s.dropna())
    if len(groups) < 2:
        logging.warning("Less than two groups present for t-test on %s", col)
        return {}
    a, b = groups.iloc[0], groups.iloc[1]
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    logging.info("t-test %s by %s: t=%.4f, p=%.4g", col, group_col, t_stat, p_val)
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


def mann_whitney_u(
    df: pd.DataFrame, col: str, group_col: str = "Outcome"
) -> Dict[str, Any]:
    """
    Non-parametric Mann-Whitney U test between two groups.
    """
    groups = df.groupby(group_col)[col].apply(lambda s: s.dropna())
    if len(groups) < 2:
        logging.warning("Less than two groups present for Mann-Whitney on %s", col)
        return {}
    a, b = groups.iloc[0], groups.iloc[1]
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    logging.info("Mann-Whitney %s by %s: U=%.4f, p=%.4g", col, group_col, u_stat, p_val)
    return {"u_stat": float(u_stat), "p_value": float(p_val)}


def pearson_correlation(df: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    """
    Pearson correlation between two numeric columns.
    """
    # a = df[x].dropna()
    # b = df[y].dropna()
    # Align lengths by dropping NA on both
    common = df[[x, y]].dropna()
    if common.shape[0] < 3:
        logging.warning(
            "Insufficient paired data for Pearson correlation between %s and %s", x, y
        )
        return {}
    corr, p_val = stats.pearsonr(common[x], common[y])
    logging.info("Pearson %s vs %s: r=%.4f, p=%.4g", x, y, corr, p_val)
    return {"r": float(corr), "p_value": float(p_val)}


def chi_square_test(df: pd.DataFrame, cat_x: str, cat_y: str) -> Dict[str, Any]:
    """
    Chi-square test of independence between two categorical variables.
    """
    contingency = pd.crosstab(df[cat_x], df[cat_y])
    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        logging.warning(
            "Insufficient data for chi-square between %s and %s", cat_x, cat_y
        )
        return {}
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    logging.info("Chi2 %s vs %s: chi2=%.4f, p=%.4g, dof=%s", cat_x, cat_y, chi2, p, dof)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected,
    }


def one_way_anova(df: pd.DataFrame, response: str, factor: str) -> Dict[str, Any]:
    """
    One-way ANOVA using statsmodels (response ~ C(factor)).
    Returns ANOVA table.
    """
    formula = f"{response} ~ C({factor})"
    try:
        model = smf.ols(formula, data=df.dropna(subset=[response, factor])).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        logging.info("ANOVA (%s by %s) completed.", response, factor)
        return {"anova_table": anova_table}
    except Exception as e:
        logging.exception("ANOVA failed: %s", e)
        return {}


def logistic_regression(df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    """
    Fit a logistic regression using statsmodels formula API.
    Example formula: 'Outcome ~ Age + BMI + Glucose'
    Returns fitted model summary and odds ratios.
    """
    try:
        model = smf.logit(formula, data=df.dropna()).fit(disp=False)
        summary = model.summary2().as_text()
        params = model.params
        conf = model.conf_int()
        odds_ratios = np.exp(params)
        logging.info("Logistic regression fitted using formula: %s", formula)
        return {
            "model": model,
            "summary": summary,
            "odds_ratios": odds_ratios,
            "conf_int": conf,
        }
    except Exception as e:
        logging.exception("Logistic regression failed for formula %s: %s", formula, e)
        return {}


def survival_analysis(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform Kaplan-Meier and Cox proportional hazards (if lifelines is installed).

    Arguments:
        df: dataframe containing columns time_col and event_col
        time_col: duration until event or censoring
        event_col: boolean/int indicating event occurrence (1) or censoring (0)
        output_dir: optional path to save plots (handled by caller)

    Returns:
        dict with Kaplan-Meier fitter and optionally CoxPH fitter objects
    """
    if not LIFELINES_AVAILABLE:
        logging.warning("lifelines package not installed: survival analysis skipped.")
        return {}

    if time_col not in df.columns or event_col not in df.columns:
        logging.warning("Time or event columns missing for survival analysis.")
        return {}

    results = {}
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df[time_col], event_observed=df[event_col])
        results["kmf"] = kmf
        logging.info(
            "Kaplan-Meier fit complete. Median survival: %s", kmf.median_survival_time_
        )

        # Optional CoxPH (requires some pre-processing for formula)
        cph = CoxPHFitter()
        # select numeric covariates
        numeric = df.select_dtypes(include=[np.number]).drop(
            columns=[time_col, event_col], errors="ignore"
        )
        # combine with time/event
        cox_df = pd.concat([numeric, df[[time_col, event_col]]], axis=1).dropna()
        if cox_df.shape[0] > 10 and cox_df.shape[1] > 2:
            cph.fit(
                cox_df, duration_col=time_col, event_col=event_col, show_progress=False
            )
            results["cph"] = cph
            logging.info("CoxPH model fitted.")
        else:
            logging.info(
                "Not enough data to fit CoxPH (need >10 rows and multiple covariates)."
            )
    except Exception as e:
        logging.exception("Error running survival analysis: %s", e)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run statistical tests on diabetes dataset"
    )
    parser.add_argument(
        "--data", type=str, default="./data/diabetes.csv", help="Path to CSV dataset"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        logging.error("File not found: %s", args.data)
        raise

    # Demonstrate a set of tests (safely guarded)
    if "Outcome" in df.columns and "Glucose" in df.columns:
        t_test_between_groups(df, "Glucose", "Outcome")
    if "Outcome" in df.columns and "Insulin" in df.columns:
        mann_whitney_u(df, "Insulin", "Outcome")
    if {"BMI", "Glucose"}.issubset(df.columns):
        pearson_correlation(df, "BMI", "Glucose")
    # Example ANOVA: glucose by binned age if age exists
    if "Age" in df.columns and "Glucose" in df.columns:
        # create age bins for ANOVA demo
        df["Age_Bin"] = pd.cut(
            df["Age"],
            bins=[0, 30, 50, 70, 120],
            labels=["0-30", "31-50", "51-70", "70+"],
        )
        one_way_anova(df, "Glucose", "Age_Bin")

    # Example logistic regression (adjust formula to available columns)
    if "Outcome" in df.columns:
        covs = []
        for c in ["Age", "BMI", "Glucose", "Insulin"]:
            if c in df.columns:
                covs.append(c)
        if covs:
            formula = "Outcome ~ " + " + ".join(covs)
            logistic_regression(df, formula)

    # Survival analysis example if 'time' and 'event' exist
    if "time" in df.columns and "event" in df.columns:
        survival_analysis(df, time_col="time", event_col="event")
