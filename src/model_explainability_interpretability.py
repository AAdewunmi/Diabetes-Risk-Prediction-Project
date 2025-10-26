#!/usr/bin/env python3
"""
src/model_explainability_interpretability.py

Patched explainability & interpretability script for the Diabetes Risk Prediction Project.

This updated version:
- Saves static SHAP PNGs and a *single* self-contained interactive HTML:
    -> reports/explain/shap_local_force_selfcontained.html
  (it NO LONGER writes a separate raw `shap_local_force.html`)
- Wraps heavy SHAP and permutation computations in a sampling step (nsamples default 200)
  and uses joblib.parallel_backend("threading") to reduce macOS loky/resource_tracker warnings
- Keeps LIME support (safe wrapper that converts numpy arrays back to DataFrames for pipelines)
- Accepts either --model or --model_path for backwards compatibility

Usage:
    python src/model_explainability_interpretability.py \
        --model_path reports/models/best_model_rf.joblib \
        --data data/diabetes.csv \
        --out_dir reports/explain \
        --method all \
        --nsamples 200
"""

from typing import Optional, Dict, Any
import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from joblib import parallel_backend

# Optional dependencies
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
    try:
        shap.initjs()
    except Exception:
        pass
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

from sklearn.inspection import permutation_importance

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_model_file(path: str):
    """
    Load a model using joblib. Accepts .joblib/.pkl and other joblib-compatible files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logging.info("Loaded model from %s", path)
    return model


def _save_png_from_matplotlib_out(plot_fn, save_path: str, figsize=(8, 6)):
    """
    Helper to call a plotting function that uses matplotlib (plot_fn draws to plt)
    and save the current figure to save_path.
    """
    try:
        plt.figure(figsize=figsize)
        plot_fn()
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved PNG to %s", save_path)
    except Exception:
        logging.exception("Failed to save PNG to %s", save_path)


def make_selfcontained_shap_html(force_obj: Any, out_html_path: str) -> bool:
    """
    Produce a self-contained HTML file for a SHAP interactive object by embedding
    a JS bundle found inside the installed shap package (if available).

    Returns True if JS was embedded, False otherwise. The HTML file is always written.
    """
    ensure_dir(os.path.dirname(out_html_path) or ".")
    # get the html string for the force object
    try:
        force_html_snippet = force_obj.html()
    except Exception:
        force_html_snippet = str(force_obj)

    embedded_js = ""
    js_found = False
    if SHAP_AVAILABLE:
        try:
            shap_pkg = pathlib.Path(shap.__file__).parent
            # heuristics to find candidate js bundles
            candidate = None
            for sub in ("javascript", "js", "static", "assets"):
                p = shap_pkg.joinpath(sub)
                if p.exists() and p.is_dir():
                    for j in p.rglob("*.js"):
                        candidate = j
                        break
                if candidate:
                    break
            if candidate is None:
                for j in shap_pkg.rglob("*.js"):
                    candidate = j
                    break
            if candidate is not None and candidate.exists():
                try:
                    embedded_js = candidate.read_text(encoding="utf-8")
                    js_found = True
                    logging.info("Found SHAP JS bundle at: %s", str(candidate))
                except Exception:
                    logging.exception("Found shap js file but failed to read it: %s", str(candidate))
        except Exception:
            logging.exception("Error while locating shap JS bundle.")

    if not js_found:
        logging.warning("Could not locate a SHAP JS bundle to embed; interactive HTML may not render everywhere.")

    head = f"<script>{embedded_js}</script>" if js_found else ""
    full_html = f"<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n{head}\n</head>\n<body>\n{force_html_snippet}\n</body>\n</html>"

    try:
        with open(out_html_path, "w", encoding="utf-8") as fh:
            fh.write(full_html)
        logging.info("Wrote self-contained SHAP HTML to %s (embedded_js=%s)", out_html_path, js_found)
    except Exception:
        logging.exception("Failed to write self-contained SHAP HTML to %s", out_html_path)
        return False

    return js_found


def shap_explain(model, X: pd.DataFrame, out_dir: str, nsamples: int = 200) -> Dict[str, Any]:
    """
    Compute SHAP explanations (global + local). Saves:
      - Self-contained interactive HTML: shap_local_force_selfcontained.html
      - Static PNGs: shap_bar.png, shap_beeswarm.png, shap_waterfall_sample0.png

    Heavy computations are sampled (nsamples) and wrapped in joblib.parallel_backend("threading").
    """
    ensure_dir(out_dir)
    results: Dict[str, Any] = {}
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available; skipping SHAP explanation.")
        return results

    try:
        logging.info("Building SHAP explainer and computing values on a sample (nsamples=%d)...", nsamples)
        sample = X.sample(n=min(nsamples, X.shape[0]), random_state=42)

        with parallel_backend("threading"):
            try:
                explainer = shap.Explainer(model, X)
            except Exception:
                explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(sample)

        # Static PNGs
        try:
            def _bar():
                shap.plots.bar(shap_values, show=False)
            _save_png_from_matplotlib_out(_bar, os.path.join(out_dir, "shap_bar.png"), figsize=(8, 6))
            results["shap_bar_png"] = os.path.join(out_dir, "shap_bar.png")
        except Exception:
            logging.exception("Failed to produce SHAP bar PNG.")

        try:
            def _beeswarm():
                shap.plots.beeswarm(shap_values, show=False)
            _save_png_from_matplotlib_out(_beeswarm, os.path.join(out_dir, "shap_beeswarm.png"), figsize=(8, 6))
            results["shap_beeswarm_png"] = os.path.join(out_dir, "shap_beeswarm.png")
        except Exception:
            logging.exception("Failed to produce SHAP beeswarm PNG.")

        try:
            def _waterfall():
                shap.plots.waterfall(shap_values[0], show=False)
            _save_png_from_matplotlib_out(_waterfall, os.path.join(out_dir, "shap_waterfall_sample0.png"), figsize=(8, 6))
            results["shap_waterfall_png"] = os.path.join(out_dir, "shap_waterfall_sample0.png")
        except Exception:
            logging.exception("Failed to produce SHAP waterfall PNG.")

        # Produce ONLY the self-contained interactive HTML (do not write raw html)
        try:
            single_idx = sample.index[0]
            force_obj = shap.plots.force(explainer(sample.loc[[single_idx]]), matplotlib=False)
            selfcontained_path = os.path.join(out_dir, "shap_local_force_selfcontained.html")
            embedded = make_selfcontained_shap_html(force_obj, selfcontained_path)
            results["shap_force_selfcontained_html"] = selfcontained_path
            results["shap_force_selfcontained_embedded_js"] = embedded
            logging.info("Created self-contained SHAP HTML at %s (embedded_js=%s)", selfcontained_path, embedded)
        except Exception:
            logging.exception("Failed to produce interactive SHAP force plot (self-contained HTML).")

        return results

    except Exception as e:
        logging.exception("SHAP explanation failed: %s", e)
        return {}


def permutation_importance_explain(model, X: pd.DataFrame, y: pd.Series, out_dir: str, n_repeats: int = 10) -> Optional[str]:
    """
    Compute permutation importance and save CSV + bar plot. Returns CSV path or None.
    Uses joblib.parallel_backend("threading").
    """
    ensure_dir(out_dir)
    try:
        logging.info("Computing permutation importance (n_repeats=%d) with threading backend...", n_repeats)
        with parallel_backend("threading"):
            r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)

        imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        }).sort_values("importance_mean", ascending=False)

        csv_path = os.path.join(out_dir, "permutation_importance.csv")
        imp_df.to_csv(csv_path, index=False)
        logging.info("Saved permutation importance CSV to %s", csv_path)

        plt.figure(figsize=(10, max(4, 0.25 * len(imp_df))))
        plt.barh(imp_df["feature"][::-1], imp_df["importance_mean"][::-1])
        plt.xlabel("Mean importance (decrease in score)")
        plt.title("Permutation Feature Importance")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "permutation_importance.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved permutation importance plot to %s", plot_path)
        return csv_path
    except Exception as e:
        logging.exception("Permutation importance failed: %s", e)
        return None


def lime_local_explain(model, X: pd.DataFrame, y: pd.Series, out_dir: str, sample_idx: int = 0) -> Optional[str]:
    """
    Produce a single LIME local explanation and save as HTML. Returns path or None.

    Wrapper ensures the model receives a DataFrame with column names (required
    by ColumnTransformer inside a Pipeline). LIME will call predict on numpy arrays,
    so we convert them back to DataFrame before calling model.predict_proba.
    """
    ensure_dir(out_dir)
    if not LIME_AVAILABLE:
        logging.warning("LIME not installed; skipping LIME explanation.")
        return None
    try:
        explainer = LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=[str(c) for c in np.unique(y)],
            mode="classification"
        )

        def predict_fn(arr: np.ndarray):
            df_input = pd.DataFrame(arr, columns=X.columns)
            return model.predict_proba(df_input)

        exp = explainer.explain_instance(X.values[sample_idx], predict_fn, num_features=min(10, X.shape[1]))
        html = exp.as_html()
        html_path = os.path.join(out_dir, "lime_local_explanation.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logging.info("Saved LIME local explanation to %s", html_path)
        return html_path
    except Exception as e:
        logging.exception("LIME explanation failed: %s", e)
        return None


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Common target column names used in the repo: 'Outcome', 'readmitted', 'target'
    """
    for candidate in ("Outcome", "readmitted", "target", "y"):
        if candidate in df.columns:
            return candidate
    return None


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model explainability and interpretability utilities.")
    parser.add_argument("--model", type=str, help="Path to saved model file (joblib/pkl).")
    parser.add_argument("--model_path", type=str, help="Alternative path to saved model file (joblib/pkl).")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--out_dir", type=str, default="./reports/explain", help="Directory to save explainability outputs")
    parser.add_argument("--method", type=str, default="all", choices=["shap", "permutation", "lime", "all"], help="Which explanation method(s) to run")
    parser.add_argument("--nsamples", type=int, default=200, help="Number of samples to use for SHAP computations (keeps run-time reasonable)")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    model_path = args.model_path or args.model
    if model_path is None:
        logging.error("No model file provided. Use --model or --model_path to point to a saved model.")
        raise SystemExit(2)

    try:
        model = load_model_file(model_path)
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        raise

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logging.error("Failed to load data file %s: %s", args.data, e)
        raise

    target_col = find_target_column(df)
    if target_col is None:
        logging.error("No target column found in data. Expected one of 'Outcome','readmitted','target','y'.")
        raise SystemExit(3)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    ensure_dir(args.out_dir)

    methods = [args.method] if args.method != "all" else ["shap", "permutation", "lime"]

    if "shap" in methods:
        logging.info("Running SHAP explanations...")
        shap_res = shap_explain(model, X, out_dir=args.out_dir, nsamples=args.nsamples)
        if not shap_res:
            logging.warning("SHAP run produced no artifacts or failed.")

    if "permutation" in methods:
        logging.info("Running permutation importance...")
        perm_csv = permutation_importance_explain(model, X, y, out_dir=args.out_dir)
        if not perm_csv:
            logging.warning("Permutation importance failed or produced no output.")

    if "lime" in methods:
        logging.info("Running LIME local explanation (if available)...")
        lime_html = lime_local_explain(model, X, y, out_dir=args.out_dir, sample_idx=0)
        if not lime_html:
            logging.warning("LIME explanation failed or produced no output (or LIME not installed).")

    logging.info("Explainability run finished. Check outputs in: %s", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
