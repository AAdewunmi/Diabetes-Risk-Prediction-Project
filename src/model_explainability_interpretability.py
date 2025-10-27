#!/usr/bin/env python3
"""
src/model_explainability_interpretability.py

Explainability & interpretability script.

- Produces only a single self-contained interactive SHAP HTML: shap_local_force_selfcontained.html
- Produces static SHAP PNGs (bar, beeswarm, waterfall)
- Uses sampling (nsamples default 200) to keep runtime reasonable
- Wraps heavy work in joblib.parallel_backend("threading") to reduce macOS resource-tracker noise
- Supports LIME (optional)
- Accepts --model or --model_path for compatibility
"""
import argparse
import logging
import os
import pathlib
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.inspection import permutation_importance

# optional libs
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_model_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = joblib.load(path)
    logging.info("Loaded model from %s", path)
    return m


def _save_png(plot_fn, out_path: str, figsize=(8, 6)):
    try:
        plt.figure(figsize=figsize)
        plot_fn()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved PNG to %s", out_path)
    except Exception:
        logging.exception("Failed to save PNG to %s", out_path)


def make_selfcontained_shap_html(force_obj: Any, out_html_path: str) -> bool:
    """
    Create a self-contained HTML by embedding a shap JS bundle when available.
    Returns True if JS was embedded.
    """
    ensure_dir(os.path.dirname(out_html_path) or ".")
    # get fragment
    try:
        frag = force_obj.html()
    except Exception:
        frag = str(force_obj)

    embedded_js = ""
    js_found = False
    if SHAP_AVAILABLE:
        try:
            shap_pkg = pathlib.Path(shap.__file__).parent
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
                embedded_js = candidate.read_text(encoding="utf-8")
                js_found = True
                logging.info("Found shap JS bundle at %s", candidate)
        except Exception:
            logging.exception("Error locating shap JS bundle.")

    if not js_found:
        logging.warning(
            "No shap JS bundle embedded; HTML may not show interactive visualizations in some contexts."
        )

    head = f"<script>{embedded_js}</script>" if js_found else ""
    full = f"<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n{head}\n</head>\n<body>\n{frag}\n</body>\n</html>"

    try:
        with open(out_html_path, "w", encoding="utf-8") as fh:
            fh.write(full)
        logging.info(
            "Wrote self-contained SHAP HTML to %s (embedded_js=%s)",
            out_html_path,
            js_found,
        )
    except Exception:
        logging.exception(
            "Failed to write self-contained SHAP HTML to %s", out_html_path
        )
        return False
    return js_found


def shap_explain(
    model, X: pd.DataFrame, out_dir: str, nsamples: int = 200
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    results: Dict[str, Any] = {}
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available; skipping.")
        return results

    try:
        sample = X.sample(n=min(nsamples, X.shape[0]), random_state=42)
        logging.info(
            "Computing SHAP values on a sample of %d rows (threading backend)...",
            sample.shape[0],
        )
        with parallel_backend("threading"):
            try:
                explainer = shap.Explainer(model, X)
            except Exception:
                explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(sample)

        # static bar
        try:

            def bar():
                shap.plots.bar(shap_values, show=False)

            _save_png(bar, os.path.join(out_dir, "shap_bar.png"))
            results["shap_bar_png"] = os.path.join(out_dir, "shap_bar.png")
        except Exception:
            logging.exception("SHAP bar failed")

        # beeswarm
        try:

            def bees():
                shap.plots.beeswarm(shap_values, show=False)

            _save_png(bees, os.path.join(out_dir, "shap_beeswarm.png"))
            results["shap_beeswarm_png"] = os.path.join(out_dir, "shap_beeswarm.png")
        except Exception:
            logging.exception("SHAP beeswarm failed")

        # waterfall for first sample
        try:

            def wf():
                shap.plots.waterfall(shap_values[0], show=False)

            _save_png(wf, os.path.join(out_dir, "shap_waterfall_sample0.png"))
            results["shap_waterfall_png"] = os.path.join(
                out_dir, "shap_waterfall_sample0.png"
            )
        except Exception:
            logging.exception("SHAP waterfall failed")

        # interactive force self-contained HTML (ONLY this HTML)
        try:
            single_idx = sample.index[0]
            force_obj = shap.plots.force(
                explainer(sample.loc[[single_idx]]), matplotlib=False
            )
            out_html = os.path.join(out_dir, "shap_local_force_selfcontained.html")
            embedded = make_selfcontained_shap_html(force_obj, out_html)
            results["shap_force_selfcontained_html"] = out_html
            results["shap_force_selfcontained_embedded_js"] = embedded
        except Exception:
            logging.exception("SHAP interactive force failed")

        return results
    except Exception:
        logging.exception("SHAP explanation failed")
        return {}


def permutation_importance_explain(
    model, X: pd.DataFrame, y: pd.Series, out_dir: str, n_repeats: int = 10
) -> Optional[str]:
    ensure_dir(out_dir)
    try:
        logging.info("Running permutation importance (threading backend)...")
        with parallel_backend("threading"):
            r = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
        imp_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        csv_path = os.path.join(out_dir, "permutation_importance.csv")
        imp_df.to_csv(csv_path, index=False)
        logging.info("Saved permutation CSV to %s", csv_path)
        plt.figure(figsize=(10, max(4, 0.25 * len(imp_df))))
        plt.barh(imp_df["feature"][::-1], imp_df["importance_mean"][::-1])
        plt.xlabel("Mean importance (decrease)")
        plt.title("Permutation Importance")
        plot_path = os.path.join(out_dir, "permutation_importance.png")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logging.info("Saved permutation importance plot to %s", plot_path)
        return csv_path
    except Exception:
        logging.exception("Permutation importance failed")
        return None


def lime_local_explain(
    model, X: pd.DataFrame, y: pd.Series, out_dir: str, sample_idx: int = 0
) -> Optional[str]:
    ensure_dir(out_dir)
    if not LIME_AVAILABLE:
        logging.warning("LIME not available; skipping.")
        return None
    try:
        explainer = LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=[str(c) for c in np.unique(y)],
            mode="classification",
        )

        def predict_fn(arr):
            df_in = pd.DataFrame(arr, columns=X.columns)
            return model.predict_proba(df_in)

        exp = explainer.explain_instance(
            X.values[sample_idx], predict_fn, num_features=min(10, X.shape[1])
        )
        html = exp.as_html()
        path = os.path.join(out_dir, "lime_local_explanation.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logging.info("Saved LIME explanation to %s", path)
        return path
    except Exception:
        logging.exception("LIME explanation failed")
        return None


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("Outcome", "readmitted", "target", "y"):
        if c in df.columns:
            return c
    return None


def parse_cli_args():
    p = argparse.ArgumentParser(description="Model explainability utilities")
    p.add_argument("--model", type=str, help="Path to model (backwards compat)")
    p.add_argument("--model_path", type=str, help="Path to model (preferred)")
    p.add_argument("--data", type=str, required=True, help="Path to CSV data")
    p.add_argument(
        "--out_dir", type=str, default="reports/explain", help="Output folder"
    )
    p.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["shap", "permutation", "lime", "all"],
    )
    p.add_argument(
        "--nsamples", type=int, default=200, help="Samples to use for SHAP calculations"
    )
    return p.parse_args()


def main():
    args = parse_cli_args()
    model_path = args.model_path or args.model
    if model_path is None:
        logging.error("No model provided (--model or --model_path).")
        raise SystemExit(2)

    model = load_model_file(model_path)
    df = pd.read_csv(args.data)
    target = find_target_column(df)
    if target is None:
        logging.error("No target column found (Outcome/readmitted/target/y).")
        raise SystemExit(3)

    X = df.drop(columns=[target])
    y = df[target]

    ensure_dir(args.out_dir)
    methods = [args.method] if args.method != "all" else ["shap", "permutation", "lime"]

    if "shap" in methods:
        logging.info("Running SHAP...")
        res = shap_explain(model, X, out_dir=args.out_dir, nsamples=args.nsamples)
        if not res:
            logging.warning("SHAP produced no artifacts or failed.")

    if "permutation" in methods:
        logging.info("Running permutation importance...")
        csv = permutation_importance_explain(model, X, y, out_dir=args.out_dir)
        if not csv:
            logging.warning("Permutation importance failed.")

    if "lime" in methods:
        logging.info("Running LIME local explanation...")
        html = lime_local_explain(model, X, y, out_dir=args.out_dir)
        if not html:
            logging.warning("LIME produced no output or failed.")

    logging.info(
        "Explainability run finished. Outputs: %s", os.path.abspath(args.out_dir)
    )


if __name__ == "__main__":
    main()
