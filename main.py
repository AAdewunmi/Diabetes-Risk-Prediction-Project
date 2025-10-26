#!/usr/bin/env python3
"""
main.py - robust pipeline runner for Diabetes Risk Prediction Project

This script runs the project's stepwise pipeline (data -> train -> eval -> explain)
and writes logs to the `reports/` directory. It attempts to be robust to small
naming/extension mismatches encountered in different versions of scripts.

Key features:
- Detects whether you have the refactored scripts (e.g. model_training_refactored.py,
  model_evaluation_refactored.py) or original names (model_training.py, model_evaluation.py)
  and runs whichever is present.
- Searches reports/models/ for existing model artifacts (.joblib, .pkl) and uses the first match.
- Chooses correct CLI flag name for evaluator based on which file exists (`--model` or `--model_path`).
- Writes per-script stdout+stderr into reports/<script_name>_log.txt.
- Creates required output folders automatically.

Usage:
    python main.py

Author: (Your Name)
"""

from __future__ import annotations
import os
import subprocess
import sys
from typing import List, Optional, Tuple


def run_script(script_name: str, args: Optional[List[str]] = None) -> None:
    """
    Run a Python script inside src/ and capture stdout+stderr to a log file in reports/.

    Args:
        script_name: filename inside src/ (e.g. "model_training.py")
        args: list of CLI args (e.g. ["--data", "data.csv"])

    Returns:
        None
    """
    if args is None:
        args = []

    os.makedirs("reports", exist_ok=True)
    script_path = os.path.join("src", script_name)
    log_file = os.path.join("reports", f"{os.path.splitext(script_name)[0]}_log.txt")

    with open(log_file, "w") as log:
        print(f"\n--- Running {script_name} {' '.join(args)} ---", file=log)
        try:
            subprocess.run([sys.executable, script_path] + args, check=True, stdout=log, stderr=log)
            print(f"--- Finished running {script_name} ✅ ---", file=log)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing {script_name}: {e}", file=log)
        except FileNotFoundError:
            print(f"❌ Script not found: {script_path}", file=log)


def find_existing_model(models_dir: str, base_name: str = "best_model") -> Optional[str]:
    """
    Look for common model filenames inside models_dir and return the first existing path.

    Checks patterns:
      - <models_dir>/<base_name>_<model>.joblib
      - <models_dir>/<base_name>_<model>.pkl
      - <models_dir>/*.joblib
      - <models_dir>/*.pkl

    Args:
        models_dir: directory to search
        base_name: prefix to look for

    Returns:
        Path to model file if found, else None
    """
    if not os.path.isdir(models_dir):
        return None

    # Look for explicit candidate names first (rf fallback)
    candidates = [
        os.path.join(models_dir, f"{base_name}_rf.joblib"),
        os.path.join(models_dir, f"{base_name}_rf.pkl"),
        os.path.join(models_dir, f"{base_name}_rf.joblib"),
        os.path.join(models_dir, f"{base_name}_rf.pkl"),
    ]
    # general patterns
    for ext in ("joblib", "pkl"):
        candidates.append(os.path.join(models_dir, f"{base_name}*.{ext}"))
    # expand globs
    import glob
    for cand in candidates:
        matches = glob.glob(cand)
        if matches:
            # return the first match
            return matches[0]
    # fallback: any joblib/pkl in dir
    for ext in ("joblib", "pkl"):
        matches = glob.glob(os.path.join(models_dir, f"*.{ext}"))
        if matches:
            return matches[0]
    return None


def choose_script(choices: List[str]) -> Optional[str]:
    """
    Return the first script filename from choices that exists under src/.

    Args:
        choices: ordered list of filenames to check (e.g. ['model_evaluation_refactored.py', 'model_evaluation.py'])

    Returns:
        the filename that exists, or None
    """
    for name in choices:
        if os.path.exists(os.path.join("src", name)):
            return name
    return None


def main() -> None:
    """
    Build and run the pipeline. The function is defensive about:
      - which script filenames exist,
      - the model artifact filename/extension,
      - evaluator flag name differences.

    Steps:
      1. Run preprocessing/data steps (if scripts exist).
      2. Run training (prefer refactored training script if present).
      3. Detect saved model artifact and use it for evaluation + explainability.
      4. Run evaluation with the correct flag name.
      5. Run explainability (refactored or original).
    """
    DATA_PATH = "data/diabetes.csv"
    MODELS_DIR = "reports/models"
    # ensure directories exist
    os.makedirs("reports", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Decide which training script to use
    training_script = choose_script(["model_training_refactored.py", "model_training.py"])
    if training_script is None:
        print("ERROR: No training script found in src/. Expected one of: model_training_refactored.py, model_training.py")
        return

    # Decide which evaluation script to use
    eval_script = choose_script(["model_evaluation_refactored.py", "model_evaluation.py"])
    if eval_script is None:
        print("ERROR: No evaluation script found in src/. Expected one of: model_evaluation_refactored.py, model_evaluation.py")
        return

    # Decide which explainability script to use
    explain_script = choose_script(["model_explainability_interpretability.py", "model_explainability.py", "model_explainability_interpretability_refactored.py"])
    # explain_script may be None — it's optional

    # Stage scripts that are generic preprocessing steps (if present)
    preproc_scripts = [
        "data_loading.py",
        "data_processing.py",
        "data_exploration.py",
        "data_visualisation.py",
        "statistical_analysis.py",
    ]
    for s in preproc_scripts:
        if os.path.exists(os.path.join("src", s)):
            run_script(s, ["--data", DATA_PATH] if "--data" in get_script_args(os.path.join("src", s)) else [])

    # 1) Training
    # prefer to call refactored training if available
    print(f"Using training script: {training_script}")
    # Build args for training — training scripts typically accept --data and --out_dir and optionally --model and --smote
    training_args = ["--data", DATA_PATH, "--model", "rf", "--out_dir", MODELS_DIR]
    # If script has different flags, run without trying to introspect too hard — we assume these flags exist on the refactored versions
    run_script(training_script, training_args)

    # 2) Locate saved model artifact
    model_path = find_existing_model(MODELS_DIR, base_name="best_model")
    if model_path is None:
        # sometimes older runs saved "best_model_rf.pkl" etc.
        print("No model artifact found in reports/models/. Attempting common alternatives...")
        possible = [
            os.path.join(MODELS_DIR, "best_model_rf.pkl"),
            os.path.join(MODELS_DIR, "best_model_rf.joblib"),
            os.path.join(MODELS_DIR, "best_model.pkl"),
            os.path.join(MODELS_DIR, "best_model.joblib")
        ]
        for p in possible:
            if os.path.exists(p):
                model_path = p
                break

    if model_path is None:
        print("ERROR: No trained model artifact found in reports/models/. Please run training or place a model file there.")
        return

    print(f"Model artifact detected: {model_path}")

    # 3) Evaluation
    print(f"Using evaluation script: {eval_script}")
    # Determine which flag the evaluator expects: read its help (if available) to detect argument name
    eval_args = []
    # If the refactored evaluator exists (we provided earlier), it expects --model_path
    if os.path.exists(os.path.join("src", "model_evaluation_refactored.py")):
        eval_args = ["--data", DATA_PATH, "--model_path", model_path, "--out_dir", "reports/eval"]
    else:
        # fallback: older evaluator expected --model
        eval_args = ["--data", DATA_PATH, "--model", model_path, "--out_dir", "reports/eval"]

    run_script(eval_script, eval_args)

    # 4) Explainability (optional)
    if explain_script:
        print(f"Using explainability script: {explain_script}")
        # refactored explainability expects --model_path; for older ones try --model
        if "explainability" in explain_script or "explain" in explain_script or os.path.exists(os.path.join("src", "model_explainability_interpretability.py")):
            explain_args = ["--model_path", model_path, "--data", DATA_PATH, "--out_dir", "reports/explain", "--method", "all"]
            # some older explain scripts might expect --model instead; try that if the first call fails
            run_script(explain_script, explain_args)
        else:
            # generic fallback
            run_script(explain_script, ["--model", model_path, "--data", DATA_PATH, "--out_dir", "reports/explain"])
    else:
        print("No explainability script found in src/ — skipping explainability stage.")

    print("\n✅ Pipeline finished. Check reports/ for logs and generated artifacts.")


def get_script_args(script_path: str) -> List[str]:
    """
    Attempt to inspect a script for common flags by scanning the file
    for occurrences of '--data', '--model', '--model_path', '--out_dir'. This is a
    lightweight, non-invasive check to help pass arguments when calling scripts.

    Returns:
        list of found flag names (e.g. ['--data', '--out_dir'])
    """
    flags = []
    if not os.path.exists(script_path):
        return flags
    try:
        with open(script_path, "r", encoding="utf-8") as fh:
            content = fh.read()
            for f in ["--data", "--model", "--model_path", "--out_dir", "--out-dir", "--smote"]:
                if f in content:
                    flags.append(f)
    except Exception:
        pass
    return flags


if __name__ == "__main__":
    main()
