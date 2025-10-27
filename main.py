#!/usr/bin/env python3
"""
main.py - pipeline runner for Diabetes Risk Prediction Project

Features:
- Prompts user to choose model for training (default: logreg)
- Runs preprocessing scripts (if present), training, evaluation, explainability
- Captures logs to reports/<script>_log.txt
- Prints a numbered status for each stage (Success/Fail)
"""
import os
import subprocess
import sys
import glob
from typing import List, Optional, Tuple

PROJECT_SRC = "src"
DATA_PATH = "data/diabetes.csv"
REPORTS_DIR = "reports"
MODELS_DIR = os.path.join(REPORTS_DIR, "models")
EVAL_DIR = os.path.join(REPORTS_DIR, "eval")
EXPLAIN_DIR = os.path.join(REPORTS_DIR, "explain")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)


def run_script(script: str, args: Optional[List[str]] = None) -> Tuple[bool, str]:
    if args is None:
        args = []
    script_path = os.path.join(PROJECT_SRC, script)
    log_path = os.path.join(REPORTS_DIR, f"{os.path.splitext(script)[0]}_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        print(f"\n--- Running {script} {' '.join(args)} ---", file=log)
        try:
            res = subprocess.run([sys.executable, script_path] + args, check=True, stdout=log, stderr=log)
            print(f"--- Finished {script} ✅ ---", file=log)
            return True, log_path
        except subprocess.CalledProcessError as e:
            print(f"--- Failed {script} ❌: {e} ---", file=log)
            return False, log_path
        except FileNotFoundError:
            msg = f"Script not found: {script_path}"
            with open(log_path, "a") as log2:
                log2.write(msg + "\n")
            return False, log_path


def find_model_artifact(models_dir: str, base_name: str = "") -> Optional[str]:
    # find joblib or pkl
    for ext in ("joblib", "pkl"):
        matches = glob.glob(os.path.join(models_dir, f"*{base_name}*.{ext}"))
        if matches:
            return matches[0]
    # fallback any
    for ext in ("joblib", "pkl"):
        matches = glob.glob(os.path.join(models_dir, f"*.{ext}"))
        if matches:
            return matches[0]
    return None


def choose_model_interactively() -> str:
    mapping = {"A": "logreg", "B": "rf", "C": "gb", "D": "xgb"}
    print("Which model do you want in pipeline?")
    print("A) logreg – simplest baseline (Default)")
    print("B) rf – Random Forest")
    print("C) gb – Gradient Boosting")
    print("D) xgb – XGBoost (requires xgboost installed)")
    choice = input("Choose either: A / B / C / D (Press Enter): ").strip().upper()
    if not choice:
        choice = "A"
    selected = mapping.get(choice, "logreg")
    print(f"{choice}) {selected} selected as training model.")
    return selected


def choose_existing_script(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if os.path.exists(os.path.join(PROJECT_SRC, c)):
            return c
    return None


def stage_print(i: int, name: str, script: str, success: bool, extra: str = ""):
    status = "Success" if success else "Fail"
    print(f"{i}. {name}: {script} --> {status} {extra}")


def main():
    selected_model = choose_model_interactively()

    statuses = []
    step = 1

    # preprocessing scripts (optional)
    preproc = [
        ("Loading data", ["data_loading.py"]),
        ("Preprocessing", ["data_processing.py"]),
        ("Exploration", ["data_exploration.py"]),
        ("Visualisation", ["data_visualisation.py"]),
        ("Statistical analysis", ["statistical_analysis.py"])
    ]

    for label, scripts in preproc:
        for script in scripts:
            if os.path.exists(os.path.join(PROJECT_SRC, script)):
                ok, log = run_script(script, ["--data", DATA_PATH] )
                stage_print(step, label, script, ok, f"(log: {log})")
                statuses.append((label, script, ok, log))
                step += 1

    # training (prefer refactored src/model_training.py)
    training_script = choose_existing_script(["model_training.py", "model_training_refactored.py"])
    if training_script is None:
        print("No training script found in src/. Aborting training stage.")
    else:
        train_args = ["--data", DATA_PATH, "--model", selected_model, "--out_dir", REPORTS_DIR]
        ok, log = run_script(training_script, train_args)
        stage_print(step, "Training", training_script, ok, f"(model: {selected_model}, log: {log})")
        statuses.append(("Training", training_script, ok, log))
        step += 1

    # find artifact
    model_path = find_model_artifact(MODELS_DIR)
    if model_path is None:
        print("No model file found in reports/models/. Attempting common names (best_model_*.joblib)...")
        model_path = find_model_artifact(MODELS_DIR, base_name="best_model")
    if model_path:
        print(f"Model artifact detected: {model_path}")
    else:
        print("No trained model artifact detected. Evaluation and explainability will be skipped.")
    
    # evaluation
    eval_script = choose_existing_script(["model_evaluation.py", "model_evaluation_refactored.py"])
    if eval_script and model_path:
        # determine flag expected: prefer --model_path for refactored, else --model
        args = ["--data", DATA_PATH, "--out_dir", EVAL_DIR]
        if os.path.exists(os.path.join(PROJECT_SRC, "model_evaluation_refactored.py")):
            args.insert(2, "--model_path")
            args.insert(3, model_path)
        else:
            args.insert(2, "--model")
            args.insert(3, model_path)
        ok, log = run_script(eval_script, args)
        stage_print(step, "Evaluation", eval_script, ok, f"(log: {log})")
        statuses.append(("Evaluation", eval_script, ok, log))
        step += 1
    else:
        if not model_path:
            stage_print(step, "Evaluation", "model_evaluation", False, "(no model artifact)")
            step += 1

    # explainability
    explain_script = choose_existing_script(["model_explainability_interpretability.py", "model_explainability.py"])
    if explain_script and model_path:
        args = ["--data", DATA_PATH, "--out_dir", EXPLAIN_DIR, "--method", "all"]
        # prefer model_path arg
        args.insert(0, "--model_path")
        args.insert(1, model_path)
        ok, log = run_script(explain_script, args)
        stage_print(step, "Explainability", explain_script, ok, f"(log: {log})")
        statuses.append(("Explainability", explain_script, ok, log))
        step += 1
    else:
        if not model_path:
            stage_print(step, "Explainability", "model_explainability", False, "(no model artifact)")
            step += 1

    print("\n✅ Pipeline operations completed. Check reports/ for logs and generated artifacts.")
    print("Summary:")
    for i, (label, script, ok, log) in enumerate(statuses, start=1):
        status = "Success" if ok else "Fail"
        print(f"{i}. {label}: {script} --> {status} (log: {log})")


if __name__ == "__main__":
    main()
